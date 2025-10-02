# Copyright 2025 Fu-Yun Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerateOutput
from ..blip3o_arch import blip3oMetaModel, blip3oMetaForCausalLM
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLModel, Qwen2_5_VLForConditionalGeneration
from ...constants import UND_IMAGE_TOKEN_IDX

# SDE-related import for probabilistic sampling
from .fm_step_prob import sde_step_with_logprob 

# Diffusion-related imports
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import numpy_to_pil
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


class blip3oQwenConfig(Qwen2_5_VLConfig):
    """
    Configuration class for blip3oQwenModel. Inherits from Qwen2_5_VLConfig.
    """
    model_type = "blip3o_qwen_inference"


class blip3oQwenModel(blip3oMetaModel, Qwen2_5_VLModel):
    """
    Core model class integrating blip3o meta features with the Qwen2_5_VL model.
    """
    config_class = blip3oQwenConfig

    def __init__(self, config: Qwen2_5_VLConfig):
        super(blip3oQwenModel, self).__init__(config)


class blip3oQwenForInferenceLM(Qwen2_5_VLForConditionalGeneration, blip3oMetaForCausalLM):
    """
    Conditional generation model for multimodal inference, including image generation.
    It combines the Qwen VL head with the blip3o architecture.
    """
    config_class = blip3oQwenConfig

    def __init__(self, config):
        Qwen2_5_VLForConditionalGeneration.__init__(self, config)
        config.model_type = "blip3o_qwen_inference"

        self.model = blip3oQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        """Returns the core blip3oQwenModel."""
        return self.model

    def _prepare_img_hidden_states(
        self,
        text: List[str],
        tokenizer: AutoTokenizer,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Processes text and optional image inputs to generate the final hidden states
        (image conditioning embeddings) needed for the DiT model.

        Args:
            text (List[str]): Input text prompts.
            tokenizer (AutoTokenizer): Tokenizer for text.
            pixel_values (Optional[torch.Tensor]): Input image pixels for image-to-image tasks.
            image_grid_thw (Optional[torch.Tensor]): Grid information for the visual encoder.

        Returns:
            Tuple[torch.Tensor, bool]: 
                - img_hidden_states (torch.Tensor): The final down-projected embeddings 
                  to condition the DiT model.
                - is_image_edit (bool): True if pixel_values was provided (i.e., I2I task).
        """
        N_QUERY = self.get_n_query()            
        is_image_edit = pixel_values is not None

        # 1. Tokenize text and get embeddings
        inputs = tokenizer(text, padding="longest", return_tensors="pt")
        device = self.get_model().device
        attention_mask = inputs.attention_mask.to(device)
        input_ids = inputs.input_ids.to(device) 
        text_embeds = self.get_model().embed_tokens(input_ids)

        # 2. Inject image embeddings if pixel_values is present (I2I)
        if is_image_edit:
            und_image_idx = (input_ids == UND_IMAGE_TOKEN_IDX)
            pixel_values = pixel_values.type(self.visual.dtype)
            und_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            # Replace placeholder tokens with visual embeddings
            assert und_image_idx.sum().item() == und_image_embeds.shape[0], "Mismatch between UND_IMAGE_TOKENs and input images."
            text_embeds[und_image_idx] = und_image_embeds.to(text_embeds.device)[:und_image_idx.sum(), :]
            latent_queries = self.get_model().i2i_queries.repeat(text_embeds.shape[0], 1, 1)
        else:
            # T2I task uses text-to-image queries
            latent_queries = self.get_model().t2i_queries.repeat(text_embeds.shape[0], 1, 1)

        # 3. Concatenate text embeddings with latent queries (conditioning tokens)
        text_embeds = torch.cat([text_embeds, latent_queries], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(latent_queries[:, :, 0])], dim=1)

        # 4. Pass through the model to get final hidden states
        outputs = self.model(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Extract hidden states corresponding to the latent queries
        hidden_states = outputs.hidden_states[-1][:,-N_QUERY:,:]
        # Down-project to match DiT encoder hidden size
        img_hidden_states = self.get_model().down_projector(hidden_states)
        
        return img_hidden_states, is_image_edit

    @torch.no_grad()
    def generate_image(
        self,
        text: List[str],
        tokenizer: AutoTokenizer,
        pixel_values: Optional[torch.Tensor] = None,
        ref_latents: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        max_var: Optional[float] = None, # Not used in current implementation but kept for signature consistency
        diffusion_kwargs: Optional[Dict] = dict(guidance_scale = 5.0, num_inference_steps=50),
        use_sde: bool = False,
    ):  
        """
        Unified image generation function, handling both T2I and I2I tasks, 
        and optionally enables SDE (stochastic differential equation) sampling.

        Args:
            text (List[str]): Input text prompts.
            tokenizer (AutoTokenizer): Tokenizer.
            pixel_values (Optional[torch.Tensor]): Input image pixels for I2I.
            ref_latents (Optional[torch.Tensor]): Reference latents for conditioning the UNet/DiT.
            image_grid_thw (Optional[torch.Tensor]): Grid information for the visual encoder.
            diffusion_kwargs (Optional[Dict]): Parameters for the diffusion process (e.g., guidance_scale).
            use_sde (bool): If True, uses SDE sampling (with log-prob calculation); otherwise, standard sampling.

        Returns:
            Union[List[Image.Image], torch.Tensor, Tuple]: Generated image samples. 
                If use_sde is True, returns (samples, log_probs, prev_latents, pred_latents, ts, [noisy_ref_latents_lst]).
        """
        scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
        
        # 1. Prepare image hidden states for conditioning
        img_hidden_states, is_image_edit = self._prepare_img_hidden_states(
            text, tokenizer, pixel_values, image_grid_thw
        )

        # 2. Call the appropriate sampling function
        if is_image_edit:
            output_img = self._sample_images_edit(
                img_hidden_states, scheduler, ref_latents=ref_latents, use_sde=use_sde, **diffusion_kwargs
            )
        else:    
            output_img = self._sample_images(
                img_hidden_states, scheduler, ref_latents=ref_latents, use_sde=use_sde, **diffusion_kwargs
            )

        return output_img

    def _sample_images_edit(
        self,
        img_hidden_states,
        scheduler,
        guidance_scale_ref: float = 1.0,
        guidance_scale: float = 5.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1,
        ref_latents: Optional[torch.Tensor] = None,
        return_tensor=False,
        use_sde: bool = False,
        **kwargs,
    ):
        """
        Performs Image-to-Image (I2I) generation using triple Conditional Guidance (uncond, ref-image, text).
        Supports both standard sampling and SDE sampling.
        """
        device = img_hidden_states.device
        dtype = img_hidden_states.dtype

        # 1. Prepare inputs for Triple CFG
        img_hidden_states_null = torch.zeros_like(img_hidden_states, device=device, dtype=dtype)
        # Structure: [uncond_text, ref_image_cond, text_cond]
        hidden_states_input = torch.cat(
            [img_hidden_states_null, img_hidden_states_null, img_hidden_states], 0
        )
        hidden_states_input = hidden_states_input.repeat_interleave(num_images_per_prompt, dim=0)

        # 2. Initialize Latents
        latent_size = self.get_model().dit.config.sample_size
        latent_channels = self.get_model().dit.config.in_channels
        b = ref_latents.shape[0] if ref_latents is not None else img_hidden_states.shape[0]
        c, h, w = (ref_latents.shape[1:]) if ref_latents is not None else (latent_channels, latent_size, latent_size)

        latents = randn_tensor(
            shape=(b * num_images_per_prompt, c, h, w),
            generator=generator,
            device=device,
            dtype=dtype,
        )

        # 3. Prepare Noisy Reference Latents for conditioning (I2I)
        noisy_ref_latents = ref_latents * 0.8 + torch.randn_like(ref_latents) * 0.2 if ref_latents is not None else None
        noisy_ref_latents = noisy_ref_latents.repeat_interleave(num_images_per_prompt, dim=0) if ref_latents is not None else None
        # Structure: [uncond_ref, ref_image_ref, text_ref]
        ref_latents_input = torch.cat([torch.zeros_like(noisy_ref_latents), noisy_ref_latents, noisy_ref_latents]) if ref_latents is not None else None


        scheduler.set_timesteps(num_inference_steps)
        
        # SDE variables initialization
        prev_latents, pred_latents, ts, log_probs, noisy_ref_latents_lst = [], [], [], [], []

        # 4. Denoising Loop
        for idx, t in enumerate(scheduler.timesteps):
            latent_model_input = latents.repeat(3, 1, 1, 1)

            if hasattr(scheduler, "scale_model_input"):
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            hidden_attention_mask = torch.ones(
                (hidden_states_input.shape[0], hidden_states_input.shape[1])
            ).to(hidden_states_input.device, dtype=hidden_states_input.dtype)

            # Forward pass through DiT
            noise_pred = self.get_model().dit(
                hidden_states=latent_model_input,
                ref_hidden_states=ref_latents_input,
                encoder_hidden_states=hidden_states_input,
                encoder_attention_mask=hidden_attention_mask,
                timestep=t.unsqueeze(0).expand(latent_model_input.shape[0]).to(latent_model_input.device),
                return_dict=False,
            )[0]

            # Triple CFG: Split noise predictions
            noise_pred_uncond, noise_pred_ref, noise_pred_text = noise_pred.chunk(3)
            
            # Apply guidance formula: pred = uncond + ref_scale*(ref - uncond) + text_scale*(text - ref)
            noise_pred = (
                noise_pred_uncond
                + guidance_scale_ref * (noise_pred_ref - noise_pred_uncond)
                + guidance_scale * (noise_pred_text - noise_pred_ref)
            )
            
            # Step: standard vs. SDE
            if use_sde:
                prev_latents.append(latents.unsqueeze(1))
                latents, log_prob, _, _ = sde_step_with_logprob(
                    scheduler, noise_pred.float(), t.unsqueeze(0), latents.float(),
                )
                log_probs.append(log_prob.unsqueeze(1))
                pred_latents.append(latents.unsqueeze(1))
                ts.append(t.unsqueeze(0).repeat(len(log_prob)).unsqueeze(1))
                noisy_ref_latents_lst.append(noisy_ref_latents.unsqueeze(1))
                latents = latents.to(dtype=hidden_states_input.dtype)
            else:
                latents = scheduler.step(noise_pred, t, latents).prev_sample


        # 5. Decode and return
        samples = self.decode_latents(latents, return_tensor=return_tensor)
        
        if use_sde:
            # Flatten SDE outputs
            prev_latents = torch.cat(prev_latents, dim=1).flatten(0, 1)
            log_probs = torch.cat(log_probs, dim=1).flatten(0, 1)
            pred_latents = torch.cat(pred_latents, dim=1).flatten(0, 1)
            ts = torch.cat(ts, dim=1).flatten(0, 1)
            noisy_ref_latents_lst = torch.cat(noisy_ref_latents_lst, dim=1).flatten(0, 1)
            
            return samples, log_probs, prev_latents, pred_latents, ts, noisy_ref_latents_lst
        else:
            return samples


    def _sample_images(
        self,
        img_hidden_states,
        scheduler,
        guidance_scale: float = 3.0, 
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1,
        ref_latents: Optional[torch.Tensor] = None, # Primarily used for ControlNet-like conditioning if needed
        return_tensor=False,
        use_sde: bool = False,
        **kwargs,
    ):
        """
        Performs Text-to-Image (T2I) generation using standard Conditional Guidance (uncond, text).
        Supports both standard sampling and SDE sampling.
        """
        device = img_hidden_states.device
        dtype = img_hidden_states.dtype

        # 1. Prepare inputs for Double CFG
        img_hidden_states_null = torch.zeros_like(img_hidden_states, device=device, dtype=dtype)
        # Structure: [uncond, text_cond]
        img_hidden_states_input = torch.cat([img_hidden_states_null, img_hidden_states], 0)
        img_hidden_states_input = img_hidden_states_input.repeat_interleave(num_images_per_prompt, dim=0)


        # 2. Initialize Latents
        latent_size = self.get_model().dit.config.sample_size
        latent_channels = self.get_model().dit.config.in_channels
        b = ref_latents.shape[0] if ref_latents is not None else img_hidden_states.shape[0]
        c, h, w = (ref_latents.shape[1:]) if ref_latents is not None else (latent_channels, latent_size, latent_size)

        latents = randn_tensor(
            shape=(b * num_images_per_prompt, c, h, w),
            generator=generator,
            device=device,
            dtype=dtype,
        )

        # 3. Prepare Reference Latents (if provided, for external conditioning)
        ref_latents = ref_latents * 0.8 + torch.randn_like(ref_latents) * 0.2 if ref_latents is not None else None
        ref_latents_input = ref_latents.repeat_interleave(num_images_per_prompt, dim=0) if ref_latents is not None else None
        # Must be repeated twice for CFG
        ref_latents_input = ref_latents_input.repeat(2, 1, 1, 1) if ref_latents_input is not None else None
        
        scheduler.set_timesteps(num_inference_steps)
        
        # SDE variables initialization
        prev_latents, pred_latents, ts, log_probs = [], [], [], []

        # 4. Denoising Loop
        for idx, t in enumerate(scheduler.timesteps):
            latent_model_input = latents.repeat(2, 1, 1, 1)

            if hasattr(scheduler, "scale_model_input"):
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            img_hidden_attention_mask = torch.ones(
                (img_hidden_states_input.shape[0], img_hidden_states_input.shape[1])).to(
                img_hidden_states_input.device, dtype=img_hidden_states_input.dtype)
                
            # Forward pass through DiT
            noise_pred = self.get_model().dit(
                hidden_states=latent_model_input,
                ref_hidden_states=ref_latents_input,
                encoder_hidden_states=img_hidden_states_input,
                encoder_attention_mask=img_hidden_attention_mask,
                timestep=t.unsqueeze(0).expand(latent_model_input.shape[0]).to(latent_model_input.device),
                return_dict=False,
            )[0]
            
            # Double CFG: Split noise predictions and apply guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Step: standard vs. SDE
            if use_sde:
                prev_latents.append(latents.unsqueeze(1))
                latents, log_prob, _, _ = sde_step_with_logprob(
                    scheduler, noise_pred.float(), t.unsqueeze(0), latents.float(),
                )
                log_probs.append(log_prob.unsqueeze(1))
                pred_latents.append(latents.unsqueeze(1))
                ts.append(t.unsqueeze(0).repeat(len(log_prob)).unsqueeze(1))
                latents = latents.to(dtype=img_hidden_states_input.dtype)
            else:
                latents = scheduler.step(noise_pred, t, latents).prev_sample


        # 5. Decode and return
        samples = self.decode_latents(latents, return_tensor=return_tensor)
        
        if use_sde:
            # Flatten SDE outputs
            prev_latents = torch.cat(prev_latents, dim=1).flatten(0, 1)
            log_probs = torch.cat(log_probs, dim=1).flatten(0, 1)
            pred_latents = torch.cat(pred_latents, dim=1).flatten(0, 1)
            ts = torch.cat(ts, dim=1).flatten(0, 1)
            
            return samples, log_probs, prev_latents, pred_latents, ts
        else:
            return samples
            
            
    def decode_latents(self, latents, normalize=True, return_tensor=False):
        """
        Decodes VAE latents into pixel space.

        Args:
            latents (torch.Tensor): The latent tensor from the diffusion process.
            normalize (bool): If True, normalizes samples to [0, 1].
            return_tensor (bool): If True, returns a torch.Tensor; otherwise, returns PIL Images.

        Returns:
            Union[torch.Tensor, List[Image.Image]]: Decoded images.
        """
        vae = self.get_model().gen_vision_tower.autoencoder
        # Apply scaling factor
        latents = latents / vae.config.scaling_factor
        if hasattr(vae.config, "shift_factor") and vae.config.shift_factor is not None:
            latents = latents + vae.config.shift_factor
        
        samples = vae.decode(latents).sample

        # Post-processing and normalization
        if normalize:
            samples = (samples / 2 + 0.5).clamp(0, 1)
        else:
            samples = samples.clamp(-1, 1)
            
        if return_tensor:
            return samples
            
        # Convert to PIL images
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
        samples = numpy_to_pil(samples)
        return samples


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        """
        Prepares model inputs for text generation, including handling multimodal inputs.
        """
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

# Register configurations for Hugging Face compatibility
AutoConfig.register("blip3o_qwen_inference", blip3oQwenConfig)
AutoModelForCausalLM.register(blip3oQwenConfig, blip3oQwenForInferenceLM)