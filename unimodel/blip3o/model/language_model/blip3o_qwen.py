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
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..blip3o_arch import blip3oMetaModel, blip3oMetaForCausalLM
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLModel, Qwen2_5_VLForConditionalGeneration

from ...constants import UND_IMAGE_TOKEN_IDX, T2I_TOKEN_IDX, I2I_TOKEN_IDX



class blip3oQwenConfig(Qwen2_5_VLConfig):
    """
    Configuration class for blip3oQwenModel during training. Inherits from Qwen2_5_VLConfig.
    """
    model_type = "blip3o_qwen"


class blip3oQwenModel(blip3oMetaModel, Qwen2_5_VLModel):
    """
    Core model class integrating blip3o meta features (like query tokens, projectors) 
    with the Qwen2_5_VL model for multimodal learning.
    """
    config_class = blip3oQwenConfig

    def __init__(self, config: Qwen2_5_VLConfig):
        super(blip3oQwenModel, self).__init__(config)


class blip3oQwenForCausalLM(Qwen2_5_VLForConditionalGeneration, blip3oMetaForCausalLM):
    """
    Causal Language Model combined with the Blip3o multimodal architecture, 
    designed for joint training of text generation and image generation tasks.
    """
    config_class = blip3oQwenConfig

    def __init__(self, config):
        Qwen2_5_VLForConditionalGeneration.__init__(self, config)
        config.model_type = "blip3o_qwen"

        self.model = blip3oQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        """Returns the core blip3oQwenModel instance."""
        return self.model


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        ids: Optional[list] = None, # Batch IDs, likely unused here
        i_s_pos: Optional[list] = None, # Start position of image query tokens in the sequence
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        gen_image: Optional[torch.FloatTensor] = None, # Target VAE image
        gen_ref_image: Optional[torch.FloatTensor] = None, # Reference image for I2I
        ref_mask: Optional[torch.FloatTensor] = None, # Mask for reference image, likely unused in final stage
        und_image: Optional[torch.FloatTensor] = None, # Underspecified image for grounding (I2I source)
        grid_thw: Optional[torch.FloatTensor] = None,
        task_types: List[str] = None, # Task type indicators, likely unused here
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Performs a forward pass for joint text and image generation training.
        
        The process involves:
        1. Preparing multimodal inputs (embedding images/query tokens into the text sequence).
        2. Running the main Qwen model (LLM) forward.
        3. Extracting and down-projecting the generated image hidden states.
        4. Computing the Image Generation Loss (Flow Matching/DiT).
        
        Args:
            ... (see detailed arguments above) ...

        Returns:
            Union[Tuple, CausalLMOutputWithPast]: Model outputs, including the total loss 
            (image generation loss only if labels is None).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                latents, 
                ref_latents,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                gen_image,
                gen_ref_image,
                und_image,
                grid_thw,
                i_s_pos,
                image_sizes,
            )

        
        # 1. Main LLM Forward Pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        total_loss = None
        if labels is not None:
            # We only calculate the image generation loss here (text generation loss is omitted in this block)
            
            # --- 2. Prepare Image Conditioning Hidden States ---
            
            # NOTE: Text generation loss (CE) calculation logic is typically done here 
            # but seems omitted/deferred in this snippet (labels are -100 for generation tokens).
            
            img_hidden_states = []
            N_QUERY = self.get_n_query()
            
            # Extract the hidden states corresponding to the image query tokens
            for b in range(hidden_states.shape[0]):
                img_hidden_states.append(hidden_states[b,i_s_pos[b]:i_s_pos[b] + N_QUERY,:])
            img_hidden_states = torch.stack(img_hidden_states,dim=0)
            
            # Down-project the conditioning hidden states for the DiT
            img_hidden_states = self.get_model().down_projector(img_hidden_states)
            
            # --- 3. Image Generation Loss (Flow Matching/DiT) ---
            
            if latents is None:
                # Fallback: if no target latent, use MSE to stabilize projector (dummy loss)
                img_loss_funct = torch.nn.MSELoss()
                img_loss = img_loss_funct(img_hidden_states, torch.clone(img_hidden_states.detach()))
            else:
                bsz = latents.shape[0]
                dtype = latents.dtype
                
                # 3.1. Prepare Noisy Latents (Flow Matching)
                noise = torch.randn_like(latents, device=latents.device)
                u = torch.sigmoid(torch.normal(mean=0, std=1, size=(bsz,)))
                timesteps = (u * self.get_model().noise_scheduler.config.num_train_timesteps).to(device=latents.device, dtype=torch.float32)
                sigmas = self.get_sigmas(timesteps, latents.device, n_dim=latents.ndim, dtype=dtype)
                
                # Forward process: noisy_latent = (1-sigma)*latent + sigma*noise
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
                
                # 3.2. Prepare Reference Latents (Add noise and apply mask)
                ref_latents = (0.8 * ref_latents + 0.2 * torch.randn_like(ref_latents) ) * ref_mask
                
                # 3.3. Apply Joint Masking/Dropping (Classifier-Free Guidance)
                img_hidden_states_input, ref_latents = self.mask_drop_joint(
                    img_hidden_states, ref_latents, 0.05,
                )
                
                # 3.4. DiT Prediction
                img_hidden_attention_mask = torch.ones(
                    (img_hidden_states_input.shape[0], img_hidden_states_input.shape[1])).to(
                    img_hidden_states_input.device, dtype=img_hidden_states_input.dtype)
                
                model_pred = self.get_model().dit(
                    hidden_states=noisy_latents,
                    ref_hidden_states=ref_latents,
                    encoder_hidden_states=img_hidden_states_input,
                    encoder_attention_mask=img_hidden_attention_mask,
                    timestep=timesteps,
                    return_dict=False,
                )[0]
                
                # 3.5. Loss Calculation (Target: noise - latent)
                target = noise - latents
                img_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
            print(f"img loss {img_loss}")
            total_loss = img_loss # Total loss is currently only the image loss

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        

    def mask_drop_joint(self, txt_latents, img_latents, drop_prob=0.05):
        """
        Jointly applies conditional dropouts to text and image latents for 
        Classifier-Free Guidance (CFG) during training.

        Args:
            txt_latents (torch.Tensor): Text conditioning latents (img_hidden_states).
            img_latents (torch.Tensor): Image conditioning latents (ref_latents).
            drop_prob (float): Base probability for dropping one or both conditions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Masked text and image latents.
        """
        if drop_prob <= 0:
            return txt_latents, img_latents

        device = txt_latents.device
        dtype = txt_latents.dtype
        batch_size = txt_latents.shape[0]

        rand = torch.rand(batch_size, device=device)

        txt_mask = torch.ones(batch_size, device=device, dtype=dtype)
        img_mask = torch.ones(batch_size, device=device, dtype=dtype)

        # Drop text conditioning (Unconditional text)
        txt_mask[(rand >= 0.00) & (rand < drop_prob)] = 0 
        # Drop image conditioning (Unconditional image/ref)
        img_mask[(rand >= drop_prob) & (rand < 2 * drop_prob)] = 0  
        # Drop both (Fully Unconditional)
        both_mask = (rand >= 2 * drop_prob) & (rand < 3 * drop_prob)
        txt_mask[both_mask] = 0
        img_mask[both_mask] = 0

        # Reshape masks to match latent dimensions
        while len(txt_mask.shape) < len(txt_latents.shape):
            txt_mask = txt_mask.unsqueeze(-1)
        while len(img_mask.shape) < len(img_latents.shape):
            img_mask = img_mask.unsqueeze(-1)

        return txt_latents * txt_mask, img_latents * img_mask
    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        gen_images, gen_ref_images, und_images, grid_thw, i_s_pos, image_sizes=None
    ):
        """
        Processes and replaces special multimodal tokens (T2I, I2I, UND_IMAGE) with 
        their corresponding embeddings or query tokens in the text sequence.
        
        Args:
            gen_images (torch.Tensor): Target image pixels (for VAE).
            gen_ref_images (torch.Tensor): Reference image pixels (for VAE).
            und_images (torch.Tensor): Underspecified image pixels (for LLM context).
            ... (other arguments from forward) ...

        Returns:
            Tuple: Modified inputs/labels and VAE latents: (input_ids, position_ids, 
            attention_mask, past_key_values, inputs_embeds, labels, target_image_embeds, ref_image_embeds)
        """
        
        vision_tower = self.visual # Vision Encoder for contextual images (UND_IMAGE)
        gen_vision_tower = self.get_gen_vision_tower() # VAE Encoder for generation targets/refs

        # Check for non-multimodal inputs
        if (gen_images is None and und_images is None) or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None
        
        # 1. Encode Target and Reference Images (for diffusion loss calculation)
        prompt_image_embeds = gen_vision_tower(gen_images) # Target VAE latent
        ref_image_embeds = gen_vision_tower(gen_ref_images) if gen_ref_images is not None else None # Reference VAE latent
      
        target_image_embeds = torch.clone(prompt_image_embeds).detach()
        
        # 2. Get Query Tokens
        t2i_queries = self.get_model().t2i_queries.repeat(input_ids.shape[0], 1, 1)
        i2i_queris = self.get_model().i2i_queries.repeat(input_ids.shape[0], 1, 1)
        H = t2i_queries.shape[-1]
        N = self.get_n_query()
        
        t2i_queries = t2i_queries.contiguous().view(-1, H)
        i2i_queries = i2i_queris.contiguous().view(-1, H)

        # 3. Encode Underspecified Images (for text context)
        if not und_images is None:
            und_image_embeds = vision_tower(und_images, grid_thw=grid_thw)
        
        # 4. Identify Special Tokens in the text sequence
        t2i_idx = (input_ids == T2I_TOKEN_IDX)
        i2i_idx = (input_ids == I2I_TOKEN_IDX)
        und_image_idx = (input_ids == UND_IMAGE_TOKEN_IDX)
        
        # Indicators for masking labels (only tokens that are NOT LLM targets)
        output_indicator = labels != -100 # LLM prediction tokens
        input_indicator = labels == -100 # LLM context tokens (inputs_embeds)
        
        text_embeds = self.get_model().embed_tokens(input_ids)
        text_embeds = text_embeds.clone() 

        # 5. Replace Image Generation Tokens (T2I_TOKEN, I2I_TOKEN)
        # These tokens are for the LLM output sequence, replaced by the query tokens
        t2i_idx = torch.logical_and(output_indicator, t2i_idx)
        i2i_idx = torch.logical_and(output_indicator, i2i_idx)
        
        assert t2i_idx.sum() % N == 0, "T2I query count mismatch"
        assert i2i_idx.sum() % N == 0, "I2I query count mismatch"
        
        text_embeds[t2i_idx] = t2i_queries[:t2i_idx.sum()]
        text_embeds[i2i_idx] = i2i_queries[:i2i_idx.sum()]
        
        # Mask labels for the query tokens (they are inputs to the LLM now, not targets)
        labels[t2i_idx] = -100
        labels[i2i_idx] = -100

        # 6. Replace Context Image Token (UND_IMAGE_TOKEN)
        # This token is for the LLM input sequence, replaced by visual embeddings
        und_img_idx = torch.logical_and(input_indicator, und_image_idx)
        
        if not und_images is None:
            text_embeds[und_img_idx] = und_image_embeds.to(text_embeds.device)[:und_img_idx.sum(), :]

        return None, position_ids, attention_mask, past_key_values, text_embeds, labels, target_image_embeds, ref_image_embeds

# Register configurations for Hugging Face compatibility
AutoConfig.register("blip3o_qwen", blip3oQwenConfig)
AutoModelForCausalLM.register(blip3oQwenConfig, blip3oQwenForCausalLM)