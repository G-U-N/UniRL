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

"""FLUX.2-klein pipeline subclass with SDE log-probability tracking.

`Flux2KleinPipelineWithSDE.sde_sampling` mirrors the contract of the existing
`FluxKontextPipeline.sde_sampling` in `unimodel.qwenkontext.fluxkontext_pipeline`:
it returns ``(images, prev_latents, log_probs, pred_latents, timesteps, batched_states)``
so the joint GRPO trainer can reuse the same diffusion-loss path across backbones.

Key differences from the Kontext pipeline:
- Klein uses a single Qwen3 text encoder (no CLIP / no T5), so there is no
  ``pooled_projections`` argument to the transformer.
- Klein's transformer is called with ``guidance=None`` (the Klein-base variant
  follows rectified flow without an embedded guidance condition).
- Image conditioning is still concatenation-based (image latents are appended to
  the noise latents along the token axis), matching Kontext.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch

from diffusers import Flux2KleinPipeline
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor


def sde_step_with_logprob(
    scheduler: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    noise_scale: float = 0.8,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """SDE step with log-prob tracking for rectified-flow schedulers.

    Numerically identical to the Kontext counterpart so the trainer can swap
    backbones without changing the GRPO objective. Kept local to avoid an
    inter-package import between the two qwen<X> modules.
    """
    step_index = [scheduler.index_for_timestep(t) for t in timestep]
    prev_step_index = [step + 1 for step in step_index]
    sigma = scheduler.sigmas[step_index].view(-1, 1, 1).to(model_output.device)
    sigma_prev = scheduler.sigmas[prev_step_index].view(-1, 1, 1).to(model_output.device)
    sigma_max = scheduler.sigmas[1].item()
    dt = sigma_prev - sigma

    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_scale

    prev_sample_mean = (
        sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
        + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
    )

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

    variance = (std_dev_t * torch.sqrt(-1 * dt)) ** 2
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * variance)
        - torch.log(torch.sqrt(variance))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )

    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample, log_prob, prev_sample_mean, std_dev_t * torch.sqrt(-1 * dt)


class Flux2KleinPipelineWithSDE(Flux2KleinPipeline):
    """Flux2KleinPipeline with an added `sde_sampling` method for diffusion GRPO.

    Inherits all standard functionality (text/image encoding, latent prep, decoding)
    and adds a denoising loop that records intermediate states at the first
    ``num_sde`` steps so the trainer can recompute log-probs and gradients.
    """

    @torch.no_grad()
    def sde_sampling(
        self,
        image: Optional[Union[List[PIL.Image.Image], PIL.Image.Image]] = None,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ("latents",),
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int, ...] = (9, 18, 27),
        num_sde: Optional[int] = None,
        noise_scale: float = 0.8,
        # The argument is accepted for symmetry with the Kontext pipeline (which uses
        # `max_area` instead of `height`/`width`). For Klein we honor `height`/`width`
        # when provided and otherwise fall back to the largest square that fits.
        max_area: Optional[int] = None,
    ):
        """SDE sampling with per-step log-prob tracking.

        Returns:
            Tuple of ``(images, prev_latents, log_probs, pred_latents, timesteps, batched_states)``
            with the same shapes as ``FluxKontextPipeline.sde_sampling``.
        """
        from diffusers.pipelines.flux2.pipeline_flux2_klein import (
            compute_empirical_mu,
            retrieve_timesteps,
        )

        # If max_area is provided and no explicit height/width, derive both.
        if max_area is not None and height is None and width is None:
            side = int(math.sqrt(max_area))
            multiple_of = self.vae_scale_factor * 2
            side = (side // multiple_of) * multiple_of
            height = width = side

        # 1. Input checks (reuse parent).
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=list(callback_on_step_end_tensor_inputs),
            guidance_scale=guidance_scale,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Batch size.
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Text embeddings.
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        do_cfg = self.do_classifier_free_guidance
        if do_cfg:
            negative_prompt = ""
            if prompt is not None and isinstance(prompt, list):
                negative_prompt = [negative_prompt] * len(prompt)
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )

        # 4. Condition image(s).
        if image is not None and not isinstance(image, list):
            image = [image]

        condition_images = None
        if image is not None:
            for img in image:
                self.image_processor.check_image_input(img)

            condition_images = []
            for img in image:
                image_width, image_height = img.size
                # If caller provided an explicit (height, width) that differs from the
                # source size, force-resize source to match. Klein conditions via
                # token-axis concat with 3D positional ids: source ids share (H, W)
                # axes with the noise grid (only T differs), so a smaller source
                # would land its content in the upper-left sub-grid of the noise
                # output. This produces a visible "quad-grid" artifact where the
                # source bleeds into one quadrant of the generated image.
                if height is not None and width is not None and (
                    image_width != int(width) or image_height != int(height)
                ):
                    img = img.resize((int(width), int(height)))
                    image_width, image_height = img.size
                elif image_width * image_height > 1024 * 1024:
                    img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                    image_width, image_height = img.size

                multiple_of = self.vae_scale_factor * 2
                image_width = (image_width // multiple_of) * multiple_of
                image_height = (image_height // multiple_of) * multiple_of
                img = self.image_processor.preprocess(
                    img, height=image_height, width=image_width, resize_mode="crop"
                )
                condition_images.append(img)
                height = height or image_height
                width = width or image_width

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 5. Latents.
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        image_latents = None
        image_latent_ids = None
        if condition_images is not None:
            image_latents, image_latent_ids = self.prepare_image_latents(
                images=condition_images,
                batch_size=batch_size * num_images_per_prompt,
                generator=generator,
                device=device,
                dtype=self.vae.dtype,
            )

        # 6. Timesteps.
        if sigmas is None:
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        self.scheduler.set_begin_index(0)

        if num_sde is None:
            num_sde = num_inference_steps

        # 7. SDE-tracked denoising loop.
        prev_latents: List[torch.Tensor] = []
        pred_latents: List[torch.Tensor] = []
        log_probs: List[torch.Tensor] = []
        ts_list: List[torch.Tensor] = []
        states = {
            "timestep": [],
            # Klein's transformer takes guidance=None unconditionally; we keep the
            # key so `**batched_states` in the trainer matches the transformer's
            # signature (it accepts None) and the existing trainer slicing code
            # that special-cases None still works.
            "guidance": [],
            "encoder_hidden_states": [],
            "txt_ids": text_ids,
            "img_ids": None,  # filled in below; depends on whether image is conditioned
        }

        latent_image_ids_full = latent_ids
        if image_latents is not None:
            latent_image_ids_full = torch.cat([latent_ids, image_latent_ids], dim=1)
        states["img_ids"] = latent_image_ids_full

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                latent_model_input = latents.to(self.transformer.dtype)
                if image_latents is not None:
                    latent_model_input = torch.cat(
                        [latents, image_latents], dim=1
                    ).to(self.transformer.dtype)

                if i < num_sde:
                    states["timestep"].append((timestep / 1000.0).unsqueeze(1))
                    states["guidance"].append(None)
                    states["encoder_hidden_states"].append(prompt_embeds.unsqueeze(1))
                    ts_list.append(t.expand(latents.shape[0]).unsqueeze(1))
                    prev_latents.append(latent_model_input.detach().clone().unsqueeze(1))

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000.0,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids_full,
                    joint_attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, : latents.size(1)]

                if do_cfg:
                    neg_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000.0,
                        guidance=None,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_image_ids_full,
                        joint_attention_kwargs=self._attention_kwargs,
                        return_dict=False,
                    )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                    noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

                if i < num_sde:
                    latents_dtype = latents.dtype
                    latents, log_prob, _prev_mean, _std = sde_step_with_logprob(
                        self.scheduler,
                        noise_pred.float(),
                        t.expand(latents.shape[0]),
                        latents.float(),
                        noise_scale=noise_scale,
                    )
                    log_probs.append(log_prob.detach().clone().unsqueeze(1))
                    pred_latents.append(latents.detach().clone().unsqueeze(1))
                    # Maintain scheduler._step_index so the subsequent deterministic
                    # scheduler.step() calls read the correct sigmas. Without this,
                    # scheduler.step()._init_step_index falls back to _begin_index=0
                    # set above, re-running the first num_sde sigmas — fine for the
                    # distilled Kontext model but catastrophic for undistilled Klein.
                    if self.scheduler.step_index is None:
                        self.scheduler._init_step_index(t)
                    self.scheduler._step_index += 1
                else:
                    latents_dtype = latents.dtype
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        self._current_timestep = None

        # 8. Decode.
        latent_height = 2 * (int(height) // (self.vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (self.vae_scale_factor * 2))
        decoded_latents = self._unpack_latents_with_ids(
            latents, latent_ids, latent_height // 2, latent_width // 2
        )
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(
            decoded_latents.device, decoded_latents.dtype
        )
        latents_bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(decoded_latents.device, decoded_latents.dtype)
        decoded_latents = decoded_latents * latents_bn_std + latents_bn_mean
        decoded_latents = self._unpatchify_latents(decoded_latents)
        if output_type == "latent":
            images = decoded_latents
        else:
            images = self.vae.decode(decoded_latents, return_dict=False)[0]
            images = self.image_processor.postprocess(images, output_type=output_type)

        # 9. Pack tracking outputs (same layout as Kontext's pipeline).
        batched_states: Dict[str, Optional[torch.Tensor]] = {}
        for key, value_list in states.items():
            if not isinstance(value_list, list):
                batched_states[key] = value_list  # txt_ids / img_ids stay shared
                continue
            if len(value_list) == 0 or value_list[0] is None:
                batched_states[key] = None
                continue
            concatenated = torch.cat(value_list, dim=1)
            if concatenated.ndim <= 2:
                batched_states[key] = concatenated.view(-1)
            else:
                batched_states[key] = concatenated.view(-1, *concatenated.shape[2:])

        prev_latents_t = torch.cat(prev_latents, dim=1)
        log_probs_t = torch.cat(log_probs, dim=1)
        pred_latents_t = torch.cat(pred_latents, dim=1)
        ts_t = torch.cat(ts_list, dim=1)

        prev_latents_t = prev_latents_t.view(
            prev_latents_t.shape[0] * prev_latents_t.shape[1], *prev_latents_t.shape[2:]
        )
        log_probs_t = log_probs_t.view(
            log_probs_t.shape[0] * log_probs_t.shape[1], *log_probs_t.shape[2:]
        )
        pred_latents_t = pred_latents_t.view(
            pred_latents_t.shape[0] * pred_latents_t.shape[1], *pred_latents_t.shape[2:]
        )
        ts_t = ts_t.view(-1)

        self.maybe_free_model_hooks()

        return (images, prev_latents_t, log_probs_t, pred_latents_t, ts_t, batched_states)
