# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
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

"""FLUX.1-dev pipeline subclass with SDE log-probability tracking.

`FluxPipelineWithSDE.sde_sampling` returns the same 6-tuple contract as the
Kontext / Klein pipelines (``(images, prev_latents, log_probs, pred_latents,
timesteps, batched_states)``) so the GenEval trainer can reuse a single
diffusion-loss path across all FLUX backbones.

Key differences from the Kontext pipeline:
- FLUX.1-dev is pure text-to-image: no source-image conditioning, no image
  latent concatenation along the token axis.
- ``guidance_embeds`` is True for the dev variant (12B): ``guidance`` is a
  per-batch float tensor rather than None.
- The dual CLIP + T5 text encoder stack contributes a ``pooled_projections``
  entry to ``batched_states`` (the CLIP pooled embedding).
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers import FluxPipeline
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
    """SDE step with log-prob tracking for the rectified-flow scheduler.

    Numerically identical to the Kontext and Klein implementations.
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


class FluxPipelineWithSDE(FluxPipeline):
    """FluxPipeline (FLUX.1-dev) with an added `sde_sampling` method for diffusion GRPO."""

    @torch.no_grad()
    def sde_sampling(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ("latents",),
        max_sequence_length: int = 512,
        num_sde: Optional[int] = None,
        noise_scale: float = 0.8,
    ):
        """SDE sampling with per-step log-prob tracking for FLUX.1-dev.

        Returns:
            Tuple of ``(images, prev_latents, log_probs, pred_latents, timesteps, batched_states)``.
        """
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Input checks.
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=list(callback_on_step_end_tensor_inputs),
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
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
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None)
            if self.joint_attention_kwargs is not None
            else None
        )
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        # 3. Text embeddings (CLIP pooled + T5 sequence).
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # 4. Latents (T2I only — no image conditioning).
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Timesteps.
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        self.scheduler.set_begin_index(0)

        # 6. Guidance vector (embedded guidance for FLUX.1-dev).
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        if num_sde is None:
            num_sde = num_inference_steps

        # 7. SDE-tracked denoising loop.
        prev_latents: List[torch.Tensor] = []
        pred_latents: List[torch.Tensor] = []
        log_probs: List[torch.Tensor] = []
        ts_list: List[torch.Tensor] = []
        states: Dict[str, Any] = {
            "timestep": [],
            "guidance": [],
            "pooled_projections": [],
            "encoder_hidden_states": [],
            "txt_ids": text_ids if text_ids is not None else None,
            "img_ids": latent_image_ids if latent_image_ids is not None else None,
        }

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                timestep = (t.expand(latents.shape[0]) / 1000.0).to(latents.dtype)

                if i < num_sde:
                    states["timestep"].append(timestep.unsqueeze(1))
                    states["guidance"].append(
                        guidance.unsqueeze(1) if torch.is_tensor(guidance) else None
                    )
                    states["pooled_projections"].append(
                        pooled_prompt_embeds.unsqueeze(1)
                        if pooled_prompt_embeds is not None
                        else None
                    )
                    states["encoder_hidden_states"].append(
                        prompt_embeds.unsqueeze(1) if prompt_embeds is not None else None
                    )
                    ts_list.append(t.expand(latents.shape[0]).unsqueeze(1))
                    prev_latents.append(latents.detach().clone().unsqueeze(1))

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                if do_true_cfg:
                    neg_noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

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
                    # Maintain scheduler._step_index so the deterministic
                    # scheduler.step() calls after the SDE prefix read sigmas
                    # at the right index. set_begin_index(0) above otherwise
                    # makes the first deterministic step re-read sigmas[0,1].
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
        if output_type == "latent":
            image = latents
        else:
            latents_unpacked = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents_unpacked = (latents_unpacked / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents_unpacked, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # 9. Pack tracking outputs.
        batched_states: Dict[str, Optional[torch.Tensor]] = {}
        for key, value_list in states.items():
            if not isinstance(value_list, list):
                batched_states[key] = value_list
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

        return (image, prev_latents_t, log_probs_t, pred_latents_t, ts_t, batched_states)
