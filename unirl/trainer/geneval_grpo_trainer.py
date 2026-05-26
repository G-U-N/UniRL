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

"""GenEval GRPO trainers.

Two trainers share a common base (`_GenEvalGRPOBase`):
- `GenEvalPromptGRPOTrainer`: trains only the Qwen2.5-VL prompt refiner (LLM),
  keeping the FLUX transformer frozen. Cheaper / more stable; standard
  `compute_loss` path with one LLM gradient update per step.
- `GenEvalJointGRPOTrainer`: trains the LLM AND the FLUX transformer jointly.
  Mirrors `EditJointGRPOTrainer`: SDE-sampled rollout for log-prob tracking and
  a chunked PPO-style diffusion loss.

Both trainers are backbone-agnostic over `qwenflux` (FLUX.1-dev T2I) and
`qwenklein` (FLUX.2-klein-base T2I), auto-detected from the model checkpoint
path substring. Compatible with the existing GenEval scorer HTTP service.
"""

import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers
from accelerate.utils import DistributedType
from datasets import Dataset, IterableDataset
from packaging import version
from PIL import Image
from transformers import AutoProcessor, GenerationConfig, PreTrainedModel, Trainer, TrainerCallback
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from trl.data_utils import maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig

from unimodel.qwenflux.flux_pipeline import sde_step_with_logprob as flux_sde_step
from unimodel.qwenflux.qwenflux_inference import QwenFluxForInferenceLM
from unimodel.qwenklein.flux2_klein_pipeline import sde_step_with_logprob as klein_sde_step
from unimodel.qwenklein.qwenklein_inference import QwenKleinForInferenceLM

if is_peft_available():
    from peft import PeftConfig, get_peft_model


RewardFunc = Callable[..., Union[List[float], Dict[str, Any]]]


# Each backbone exposes the same Qwen2.5-VL + diffusion-expert API; the only
# things that differ are which inference class to instantiate, which SDE step
# helper to call, and what `batched_states` look like (Klein has no
# pooled_projections, etc.).
_BACKBONE_MODELS = {
    "qwenflux": (QwenFluxForInferenceLM, flux_sde_step),
    "qwenklein": (QwenKleinForInferenceLM, klein_sde_step),
}


def _resolve_backbone(model_id: str):
    lowered = model_id.lower()
    for token, value in _BACKBONE_MODELS.items():
        if token in lowered:
            return value
    raise ValueError(
        "GenEval training expects a checkpoint path containing one of "
        f"{sorted(_BACKBONE_MODELS)} (got {model_id!r})."
    )


class _GenEvalGRPOBase(Trainer):
    """Shared scaffolding for GenEval prompt-refinement RL."""

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: List[Tuple[str, Optional[Any], RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, Dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[Any] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        processor_name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        attn_implementation: str = "flash_attention_2",
        num_sde: int = 8,
        sde_noise_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            args = GRPOConfig(f"{os.path.basename(model_name)}-geneval-grpo")

        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )

        if isinstance(model, str):
            self.model_id = model
            self._backbone_cls, self._sde_step_fn = _resolve_backbone(model)
            model = self._load_model(model, model_init_kwargs)
        else:
            self.model_id = model.config._name_or_path
            self._backbone_cls, self._sde_step_fn = _resolve_backbone(self.model_id)
            if args.model_init_kwargs is not None:
                raise ValueError("model_init_kwargs can only be used when model is a path.")

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        self._configure_trainable_parameters(model)
        self.ref_model = self._create_reference_model(model, model_init_kwargs)
        self.scheduler = model.get_model().diffusion_expert.scheduler

        if processing_class is None:
            processing_class = self._create_processor(processor_name_or_path)
        self.processing_class = processing_class

        self.reward_funcs = reward_funcs
        self.max_prompt_length = args.max_prompt_length
        self.num_generations = args.num_generations
        self.beta = args.beta
        self.num_sde = num_sde
        if sde_noise_scale is None:
            sde_noise_scale = float(os.getenv("PROMPTRL_GENEVAL_SDE_NOISE_SCALE", "0.8"))
        self.sde_noise_scale = float(sde_noise_scale)

        backbone_token = next(token for token in _BACKBONE_MODELS if token in self.model_id.lower())
        backbone_defaults = self._backbone_defaults(backbone_token)
        self._diffusion_defaults = {
            "guidance_scale": guidance_scale
            if guidance_scale is not None
            else backbone_defaults["guidance_scale"],
            "num_inference_steps": num_inference_steps
            if num_inference_steps is not None
            else backbone_defaults["num_inference_steps"],
            "height": height if height is not None else backbone_defaults["height"],
            "width": width if width is not None else backbone_defaults["width"],
        }

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length or 256,
            do_sample=True,
            temperature=1.0,
            num_return_sequences=1,
            pad_token_id=processing_class.pad_token_id,
            eos_token_id=processing_class.eos_token_id,
        )
        model.generation_config = self.generation_config
        self.ref_model.generation_config = self.generation_config
        if hasattr(model, "warnings_issued"):
            model.warnings_issued["estimate_tokens"] = True

        self._metrics = defaultdict(list)

        def data_collator(features):
            return features

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        self.model_accepts_loss_kwargs = False

        if self.is_deepspeed_enabled and is_deepspeed_zero3_enabled():
            self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
        else:
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        self.diffusion_generation_config = self._get_diffusion_config()
        self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.log_dir = os.path.join(args.output_dir, "training_samples", self.start_time)
        os.makedirs(self.log_dir, exist_ok=True)

    # --- skeleton helpers ----------------------------------------------------

    @staticmethod
    def _backbone_defaults(token: str) -> Dict[str, Any]:
        if token == "qwenflux":
            # FLUX.1-dev: 12B distilled-guidance T2I. 20 inference, 10 SDE steps
            # (matches what we run in the launch script defaults).
            return {
                "guidance_scale": 3.5,
                "num_inference_steps": 20,
                "height": 1024,
                "width": 1024,
            }
        if token == "qwenklein":
            # FLUX.2-klein-base-4B T2I. 20 inference, 8 SDE steps by default.
            return {
                "guidance_scale": 4.0,
                "num_inference_steps": 20,
                "height": 1024,
                "width": 1024,
            }
        raise ValueError(f"No defaults registered for backbone token {token!r}.")

    def _load_model(self, model_id: str, model_init_kwargs: Dict[str, Any]) -> PreTrainedModel:
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, str) and torch_dtype != "auto":
            model_init_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
        return self._backbone_cls.from_pretrained(model_id, **model_init_kwargs)

    def _create_reference_model(
        self, model: PreTrainedModel, model_init_kwargs: Dict[str, Any]
    ) -> PreTrainedModel:
        if is_deepspeed_zero3_enabled():
            ref_model = self._load_model(self.model_id, model_init_kwargs)
        else:
            ref_model = create_reference_model(model)
        for parameter in ref_model.parameters():
            parameter.requires_grad = False
        return ref_model

    def _create_processor(self, processor_name_or_path: str) -> AutoProcessor:
        processor = AutoProcessor.from_pretrained(processor_name_or_path)
        processor.pad_token_id = processor.tokenizer.pad_token_id
        processor.eos_token_id = processor.tokenizer.eos_token_id
        return processor

    def _get_diffusion_config(self) -> Dict[str, Any]:
        device_text = str(self.accelerator.device)
        device_id = int(device_text.split(":")[-1]) if ":" in device_text else 0
        return {
            "guidance_scale": float(
                os.getenv("PROMPTRL_GENEVAL_GUIDANCE_SCALE", str(self._diffusion_defaults["guidance_scale"]))
            ),
            "num_inference_steps": int(
                os.getenv(
                    "PROMPTRL_GENEVAL_NUM_INFERENCE_STEPS",
                    str(self._diffusion_defaults["num_inference_steps"]),
                )
            ),
            "num_images_per_prompt": 1,
            "generator": torch.manual_seed(42 + device_id),
            "height": int(os.getenv("PROMPTRL_GENEVAL_HEIGHT", str(self._diffusion_defaults["height"]))),
            "width": int(os.getenv("PROMPTRL_GENEVAL_WIDTH", str(self._diffusion_defaults["width"]))),
        }

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _configure_trainable_parameters(self, model: PreTrainedModel) -> None:
        """Default: train LLM only; subclasses may override for joint training."""
        try:
            model.get_model().diffusion_expert.enable_vae_slicing()
        except AttributeError:
            try:
                model.get_model().diffusion_expert.vae.enable_slicing()
            except AttributeError:
                pass

        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.get_model().parameters():
            parameter.requires_grad = True
        for parameter in model.lm_head.parameters():
            parameter.requires_grad = True

        # Freeze vision (we generate without source images, but the tower is still loaded).
        if hasattr(model, "visual"):
            for parameter in model.visual.parameters():
                parameter.requires_grad = False

        for component_name in (
            "visual",
            "vae",
            "text_encoder",
            "text_encoder_2",
            "text_encoder_3",
            "transformer",
        ):
            component = getattr(model.get_model(), component_name, None)
            if component is not None:
                for parameter in component.parameters():
                    parameter.requires_grad = False

    # --- shared inference + reward path -------------------------------------

    def _build_chat_prompt_inputs(self, inputs: List[Dict]) -> Dict[str, torch.Tensor]:
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
            for _ in range(self.num_generations)
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]
        return prompt_inputs

    def _refined_completions(
        self,
        model: nn.Module,
        prompt_inputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, int, List[str], List[str]]:
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            with torch.no_grad():
                prompt_completion_ids = unwrapped_model.generate(
                    **prompt_inputs,
                    generation_config=self.generation_config,
                )

        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        completions = self.processing_class.tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        refined_prompts = [self.model.extract_thinking_content(comp) for comp in completions]
        return prompt_completion_ids, completion_ids, prompt_length, completions, refined_prompts

    def compute_rewards(
        self,
        inputs: List[Dict],
        images: List[Image.Image],
        completions: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(images), len(self.reward_funcs), device=device)

        for index, (func_name, _, reward_func) in enumerate(self.reward_funcs):
            if func_name == "format":
                rewards_per_func[:, index] = torch.tensor(
                    reward_func(completions), device=device, dtype=torch.float32
                )
            elif func_name == "gen_eval":
                expanded_prompts = [
                    example["caption"] for example in inputs for _ in range(self.num_generations)
                ]
                expanded_meta = [
                    example.get("metadata", {}) for example in inputs for _ in range(self.num_generations)
                ]
                result = reward_func(images, expanded_prompts, expanded_meta)
                rewards_per_func[:, index] = torch.tensor(
                    result["scores"], device=device, dtype=torch.float32
                )
            else:
                raise ValueError(f"Unsupported reward function for GenEval training: {func_name}")

        return rewards_per_func.sum(dim=1), rewards_per_func

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        grouped_rewards = rewards.view(-1, self.num_generations)
        mean = grouped_rewards.mean(dim=1).repeat_interleave(self.num_generations, dim=0)
        std = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(self.num_generations, dim=0)
        return torch.clamp((rewards - mean) / (std + 1e-4), -5, 5)

    def _completion_mask(self, completion_ids: torch.Tensor) -> torch.Tensor:
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = completion_ids.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        return (sequence_indices <= eos_idx.unsqueeze(1)).int()

    def _get_per_token_logps(self, model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        logits = model(input_ids).logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        per_token_logps = []
        for logits_row, target_ids_row in zip(logits, target_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            per_token_logps.append(
                torch.gather(log_probs, dim=1, index=target_ids_row.unsqueeze(1)).squeeze(1)
            )
        return torch.stack(per_token_logps)

    def cot_loss_computation(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        prompt_length: int,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        per_token_logps = self._get_per_token_logps(model, input_ids)[:, prompt_length - 1 :]
        with torch.inference_mode():
            ref_per_token_logps = self._get_per_token_logps(self.ref_model, input_ids)[:, prompt_length - 1 :]

        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps)
            - 1
        )
        completion_mask = self._completion_mask(completion_ids)
        per_token_loss = (
            torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        )
        per_token_loss = -(per_token_loss - 0.01 * per_token_kl)
        cot_loss = (
            (per_token_loss * completion_mask).sum(dim=1)
            / completion_mask.sum(dim=1).clamp_min(1)
        ).mean()
        mean_kl = (
            (per_token_kl * completion_mask).sum(dim=1)
            / completion_mask.sum(dim=1).clamp_min(1)
        ).mean()

        self._metrics["completion_length"].append(
            self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        )
        self._metrics["cot_kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        self._metrics["cot_loss"].append(self.accelerator.gather_for_metrics(cot_loss).mean().item())
        return cot_loss

    def _log_samples(
        self,
        prompts: List[str],
        images: List[Image.Image],
        advantages: torch.Tensor,
        completions: List[str],
    ) -> None:
        global_step = self.state.global_step
        if global_step % 10 != 0 or not images:
            return

        device_id = str(self.accelerator.device).replace(":", "")
        txt_path = os.path.join(self.log_dir, f"step_{global_step}_{device_id}.txt")
        text_lines = []
        for idx in range(self.num_generations):
            text_lines.append(f"[gen {idx}] refined: {prompts[idx] if idx < len(prompts) else ''}")
            text_lines.append(f"[gen {idx}] completion: {completions[idx] if idx < len(completions) else ''}")
        if not os.path.exists(txt_path):
            with open(txt_path, "w", encoding="utf-8") as file:
                file.write("\n".join(text_lines))

        for idx, image in enumerate(images):
            image.save(
                os.path.join(
                    self.log_dir,
                    f"step_{global_step}_{device_id}_gen{idx}_{advantages[idx].item():.5f}.jpg",
                )
            )

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(value) / len(value) for key, value in self._metrics.items() if value}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()


class GenEvalPromptGRPOTrainer(_GenEvalGRPOBase):
    """Prompt-only GRPO trainer (LLM updates, DiT frozen)."""

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        opt_kwargs = {
            "lr": float(os.getenv("LLM_LEARNING_RATE", os.getenv("PROMPTRL_LLM_LR", "1e-6"))),
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "eps": self.args.adam_epsilon,
            "weight_decay": self.args.weight_decay,
        }
        llm_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not llm_params:
            raise ValueError("No trainable parameters were found for GenEval prompt GRPO training.")
        self.optimizer = torch.optim.AdamW(llm_params, **opt_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("GenEvalPromptGRPOTrainer does not support return_outputs.")

        prompt_inputs = self._build_chat_prompt_inputs(inputs)
        (
            prompt_completion_ids,
            completion_ids,
            prompt_length,
            completions,
            refined_prompts,
        ) = self._refined_completions(model, prompt_inputs)

        # Non-SDE inference is enough since DiT is frozen — log-probs aren't needed.
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            with torch.no_grad():
                images = unwrapped_model.generate_image(
                    texts=refined_prompts,
                    diffusion_kwargs=self.diffusion_generation_config,
                    sde_sampling=False,
                )

        rewards, rewards_per_func = self.compute_rewards(inputs, images, completions)
        advantages = self.compute_advantages(rewards)

        self._metrics["reward"].append(
            self.accelerator.gather_for_metrics(rewards).mean().item()
        )
        for index, (func_name, _, _) in enumerate(self.reward_funcs):
            self._metrics[f"reward/{func_name}"].append(
                self.accelerator.gather_for_metrics(rewards_per_func[:, index]).mean().item()
            )

        self._log_samples(refined_prompts, images, advantages, completions)

        cot_loss = self.cot_loss_computation(
            model, prompt_completion_ids, completion_ids, prompt_length, advantages
        )
        return cot_loss


class GenEvalJointGRPOTrainer(_GenEvalGRPOBase):
    """Joint GRPO trainer (LLM + DiT updates with SDE log-prob tracking)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # `num_sde` was already set by the base; nothing extra to do here.

    def _configure_trainable_parameters(self, model: PreTrainedModel) -> None:
        try:
            model.get_model().diffusion_expert.enable_vae_slicing()
        except AttributeError:
            try:
                model.get_model().diffusion_expert.vae.enable_slicing()
            except AttributeError:
                pass

        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.get_model().parameters():
            parameter.requires_grad = True
        for parameter in model.lm_head.parameters():
            parameter.requires_grad = True

        if hasattr(model, "visual"):
            for parameter in model.visual.parameters():
                parameter.requires_grad = False

        for component_name in ("visual", "vae", "text_encoder", "text_encoder_2", "text_encoder_3"):
            component = getattr(model.get_model(), component_name, None)
            if component is not None:
                for parameter in component.parameters():
                    parameter.requires_grad = False

        transformer = getattr(model.get_model(), "transformer", None)
        if transformer is None:
            raise ValueError("GenEval joint training requires a FLUX transformer.")
        for parameter in transformer.parameters():
            parameter.requires_grad = True

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        optimizer_kwargs = {
            "eps": self.args.adam_epsilon,
            "weight_decay": self.args.weight_decay,
        }
        dit_lr = float(os.getenv("DIT_LEARNING_RATE", os.getenv("PROMPTRL_DIT_LR", "2e-7")))
        llm_lr = float(os.getenv("LLM_LEARNING_RATE", os.getenv("PROMPTRL_LLM_LR", "3e-7")))
        dit_beta1 = float(os.getenv("DIT_BETA1", str(self.args.adam_beta1)))
        dit_beta2 = float(os.getenv("DIT_BETA2", str(self.args.adam_beta2)))
        llm_beta1 = float(os.getenv("LLM_BETA1", str(self.args.adam_beta1)))
        llm_beta2 = float(os.getenv("LLM_BETA2", str(self.args.adam_beta2)))

        dit_params = [
            p for p in self.model.get_model().transformer.parameters() if p.requires_grad
        ]
        dit_param_ids = {id(p) for p in dit_params}
        llm_params = [
            p for p in self.model.parameters() if p.requires_grad and id(p) not in dit_param_ids
        ]
        param_groups = []
        if dit_params:
            param_groups.append({"params": dit_params, "lr": dit_lr, "betas": (dit_beta1, dit_beta2)})
        if llm_params:
            param_groups.append({"params": llm_params, "lr": llm_lr, "betas": (llm_beta1, llm_beta2)})
        if not param_groups:
            raise ValueError("No trainable parameters were found for GenEval joint GRPO training.")

        self.optimizer = torch.optim.AdamW(param_groups, **optimizer_kwargs)
        return self.optimizer

    def _diffusion_kwargs_with_sde(self) -> Dict[str, Any]:
        kwargs = dict(self.diffusion_generation_config)
        kwargs["num_sde"] = self.num_sde
        kwargs["noise_scale"] = self.sde_noise_scale
        return kwargs

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ):
        model.eval()
        self.ref_model.eval()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        def loss_update(loss: torch.Tensor, scale_factor: float = 1.0) -> None:
            if self.args.n_gpu > 1:
                loss = loss.mean()
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                loss = loss / scale_factor
                model.backward(loss)
            else:
                self.accelerator.backward(loss / scale_factor)

        with self.compute_loss_context_manager():
            generations = self.generate_samples(model, inputs)
            torch.cuda.empty_cache()

            cot_loss = self.cot_loss_computation(
                model,
                generations["prompt_completion_ids"],
                generations["completion_ids"],
                generations["prompt_length"],
                generations["advantages"],
            )
            loss_update(cot_loss, 1.0)

            diff_advantages = generations["advantages"].repeat_interleave(self.num_sde, dim=0)
            total_len = diff_advantages.shape[0]
            diff_loss_values = []
            diff_kl_values = []
            diffusion_batch_size = int(os.getenv("PROMPTRL_DIFFUSION_LOSS_BATCH_SIZE", "4"))

            for idx in range(0, total_len, diffusion_batch_size):
                batched_states_slice = {}
                for key, value in generations["batched_states"].items():
                    if key in {"img_ids", "txt_ids"}:
                        batched_states_slice[key] = value
                    elif value is None:
                        batched_states_slice[key] = None
                    else:
                        batched_states_slice[key] = value[idx : idx + diffusion_batch_size]

                diff_loss, diff_kl = self.diffusion_loss_computation(
                    generations["prev_latents"][idx : idx + diffusion_batch_size],
                    generations["diff_sampling_log_probs"][idx : idx + diffusion_batch_size],
                    generations["pred_latents"][idx : idx + diffusion_batch_size],
                    generations["ts"][idx : idx + diffusion_batch_size],
                    batched_states_slice,
                    diff_advantages[idx : idx + diffusion_batch_size],
                )
                loss_update(diff_loss, max(1.0, float(total_len / diffusion_batch_size)))
                diff_loss_values.append(diff_loss.detach())
                diff_kl_values.append(diff_kl.detach())

        diff_loss = torch.stack(diff_loss_values).mean()
        diff_kl = torch.stack(diff_kl_values).mean()
        loss = diff_loss + cot_loss.detach()

        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            torch.cuda.empty_cache()

        if hasattr(model, "step") and callable(model.step):
            model.step()

        self._metrics["diff_kl"].append(self.accelerator.gather_for_metrics(diff_kl).mean().item())
        self._metrics["diff_loss"].append(self.accelerator.gather_for_metrics(diff_loss).mean().item())
        torch.cuda.empty_cache()
        return loss.detach()

    def generate_samples(self, model: nn.Module, inputs: List[Dict]) -> Dict[str, Any]:
        prompt_inputs = self._build_chat_prompt_inputs(inputs)
        (
            prompt_completion_ids,
            completion_ids,
            prompt_length,
            completions,
            refined_prompts,
        ) = self._refined_completions(model, prompt_inputs)

        sde_kwargs = self._diffusion_kwargs_with_sde()
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            with torch.no_grad():
                (
                    images,
                    prev_latents,
                    diff_sampling_log_probs,
                    pred_latents,
                    timesteps,
                    batched_states,
                ) = unwrapped_model.generate_image(
                    texts=refined_prompts,
                    diffusion_kwargs=sde_kwargs,
                    sde_sampling=True,
                )

        rewards, rewards_per_func = self.compute_rewards(inputs, images, completions)
        advantages = self.compute_advantages(rewards)

        self._metrics["reward"].append(
            self.accelerator.gather_for_metrics(rewards).mean().item()
        )
        for index, (func_name, _, _) in enumerate(self.reward_funcs):
            self._metrics[f"reward/{func_name}"].append(
                self.accelerator.gather_for_metrics(rewards_per_func[:, index]).mean().item()
            )

        self._log_samples(refined_prompts, images, advantages, completions)

        return {
            "images": images,
            "prev_latents": prev_latents,
            "diff_sampling_log_probs": diff_sampling_log_probs,
            "pred_latents": pred_latents,
            "batched_states": batched_states,
            "prompt_length": prompt_length,
            "completion_ids": completion_ids,
            "prompt_completion_ids": prompt_completion_ids,
            "advantages": advantages,
            "ts": timesteps,
        }

    def diffusion_loss_computation(
        self,
        prev_latents: torch.Tensor,
        diff_sampling_log_probs: torch.Tensor,
        pred_latents: torch.Tensor,
        timesteps: torch.Tensor,
        batched_states: Dict[str, torch.Tensor],
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model_pred = self.model.get_model().transformer(
            hidden_states=prev_latents.to(self.model.device),
            **batched_states,
            joint_attention_kwargs={},
            return_dict=False,
        )[0][:, : pred_latents.size(1)]

        with torch.no_grad():
            ref_model_pred = self.ref_model.get_model().transformer(
                hidden_states=prev_latents.to(self.model.device),
                **batched_states,
                joint_attention_kwargs={},
                return_dict=False,
            )[0][:, : pred_latents.size(1)]

        _, log_prob, prev_sample_mean, std_dev_t = self._sde_step_fn(
            self.scheduler,
            model_pred.float(),
            timesteps,
            prev_latents[:, : pred_latents.size(1)].float(),
            pred_latents.float(),
            noise_scale=self.sde_noise_scale,
        )
        _, _, ref_prev_sample_mean, ref_std_dev_t = self._sde_step_fn(
            self.scheduler,
            ref_model_pred.float(),
            timesteps,
            prev_latents[:, : pred_latents.size(1)].float(),
            pred_latents.float(),
            noise_scale=self.sde_noise_scale,
        )
        if not torch.equal(std_dev_t, ref_std_dev_t):
            raise RuntimeError("Current and reference SDE std-dev tensors diverged.")

        kl = ((prev_sample_mean - ref_prev_sample_mean) ** 2 / (2 * std_dev_t**2)).mean(
            dim=tuple(range(1, prev_sample_mean.ndim))
        )
        ratio = torch.exp(log_prob - diff_sampling_log_probs)
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * torch.clamp(ratio, 1.0 - 1e-4, 1.0 + 1e-4)
        diff_loss = torch.maximum(unclipped_loss, clipped_loss).mean() + self.beta * kl.mean()
        return diff_loss, kl
