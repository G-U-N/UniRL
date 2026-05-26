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

from unimodel.qwenkontext.fluxkontext_pipeline import sde_step_with_logprob
from unimodel.qwenkontext.qwenkontext_inference import QwenKontextForInferenceLM
from unimodel.qwenklein.qwenklein_inference import QwenKleinForInferenceLM

if is_peft_available():
    from peft import PeftConfig, get_peft_model


RewardFunc = Callable[..., Union[List[float], Dict[str, Any]]]


# Maps a substring in the checkpoint path to the inference-model class. Both
# backbones produce a HF Qwen2.5-VL conditional-generation model with an embedded
# diffusion expert exposing the same `generate_image(sde_sampling=True)` 6-tuple
# contract, so the rest of the trainer is genuinely backbone-agnostic.
_BACKBONE_MODELS = {
    "qwenkontext": QwenKontextForInferenceLM,
    "qwenklein": QwenKleinForInferenceLM,
}


def _resolve_backbone_class(model_id: str):
    lowered = model_id.lower()
    for token, cls in _BACKBONE_MODELS.items():
        if token in lowered:
            return cls
    raise ValueError(
        "Edit joint training expects a checkpoint path containing one of "
        f"{sorted(_BACKBONE_MODELS)} (got {model_id!r})."
    )


def compute_log_prob(
    model_pred: torch.Tensor,
    scheduler,
    prev_latents: torch.Tensor,
    pred_latents: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scale: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return sde_step_with_logprob(
        scheduler,
        model_pred.float(),
        timesteps,
        prev_latents.float(),
        pred_latents.float(),
        noise_scale=noise_scale,
    )


class EditJointGRPOTrainer(Trainer):
    """Joint GRPO trainer for Qwen prompt refinement and FLUX edit generation.

    Backbone-agnostic over the FLUX variant: works with both QwenKontext
    (FLUX.1-Kontext-dev) and QwenKlein (FLUX.2-klein-base) embedded diffusion
    experts. The backbone is auto-detected from the model checkpoint path.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: List[Tuple[str, Optional[Any], RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, Dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Any] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: int = 200704,
        min_pixels: int = 200704,
        processor_name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        attn_implementation: str = "flash_attention_2",
        num_skip_refinement: int = 2,
        num_sde: int = 4,
        sde_noise_scale: Optional[float] = None,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            args = GRPOConfig(f"{os.path.basename(model_name)}-edit-joint-grpo")

        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        model_init_kwargs["use_cache"] = False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")

        if isinstance(model, str):
            self.model_id = model
            model = self._load_model(model, model_init_kwargs)
        else:
            self.model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError("model_init_kwargs can only be used when model is a path.")

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        self._configure_trainable_parameters(model)
        # Snapshot VAE BatchNorm buffers from the on-disk safetensors so we
        # can restore them on every training_step. DeepSpeed ZeRO-3's
        # `zero.Init()` context skips loading non-parameter buffers, leaving
        # BN running_mean/running_var at nn.BatchNorm defaults (zeros/ones).
        # Klein's pipeline relies on those buffers to reverse latent
        # whitening at decode time; without the real values the decode
        # is washed-out / desaturated.
        self._vae_bn_buffers_snapshot = self._snapshot_vae_bn_buffers(model)
        if self._vae_bn_buffers_snapshot:
            sample = next(iter(self._vae_bn_buffers_snapshot.values()))
            rm = sample["running_mean"][:3].tolist()
            rv = sample["running_var"][:3].tolist()
            print(
                f"[VAE-bn-snapshot] captured {len(self._vae_bn_buffers_snapshot)} BN module(s): "
                f"running_mean[:3]={rm} running_var[:3]={rv}",
                flush=True,
            )
        self.ref_model = self._create_reference_model(model, model_init_kwargs)
        self.scheduler = model.get_model().diffusion_expert.scheduler

        if processing_class is None:
            processing_class = self._create_processor(processor_name_or_path, max_pixels, min_pixels)
        self.processing_class = processing_class
        self.reward_funcs = reward_funcs
        self.max_prompt_length = args.max_prompt_length
        self.num_generations = args.num_generations
        self.beta = args.beta
        self.num_sde = num_sde

        if not 0 <= num_skip_refinement < self.num_generations:
            raise ValueError(
                f"num_skip_refinement must be in [0, num_generations), got {num_skip_refinement} "
                f"for num_generations={self.num_generations}."
            )
        self.num_skip_refinement = num_skip_refinement
        self.num_refined = self.num_generations - num_skip_refinement
        if sde_noise_scale is None:
            sde_noise_scale = float(os.getenv("PROMPTRL_EDIT_SDE_NOISE_SCALE", "0.8"))
        self.sde_noise_scale = float(sde_noise_scale)

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

    def _load_model(self, model_id: str, model_init_kwargs: Dict[str, Any]) -> PreTrainedModel:
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, str) and torch_dtype != "auto":
            model_init_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
        backbone_cls = _resolve_backbone_class(model_id)
        return backbone_cls.from_pretrained(model_id, **model_init_kwargs)

    def _create_reference_model(self, model: PreTrainedModel, model_init_kwargs: Dict[str, Any]) -> PreTrainedModel:
        if is_deepspeed_zero3_enabled():
            ref_model = self._load_model(self.model_id, model_init_kwargs)
        else:
            ref_model = create_reference_model(model)
        for parameter in ref_model.parameters():
            parameter.requires_grad = False
        return ref_model

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
                # Freezing params does NOT freeze BatchNorm / Dropout — those
                # still update running stats / sample noise in `train()` mode.
                # Klein's VAE has a BatchNorm-based latent normalization
                # (see flux2_klein_pipeline.py:392-398): if its running_mean /
                # running_var drift during training, decoded color statistics
                # shift and outputs look washed out / desaturated compared to
                # the source. Force component to eval mode here AND we re-pin
                # it in training_step (model.train() at the start of each step
                # would otherwise flip it back to train mode).
                component.eval()

        transformer = getattr(model.get_model(), "transformer", None)
        if transformer is None:
            raise ValueError("Joint-edit model does not expose a FLUX transformer.")
        for parameter in transformer.parameters():
            parameter.requires_grad = True

    @staticmethod
    def _vae_bn_modules(model: nn.Module):
        """Yield every BatchNorm-like module under model.get_model().vae.

        Klein's VAE uses a single top-level `vae.bn` for latent whitening, but
        we walk all sub-modules with `running_mean` so the snapshot/restore
        survives future VAE variants that nest the BN deeper.
        """
        inner = model.module if hasattr(model, "module") else model
        inner = inner.get_model() if hasattr(inner, "get_model") else inner
        vae = getattr(inner, "vae", None)
        if vae is None:
            return
        for name, m in vae.named_modules():
            if hasattr(m, "running_mean") and hasattr(m, "running_var"):
                yield name, m

    def _snapshot_vae_bn_buffers(self, model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
        """Snapshot VAE BN running stats, preferring the on-disk checkpoint.

        Under DeepSpeed ZeRO-3 `zero.Init()` the model is constructed before
        from_pretrained loads buffers, and non-parameter buffers (BN running
        stats here) are silently left at their nn.BatchNorm defaults. Reading
        from the live model in that state would snapshot zeros / ones. We try
        the safetensors files first; fall back to in-memory if that fails.
        """
        snapshot: Dict[str, Dict[str, torch.Tensor]] = {}
        names = [name for name, _ in self._vae_bn_modules(model)]
        if not names:
            return snapshot

        disk = self._load_vae_bn_buffers_from_disk(names)
        for name, bn in self._vae_bn_modules(model):
            disk_entry = disk.get(name)
            if disk_entry is not None:
                snapshot[name] = disk_entry
            else:
                snapshot[name] = {
                    "running_mean": bn.running_mean.detach().clone().cpu(),
                    "running_var": bn.running_var.detach().clone().cpu(),
                    "num_batches_tracked": bn.num_batches_tracked.detach().clone().cpu()
                    if hasattr(bn, "num_batches_tracked") and bn.num_batches_tracked is not None
                    else None,
                }
        return snapshot

    def _load_vae_bn_buffers_from_disk(self, names: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Pull `vae.<name>.running_{mean,var,num_batches_tracked}` straight
        from the safetensors checkpoint at `self.model_id`. Returns {} on any
        failure so the caller can fall back to the in-memory snapshot.
        """
        out: Dict[str, Dict[str, torch.Tensor]] = {}
        ckpt_root = getattr(self, "model_id", None)
        if not ckpt_root or not os.path.isdir(ckpt_root):
            return out
        index_path = os.path.join(ckpt_root, "model.safetensors.index.json")
        if not os.path.isfile(index_path):
            return out

        try:
            import json
            from safetensors import safe_open
            with open(index_path) as fp:
                weight_map = json.load(fp).get("weight_map", {})
        except Exception:
            return out

        for name in names:
            # HF saves the VAE under `model.vae.<sub>`. Module names from
            # `vae.named_modules()` already include the leading sub-path
            # (empty string for the top-level vae.bn).
            prefix = f"model.vae.{name}." if name else "model.vae."
            keys = {
                "running_mean": f"{prefix}running_mean",
                "running_var": f"{prefix}running_var",
                "num_batches_tracked": f"{prefix}num_batches_tracked",
            }
            tensors: Dict[str, Any] = {}
            try:
                for label, full_key in keys.items():
                    shard = weight_map.get(full_key)
                    if shard is None:
                        if label == "num_batches_tracked":
                            tensors[label] = None
                            continue
                        raise KeyError(full_key)
                    shard_path = os.path.join(ckpt_root, shard)
                    with safe_open(shard_path, framework="pt") as f:
                        tensors[label] = f.get_tensor(full_key).detach().clone()
            except Exception:
                continue
            out[name] = tensors
        return out

    def _restore_vae_bn_buffers(self, model: nn.Module) -> None:
        if not getattr(self, "_vae_bn_buffers_snapshot", None):
            return
        for name, bn in self._vae_bn_modules(model):
            saved = self._vae_bn_buffers_snapshot.get(name)
            if saved is None:
                continue
            bn.running_mean.data.copy_(
                saved["running_mean"].to(bn.running_mean.device, dtype=bn.running_mean.dtype)
            )
            bn.running_var.data.copy_(
                saved["running_var"].to(bn.running_var.device, dtype=bn.running_var.dtype)
            )
            if saved["num_batches_tracked"] is not None and bn.num_batches_tracked is not None:
                bn.num_batches_tracked.data.copy_(
                    saved["num_batches_tracked"].to(bn.num_batches_tracked.device)
                )

    def _create_processor(self, processor_name_or_path: str, max_pixels: int, min_pixels: int) -> AutoProcessor:
        processor = AutoProcessor.from_pretrained(processor_name_or_path)
        processor.pad_token_id = processor.tokenizer.pad_token_id
        processor.eos_token_id = processor.tokenizer.eos_token_id
        processor.image_processor.max_pixels = max_pixels
        processor.image_processor.min_pixels = min_pixels
        return processor

    def _get_diffusion_config(self) -> Dict[str, Any]:
        device_text = str(self.accelerator.device)
        device_id = int(device_text.split(":")[-1]) if ":" in device_text else 0
        return {
            "guidance_scale": float(os.getenv("PROMPTRL_EDIT_GUIDANCE_SCALE", "2.5")),
            "num_inference_steps": int(os.getenv("PROMPTRL_EDIT_NUM_INFERENCE_STEPS", "8")),
            "num_images_per_prompt": 1,
            "generator": torch.manual_seed(42 + device_id),
            "height": int(os.getenv("PROMPTRL_EDIT_HEIGHT", "1024")),
            "width": int(os.getenv("PROMPTRL_EDIT_WIDTH", "1024")),
            "num_sde": self.num_sde,
            "noise_scale": self.sde_noise_scale,
        }

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        optimizer_kwargs = {
            "eps": self.args.adam_epsilon,
            "weight_decay": self.args.weight_decay,
        }
        dit_lr = float(os.getenv("DIT_LEARNING_RATE", os.getenv("PROMPTRL_DIT_LR", "2e-7")))
        llm_lr = float(os.getenv("LLM_LEARNING_RATE", os.getenv("PROMPTRL_LLM_LR", "3e-7")))
        # Per-group betas: in joint RL the DiT sees a non-stationary prompt distribution
        # (refined by the LLM), so a lower beta1 there often helps. The LLM side is the
        # policy itself; HF defaults stay sensible. Fall back to args.adam_beta* when unset.
        dit_beta1 = float(os.getenv("DIT_BETA1", str(self.args.adam_beta1)))
        dit_beta2 = float(os.getenv("DIT_BETA2", str(self.args.adam_beta2)))
        llm_beta1 = float(os.getenv("LLM_BETA1", str(self.args.adam_beta1)))
        llm_beta2 = float(os.getenv("LLM_BETA2", str(self.args.adam_beta2)))

        dit_params = [
            parameter for parameter in self.model.get_model().transformer.parameters() if parameter.requires_grad
        ]
        dit_param_ids = {id(parameter) for parameter in dit_params}
        llm_params = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in dit_param_ids
        ]

        param_groups = []
        if dit_params:
            param_groups.append({"params": dit_params, "lr": dit_lr, "betas": (dit_beta1, dit_beta2)})
        if llm_params:
            param_groups.append({"params": llm_params, "lr": llm_lr, "betas": (llm_beta1, llm_beta2)})
        if not param_groups:
            raise ValueError("No trainable parameters were found for edit joint GRPO training.")

        self.optimizer = torch.optim.AdamW(param_groups, **optimizer_kwargs)
        return self.optimizer

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None):
        model.eval()
        self.ref_model.eval()
        # Re-pin frozen sub-modules to eval explicitly. `model.eval()` should
        # recurse, but DeepSpeed ZeRO-3 wrappers and gradient-checkpointing
        # hooks can flip individual sub-modules back to train mode silently.
        # Klein's VAE has a BatchNorm latent normalization whose running
        # stats drift in train mode → washed-out colors at decode time.
        for trainee in (model, self.ref_model):
            unwrapped = trainee.module if hasattr(trainee, "module") else trainee
            inner = unwrapped.get_model() if hasattr(unwrapped, "get_model") else unwrapped
            for component_name in ("visual", "vae", "text_encoder", "text_encoder_2", "text_encoder_3"):
                component = getattr(inner, component_name, None)
                if component is not None:
                    component.eval()
        # Restore VAE BatchNorm running stats both on the policy model and the
        # reference model. DeepSpeed ZeRO-3 wraps both and silently resets
        # buffers on frozen modules to their nn.BatchNorm defaults; without
        # this restore Klein's decoded latents skip reverse-whitening and
        # decode washed-out / desaturated.
        self._restore_vae_bn_buffers(model)
        self._restore_vae_bn_buffers(self.ref_model)
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

        # Per-section timing diagnostic. Set PROMPTRL_PROFILE=1 to enable.
        profile = os.getenv("PROMPTRL_PROFILE", "0") == "1"
        if profile:
            torch.cuda.synchronize()
            import time as _time
            t_start = _time.time()

        with self.compute_loss_context_manager():
            generations = self.generate_samples(model, inputs)
            torch.cuda.empty_cache()
            if profile:
                torch.cuda.synchronize()
                t_gen = _time.time()

            if self.num_refined > 0:
                cot_loss = self.cot_loss_computation(
                    model,
                    generations["prompt_completion_ids"],
                    generations["completion_ids"],
                    generations["prompt_length"],
                    generations["advantages_refined"],
                    generations["prompt_inputs"],
                )
                loss_update(cot_loss, 1.0)
            else:
                cot_loss = torch.tensor(0.0, device=self.accelerator.device)
            if profile:
                torch.cuda.synchronize()
                t_cot = _time.time()

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
                # Release the per-slice activations between micro-backwards.
                # Without this, the autograd graph + slice tensors live in
                # the caching allocator until end-of-loop, building up GB-sized
                # blocks at 1024² that fragment future allocations.
                del batched_states_slice, diff_loss, diff_kl
                torch.cuda.empty_cache()

        diff_loss = torch.stack(diff_loss_values).mean()
        diff_kl = torch.stack(diff_kl_values).mean()
        loss = diff_loss + cot_loss.detach()
        # Drop the large generations dict (prev_latents / pred_latents / batched_states
        # tensors are GB-sized at 1024²). Holding them through optimizer.step
        # prevents the allocator from reusing those blocks for the next step.
        del generations, diff_advantages, diff_loss_values, diff_kl_values
        torch.cuda.empty_cache()
        if profile:
            torch.cuda.synchronize()
            t_diff = _time.time()

        if self.args.torch_empty_cache_steps is not None and self.state.global_step % self.args.torch_empty_cache_steps == 0:
            torch.cuda.empty_cache()

        if hasattr(model, "step") and callable(model.step):
            model.step()
        if profile:
            torch.cuda.synchronize()
            t_step = _time.time()
            rank = self.accelerator.process_index if hasattr(self.accelerator, "process_index") else 0
            if rank == 0:
                print(
                    f"[profile step={self.state.global_step}] "
                    f"gen={t_gen-t_start:.1f}s cot={t_cot-t_gen:.1f}s "
                    f"diff_loss_loop={t_diff-t_cot:.1f}s optim_step={t_step-t_diff:.1f}s "
                    f"total={t_step-t_start:.1f}s",
                    flush=True,
                )

        self._metrics["diff_kl"].append(self.accelerator.gather_for_metrics(diff_kl).mean().item())
        self._metrics["diff_loss"].append(self.accelerator.gather_for_metrics(diff_loss).mean().item())
        # Per-step memory diagnostic — prints at most once every PROMPTRL_MEM_LOG_STEPS
        # (default 5). Helps catch leaks: if "reserved" keeps climbing across
        # steps with no plateau, there's a fragmentation issue we missed.
        log_every = int(os.getenv("PROMPTRL_MEM_LOG_STEPS", "5"))
        if log_every > 0 and self.state.global_step % log_every == 0:
            rank = self.accelerator.process_index if hasattr(self.accelerator, "process_index") else 0
            if rank == 0:
                alloc_gb = torch.cuda.memory_allocated() / 1024**3
                reserved_gb = torch.cuda.memory_reserved() / 1024**3
                max_alloc_gb = torch.cuda.max_memory_allocated() / 1024**3
                max_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
                print(
                    f"[mem step={self.state.global_step}] "
                    f"alloc={alloc_gb:.1f}GB reserved={reserved_gb:.1f}GB "
                    f"peak_alloc={max_alloc_gb:.1f}GB peak_reserved={max_reserved_gb:.1f}GB",
                    flush=True,
                )
                torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        return loss.detach()

    def generate_samples(self, model: nn.Module, inputs: List[Dict]) -> Dict[str, Any]:
        # One-time diagnostic at step 0 to verify VAE eval state and BN buffer
        # contents — Klein decodes through vae.bn running_mean/var, and any
        # drift here shows up as washed-out colors at decode time.
        if self.state.global_step == 0 and not getattr(self, "_vae_state_logged", False):
            self._vae_state_logged = True
            unwrapped = model.module if hasattr(model, "module") else model
            inner = unwrapped.get_model() if hasattr(unwrapped, "get_model") else unwrapped
            vae = getattr(inner, "vae", None)
            if vae is not None:
                bn = getattr(vae, "bn", None)
                rank = self.accelerator.process_index if hasattr(self.accelerator, "process_index") else 0
                print(
                    f"[VAE-diag rank{rank}] "
                    f"vae.training={vae.training} "
                    f"bn={bn is not None} "
                    + (
                        f"bn.training={bn.training} num_batches_tracked={int(bn.num_batches_tracked)} "
                        f"running_mean[:3]={bn.running_mean[:3].tolist()} "
                        f"running_var[:3]={bn.running_var[:3].tolist()}"
                        if bn is not None else ""
                    ),
                    flush=True,
                )
        source_images = [example["image"] for example in inputs]
        batch_size = len(inputs)
        prompt_inputs = None
        prompt_completion_ids = None
        completion_ids = None
        prompt_length = 0
        completions_refined: List[str] = []
        refined_prompts: List[str] = []

        if self.num_refined > 0:
            prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
            prompt_inputs = self.processing_class(
                images=[image for image in source_images for _ in range(self.num_refined)],
                text=[prompt for prompt in prompts_text for _ in range(self.num_refined)],
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            if self.max_prompt_length is not None:
                prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
                prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                with torch.no_grad():
                    prompt_completion_ids = unwrapped_model.generate(
                        **prompt_inputs,
                        generation_config=self.generation_config,
                    )

            prompt_length = prompt_inputs["input_ids"].size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]
            completions_refined = self.processing_class.tokenizer.batch_decode(
                completion_ids,
                skip_special_tokens=True,
            )
            refined_prompts = [self.model.extract_thinking_content(completion) for completion in completions_refined]

        original_prompts = [
            example["editing_instruction"]
            for example in inputs
            for _ in range(self.num_skip_refinement)
        ]
        all_prompts: List[str] = []
        for batch_idx in range(batch_size):
            refined_start = batch_idx * self.num_refined
            refined_end = refined_start + self.num_refined
            all_prompts.extend(refined_prompts[refined_start:refined_end])

            original_start = batch_idx * self.num_skip_refinement
            original_end = original_start + self.num_skip_refinement
            all_prompts.extend(original_prompts[original_start:original_end])

        all_source_images = [image for image in source_images for _ in range(self.num_generations)]
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            with torch.no_grad():
                (
                    edited_images,
                    prev_latents,
                    diff_sampling_log_probs,
                    pred_latents,
                    timesteps,
                    batched_states,
                ) = unwrapped_model.generate_image(
                    images=all_source_images,
                    texts=all_prompts,
                    diffusion_kwargs=self.diffusion_generation_config,
                    sde_sampling=True,
                )

        rewards, rewards_per_func = self.compute_rewards(inputs, edited_images, completions_refined)
        advantages = self.compute_advantages(rewards)
        advantages_refined = (
            advantages.view(batch_size, self.num_generations)[:, : self.num_refined].flatten()
            if self.num_refined > 0
            else torch.tensor([], device=advantages.device)
        )

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        for index, (func_name, _, _) in enumerate(self.reward_funcs):
            self._metrics[f"reward/{func_name}"].append(
                self.accelerator.gather_for_metrics(rewards_per_func[:, index]).mean().item()
            )
        self._log_samples(source_images, edited_images, all_prompts, advantages)

        return {
            "images": edited_images,
            "prev_latents": prev_latents,
            "diff_sampling_log_probs": diff_sampling_log_probs,
            "pred_latents": pred_latents,
            "batched_states": batched_states,
            "prompt_length": prompt_length,
            "completion_ids": completion_ids,
            "prompt_completion_ids": prompt_completion_ids,
            "prompt_inputs": prompt_inputs,
            "advantages": advantages,
            "advantages_refined": advantages_refined,
            "ts": timesteps,
        }

    def compute_rewards(
        self,
        inputs: List[Dict],
        edited_images: List[Image.Image],
        completions_refined: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(edited_images), len(self.reward_funcs), device=device)
        batch_size = len(inputs)

        for index, (func_name, _, reward_func) in enumerate(self.reward_funcs):
            if func_name == "format":
                refined_scores = torch.tensor(reward_func(completions_refined), device=device, dtype=torch.float32)
                for batch_idx in range(batch_size):
                    start = batch_idx * self.num_generations
                    refined_start = batch_idx * self.num_refined
                    refined_end = refined_start + self.num_refined
                    batch_refined_scores = refined_scores[refined_start:refined_end]
                    rewards_per_func[start : start + self.num_refined, index] = batch_refined_scores
                    rewards_per_func[start + self.num_refined : start + self.num_generations, index] = (
                        batch_refined_scores.mean() if len(batch_refined_scores) else 0.0
                    )
            elif func_name == "editreward":
                source_images = [example["image"] for example in inputs for _ in range(self.num_generations)]
                prompts = [example["editing_instruction"] for example in inputs for _ in range(self.num_generations)]
                rewards_per_func[:, index] = torch.tensor(
                    reward_func(source_images, edited_images, prompts)["scores"],
                    device=device,
                    dtype=torch.float32,
                )
            else:
                raise ValueError(f"Unsupported reward function for edit joint training: {func_name}")

        return rewards_per_func.sum(dim=1), rewards_per_func

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        grouped_rewards = rewards.view(-1, self.num_generations)
        mean = grouped_rewards.mean(dim=1).repeat_interleave(self.num_generations, dim=0)
        std = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(self.num_generations, dim=0)
        return torch.clamp((rewards - mean) / (std + 1e-4), -5, 5)

    def cot_loss_computation(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        prompt_length: int,
        advantages: torch.Tensor,
        prompt_inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        image_kwargs = {
            key: value for key, value in prompt_inputs.items() if key not in {"input_ids", "attention_mask"}
        }
        per_token_logps = self._get_per_token_logps(model, input_ids, image_kwargs)[:, prompt_length - 1 :]
        with torch.inference_mode():
            ref_per_token_logps = self._get_per_token_logps(self.ref_model, input_ids, image_kwargs)[:, prompt_length - 1 :]

        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        completion_mask = self._completion_mask(completion_ids)
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - 0.01 * per_token_kl)
        cot_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp_min(1)).mean()
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp_min(1)).mean()

        self._metrics["completion_length"].append(
            self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        )
        self._metrics["cot_kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        self._metrics["cot_loss"].append(self.accelerator.gather_for_metrics(cot_loss).mean().item())
        return cot_loss

    def _get_per_token_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        image_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        logits = model(input_ids, **image_kwargs).logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        per_token_logps = []
        for logits_row, target_ids_row in zip(logits, target_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            per_token_logps.append(torch.gather(log_probs, dim=1, index=target_ids_row.unsqueeze(1)).squeeze(1))
        return torch.stack(per_token_logps)

    def _completion_mask(self, completion_ids: torch.Tensor) -> torch.Tensor:
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = completion_ids.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        return (sequence_indices <= eos_idx.unsqueeze(1)).int()

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

        _, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(
            model_pred,
            self.scheduler,
            prev_latents[:, : pred_latents.size(1)],
            pred_latents,
            timesteps,
            noise_scale=self.sde_noise_scale,
        )
        _, _, ref_prev_sample_mean, ref_std_dev_t = compute_log_prob(
            ref_model_pred,
            self.scheduler,
            prev_latents[:, : pred_latents.size(1)],
            pred_latents,
            timesteps,
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

    def _log_samples(
        self,
        source_images: List[Image.Image],
        edited_images: List[Image.Image],
        prompts: List[str],
        advantages: torch.Tensor,
    ) -> None:
        global_step = self.state.global_step
        if global_step % 10 != 0 or not edited_images:
            return

        device_id = str(self.accelerator.device).replace(":", "")
        text_content = []
        for batch_idx in range(len(source_images)):
            for gen_idx in range(self.num_generations):
                overall_idx = batch_idx * self.num_generations + gen_idx
                status = "REFINED" if gen_idx < self.num_refined else "ORIGINAL"
                text_content.append(f"[{status}] Generation {gen_idx}: {prompts[overall_idx]}")
            text_content.append("")

        txt_path = os.path.join(self.log_dir, f"step_{global_step}_{device_id}.txt")
        if not os.path.exists(txt_path):
            with open(txt_path, "w", encoding="utf-8") as file:
                file.write("\n".join(text_content))

        for batch_idx, source_image in enumerate(source_images):
            source_image.save(os.path.join(self.log_dir, f"step_{global_step}_{device_id}_batch{batch_idx}_source.jpg"))
            for gen_idx in range(self.num_generations):
                overall_idx = batch_idx * self.num_generations + gen_idx
                prefix = "refined" if gen_idx < self.num_refined else "original"
                edited_images[overall_idx].save(
                    os.path.join(
                        self.log_dir,
                        f"step_{global_step}_{device_id}_batch{batch_idx}_{prefix}_gen{gen_idx}_{advantages[overall_idx].item():.5f}.jpg",
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


# Back-compat alias: external scripts and earlier docs reference this name.
QwenKontextEditGRPOTrainer = EditJointGRPOTrainer

