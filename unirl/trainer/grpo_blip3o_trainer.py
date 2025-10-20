# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""
Unified GRPO Trainer for BLIP3o Models (Text-to-Image and Image-to-Image)

This module implements Group Relative Policy Optimization (GRPO) for vision-language
models, supporting both text-to-image generation and image-to-image editing tasks.

"""

import os
import sys
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime

import torch
import torch.distributed as dist
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoProcessor,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)


import transformers
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from transformers import CLIPModel, CLIPTextModel

from deepspeed.runtime.zero import GatheredParameters
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode


from trl.data_utils import maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from unimodel.blip3o.model import blip3oQwenForInferenceLM
from unimodel.blip3o.model import sde_step_with_logprob
from unimodel.blip3o.constants import UND_IMAGE_TOKEN, UND_IMAGE_TOKEN_IDX

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb


# Type aliases
RewardFunc = Union[str, PreTrainedModel, Callable]
RewardFuncSpec = Union[RewardFunc, List[RewardFunc]]


def compute_log_prob(model_pred: torch.Tensor, 
                     scheduler: FlowMatchEulerDiscreteScheduler,
                     prev_latents: torch.Tensor, 
                     pred_latents: torch.Tensor, 
                     ts: torch.Tensor):
    """
    Compute log probability and related statistics for SDE step.
    
    Args:
        model_pred: Model prediction output
        scheduler: Flow matching scheduler
        prev_latents: Previous latent state
        pred_latents: Predicted latent state
        ts: Timestep
        
    Returns:
        Tuple of (prev_sample, log_prob, prev_sample_mean, std_dev_t)
    """ 
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        scheduler,
        model_pred.float(),
        ts,
        prev_latents.float(),
        pred_latents.float(),
    )
    return prev_sample, log_prob, prev_sample_mean, std_dev_t


class BaseBLIP3oGRPOTrainer(Trainer):
    """
    Base trainer for GRPO with BLIP3o models.
    
    This class provides common functionality for both T2I and I2I variants.
    """
    
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: RewardFuncSpec,
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, Dict]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: tuple = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: int = 12845056,
        min_pixels: int = 3136,
        attn_implementation: str = "flash_attention_2",
        task_type: str = "t2i",  # "t2i" or "i2i"
    ):
        """
        Initialize BLIP3o GRPO Trainer.
        
        Args:
            model: Model ID or PreTrainedModel instance
            reward_funcs: Reward function(s) for optimization
            args: Training configuration
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            processing_class: Tokenizer/processor
            callbacks: Training callbacks
            optimizers: Optimizer and scheduler tuple
            peft_config: PEFT configuration for parameter-efficient training
            max_pixels: Maximum pixels for image processing
            min_pixels: Minimum pixels for image processing
            attn_implementation: Attention implementation type
            task_type: Task type ("t2i" or "i2i")
        """
        # Configuration
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        
        self.task_type = task_type
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.beta = args.beta
        
        # Model initialization
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            if "BLIP3o".lower() not in model_id.lower():
                raise ValueError(f"Only BLIP3o models are supported, got: {model_id}")
            
            model = blip3oQwenForInferenceLM.from_pretrained(model_id, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
        
        if peft_config is not None:
            model = get_peft_model(model, peft_config)
        
        # Reference model
        if is_deepspeed_zero3_enabled():
            self.ref_model = blip3oQwenForInferenceLM.from_pretrained(model_id, **model_init_kwargs)
        else:
            self.ref_model = create_reference_model(model)
        
        # Freeze/unfreeze parameters based on task
        self._configure_parameters(model)
        
        # Scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
        self.diffusion_config = {
            "guidance_scale": 3.5,
            "num_inference_steps": 10,
            "num_images_per_prompt": self.num_generations,
        }
        self.scheduler.set_timesteps(self.diffusion_config["num_inference_steps"])
        
        # Processing class
        if processing_class is None:
            processor_id = "Qwen/Qwen2.5-VL-3B-Instruct"
            processing_class = AutoProcessor.from_pretrained(processor_id)
            processing_class.pad_token_id = processing_class.tokenizer.pad_token_id
            processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            processing_class.image_processor.max_pixels = max_pixels
            processing_class.image_processor.min_pixels = min_pixels
        
        self.reward_funcs = reward_funcs
        
        # Image transforms
        self._setup_transforms()
        
        # Logging
        self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.log_dir = os.path.join(args.output_dir, "training_samples", self.start_time)
        os.makedirs(self.log_dir, exist_ok=True)
        self._metrics = defaultdict(list)
        
        # Initialize parent
        def data_collator(features):
            return features
        
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        
        self.model_accepts_loss_kwargs = False
        
        # Prepare reference model
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
    
    def _configure_parameters(self, model: PreTrainedModel):
        """Configure which parameters to train based on task type."""
        # Freeze reference model
        for p in self.ref_model.parameters():
            p.requires_grad = False
        
        # Get model components
        model_base = model.get_model()
        
        # Freeze base components
        for p in model_base.parameters():
            p.requires_grad = False
        for p in model.visual.parameters():
            p.requires_grad = False
        for p in model.lm_head.parameters():
            p.requires_grad = False
        
        # Unfreeze task-specific components
        if self.task_type == "t2i":
            model_base.down_projector.requires_grad_(True)
            model_base.t2i_queries.requires_grad = True
            model_base.dit.requires_grad_(True)
        elif self.task_type == "i2i":
            model_base.down_projector.requires_grad_(True)
            model_base.i2i_queries.requires_grad = True
            model_base.dit.requires_grad_(True)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def _setup_transforms(self):
        """Setup image transformation pipelines."""
        self.ref_trsf = T.Compose([
            T.Lambda(lambda x: x.convert("RGB")),
            T.Resize(1024, interpolation=InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(1024),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2 - 1)
        ])
        
        self.und_trsf = T.Compose([
            T.Lambda(lambda x: x.convert("RGB")),
            T.Resize(672, interpolation=InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(672)
        ])
    
    def _set_signature_columns_if_needed(self):
        """Set required dataset columns."""
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]
    
    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Skip automatic tensor conversion."""
        return inputs
    
    def _compute_rewards(self, inputs: List[Dict], images: List[Any]) -> torch.Tensor:
        """
        Compute rewards from all reward functions.
        
        Args:
            inputs: Input batch
            images: Generated images
            
        Returns:
            Tensor of shape (batch_size * num_generations,) with rewards
        """
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(images), len(self.reward_funcs), device=device)
        
        # Extract metadata
        captions = [ex.get("caption", ex.get("target_caption", "")) for ex in inputs]
        
        for i, (func_name, _, reward_func) in enumerate(self.reward_funcs):
            if func_name == "jpeg_compressibility" or func_name == "jpeg_incompressibility":
                rewards_per_func[:, i] = reward_func(images)
            elif func_name in ["pickscore", "hps", "deqa", "image_reward", "aesthetic"]:
                scores = reward_func(
                    images, 
                    [cap for cap in captions for _ in range(self.num_generations)]
                )["scores"]
                rewards_per_func[:, i] = torch.tensor(scores).to(device)
            elif func_name == "gen_eval":
                meta_files = [ex.get("metadata") for ex in inputs]
                meta_input = {"meta_datas": [m for m in meta_files for _ in range(self.num_generations)]}
                scores = reward_func(
                    images,
                    [cap for cap in captions for _ in range(self.num_generations)],
                    meta_input
                )["scores"]
                rewards_per_func[:, i] = torch.tensor(scores).to(device)
        
        # Aggregate rewards (can be customized)
        return rewards_per_func.sum(dim=1), rewards_per_func
    
    def _compute_diffusion_loss(
        self, 
        model_to_use: PreTrainedModel,
        prompts_text: List[str],
        prev_latents: torch.Tensor,
        pred_latents: torch.Tensor,
        ts: torch.Tensor,
        query_attr: str,
        embedding_repeat_num: int = 1,
        **kwargs
    ):
        """
        Compute diffusion model predictions and log probabilities.
        
        Args:
            model_to_use: Model to use for prediction
            prompts_text: Text prompts
            prev_latents: Previous latent states
            pred_latents: Predicted latent states
            ts: Timesteps
            query_attr: Query attribute name ("t2i_queries" or "i2i_queries")
            **kwargs: Additional inputs (e.g., pixel_values for I2I)
            
        Returns:
            Tuple of (log_prob, kl_divergence)
        """
        device = self.accelerator.device
        
        # Text embeddings
        text_inputs = self.processing_class.tokenizer(
            prompts_text, 
            padding="longest", 
            return_tensors="pt"
        ).to(device)
        text_embeds = model_to_use.get_model().embed_tokens(text_inputs.input_ids)
        attention_mask = text_inputs.attention_mask
        
        # Handle I2I case with understanding images
        if self.task_type == "i2i" and "pixel_values" in kwargs:
            und_image_idx = (text_inputs.input_ids == UND_IMAGE_TOKEN_IDX)
            und_pixel_values = kwargs["pixel_values"].type(model_to_use.visual.dtype)
            und_image_embeds = model_to_use.visual(
                und_pixel_values, 
                grid_thw=kwargs["image_grid_thw"]
            )
            text_embeds[und_image_idx] = und_image_embeds[:und_image_idx.sum(), :]
        
        # Query embeddings
        with GatheredParameters([getattr(model_to_use.get_model(), query_attr)], modifier_rank=None):
            queries = getattr(model_to_use.get_model(), query_attr)
            latent_queries = queries.repeat(text_embeds.shape[0], 1, 1)
        
        # Combine embeddings
        text_embeds = torch.cat([text_embeds, latent_queries], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(latent_queries[:, :, 0])], dim=1)
        
        # Forward pass
        outputs = model_to_use.model(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Extract and project hidden states
        n_query = model_to_use.get_n_query()
        hidden_states = outputs.hidden_states[-1][:, -n_query:, :]
        img_hidden_states = model_to_use.get_model().down_projector(hidden_states)
        
        # Repeat for all steps
        num_steps = embedding_repeat_num * self.diffusion_config["num_inference_steps"]
        img_hidden_states = img_hidden_states.repeat_interleave(num_steps, dim=0)
        img_attention_mask = torch.ones(
            (img_hidden_states.shape[0], img_hidden_states.shape[1]),
            device=device,
            dtype=img_hidden_states.dtype
        )
        
        # DiT predictions
        dit_kwargs = {
            "hidden_states": prev_latents.to(device),
            "encoder_hidden_states": img_hidden_states,
            "encoder_attention_mask": img_attention_mask,
            "timestep": ts.to(device),
            "return_dict": False,
        }
        
        # Add reference latents for I2I
        if self.task_type == "i2i" and "ref_latents" in kwargs:
            dit_kwargs["ref_hidden_states"] = kwargs["ref_latents"]
        
        # Conditional prediction
        model_pred_cond = model_to_use.get_model().dit(**dit_kwargs)[0]
        
        # Unconditional prediction
        dit_kwargs["encoder_hidden_states"] = torch.zeros_like(img_hidden_states)
        model_pred_uncond = model_to_use.get_model().dit(**dit_kwargs)[0]
        
        # Apply classifier-free guidance
        guidance_scale = self.diffusion_config["guidance_scale"]
        model_pred = model_pred_uncond + guidance_scale * (model_pred_cond - model_pred_uncond)
        
        return model_pred
    def _log_step(self, images, prompts_text, advantages, completions):
        global_step = self.state.global_step
        
        if not global_step % 5 == 0:
            return 
    
        device_id = str(self.model.device).replace(":", "")
        
        
        log_dir = self.log_dir
        
        text_content = f"Prompt: {prompts_text[0]}"
        
        if completions is not None:
            for idx in range (self.num_generations):
                text_content += f"\nCompletion {idx}: {completions[idx]}"
            
        if os.path.exists(os.path.join(log_dir, f"step_{global_step}_{device_id}.txt")):
            return 
        with open(os.path.join(log_dir, f"step_{global_step}_{device_id}.txt"), "w", encoding="utf-8") as f:
            f.write(text_content)
            
        for idx in range(self.num_generations):
            rev_img = images[idx]
            rev_img_pil = rev_img 
            advantage = advantages[idx]
            rev_img_pil.save(os.path.join(log_dir, f"step_{global_step}_{device_id}_{advantage.item()}_{idx}.jpg"))

    def compute_loss(
        self, 
        model: PreTrainedModel, 
        inputs: List[Dict[str, Any]], 
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None
    ):
        """
        Compute GRPO loss.
        
        Args:
            model: Policy model
            inputs: Batch of inputs
            return_outputs: Whether to return outputs (not supported)
            num_items_in_batch: Number of items in batch
            
        Returns:
            Loss tensor
        """
        if return_outputs:
            raise ValueError("GRPOTrainer does not support returning outputs")
        
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement compute_loss")
    
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None):
        """Log metrics with custom tracking."""
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        
        self._metrics.clear()


class T2IGRPOTrainer(BaseBLIP3oGRPOTrainer):
    """GRPO Trainer for Text-to-Image generation."""
    
    def __init__(self, *args, **kwargs):
        kwargs["task_type"] = "t2i"
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute T2I GRPO loss."""
        if return_outputs:
            raise ValueError("GRPOTrainer does not support returning outputs")
        
        device = self.accelerator.device
        prompts_text = [
            maybe_apply_chat_template(ex, self.processing_class)["prompt"] 
            for ex in inputs
        ]
        
        # Generate images
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            with torch.no_grad():
                images, log_probs_traj, prev_latents, pred_latents, ts = \
                    unwrapped_model.generate_image(
                        text=prompts_text,
                        tokenizer=self.processing_class.tokenizer,
                        diffusion_kwargs=self.diffusion_config,
                        use_sde=True,
                    )
        
        # Compute rewards and advantages
        rewards, rewards_per_func = self._compute_rewards(inputs, images)
        reshaped_rewards = rewards.view(-1, self.num_generations)
        mean_rewards = reshaped_rewards.mean(dim=1).repeat_interleave(self.num_generations)
        std_rewards = reshaped_rewards.std(dim=1).repeat_interleave(self.num_generations)
        advantages = (rewards - mean_rewards) / (std_rewards + 1e-4)
        advantages = torch.clamp(advantages, -5, 5)
        
        
        self._log_step(images, prompts_text, advantages, None)
        
        # Compute policy predictions
        model_pred = self._compute_diffusion_loss(
            model, prompts_text, prev_latents, pred_latents, ts, "t2i_queries", self.num_generations
        )
        
        # Compute reference predictions
        with torch.no_grad():
            ref_model_pred = self._compute_diffusion_loss(
                self.ref_model, prompts_text, prev_latents, pred_latents, ts, "t2i_queries", self.num_generations
            )
        
        # Compute log probs and KL
        _, log_prob_policy, mean_policy, std_policy = compute_log_prob(
            model_pred, self.scheduler, prev_latents, pred_latents, ts
        )
        _, _, mean_ref, std_ref = compute_log_prob(
            ref_model_pred, self.scheduler, prev_latents, pred_latents, ts
        )
        
        kl = (mean_policy - mean_ref)**2 / (2 * std_policy**2)
        kl = kl.mean(dim=tuple(range(1, kl.ndim)))
        
        # GRPO loss
        advantages_steps = advantages.repeat_interleave(
            self.diffusion_config["num_inference_steps"], dim=0
        )
        ratio = torch.exp(log_prob_policy - log_probs_traj)
        assert (ratio == 1).all(), f"{ratio}"
        unclipped_loss = -advantages_steps * ratio
        clipped_loss = -advantages_steps * torch.clamp(ratio, 1.0 - 1e-4, 1.0 + 1e-4)
        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
        
        loss = policy_loss + self.beta * kl.mean()
        
        # Logging
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(kl).mean().item())
        
        for i, (func_name, _, _) in enumerate(self.reward_funcs):
            self._metrics[f"reward/{func_name}"].append(
                self.accelerator.gather_for_metrics(
                    rewards_per_func[:, i]
                ).mean().item())
        return loss


class T2ICoTGRPOTrainer(BaseBLIP3oGRPOTrainer):
    """
    GRPO Trainer for Text-to-Image with Chain-of-Thought reasoning.
    
    This trainer combines CoT text generation with T2I diffusion, optimizing both
    the reasoning process and the image generation jointly.
    """
    
    def __init__(self, *args, **kwargs):
        kwargs["task_type"] = "t2i"
        super().__init__(*args, **kwargs)
        
        # Setup generation config for CoT
        self.generation_config = {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 1.0,
            "num_return_sequences": self.num_generations,
            "pad_token_id": self.processing_class.pad_token_id,
            "eos_token_id": self.processing_class.eos_token_id,
        }
        self.diffusion_config = {
            "guidance_scale": 3.5,
            "num_inference_steps": 10,
            "num_images_per_prompt": 1,
        }
    
    def _configure_parameters(self, model: PreTrainedModel):
        """Configure parameters for CoT + T2I training."""
        # Freeze reference model
        for p in self.ref_model.parameters():
            p.requires_grad = False
        
        # Get model components
        model_base = model.get_model()
        
        # For CoT+T2I, we train both language and diffusion components
        for p in model_base.parameters():
            p.requires_grad = True
        for p in model.visual.parameters():
            p.requires_grad = True
        for p in model.lm_head.parameters():
            p.requires_grad = True
        
        # T2I components
        model_base.down_projector.requires_grad_(True)
        model_base.t2i_queries.requires_grad = True
        model_base.dit.requires_grad_(True)
    
    def create_optimizer(self):
        """
        Create optimizer with different learning rates for DiT and LLM.
        DiT gets higher LR (1e-5), LLM gets lower LR (1e-6).
        """
        opt_kwargs = {
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "eps": self.args.adam_epsilon,
            "weight_decay": self.args.weight_decay,
        }
        
        dit_params = []
        llm_params = []
        
        # Collect DiT parameters
        for name, param in self.model.get_model().dit.named_parameters():
            if param.requires_grad:
                dit_params.append(param)
        
        for name, param in self.model.get_model().down_projector.named_parameters():
            if param.requires_grad:
                dit_params.append(param)
        
        if self.model.get_model().t2i_queries.requires_grad:
            dit_params.append(self.model.get_model().t2i_queries)
        
        # Collect LLM parameters
        for name, param in self.model.get_model().named_parameters():
            if param.requires_grad and "dit" not in name and "queries" not in name and "down_projector" not in name:
                llm_params.append(param)
        
        # Create parameter groups with different LRs
        param_groups = [
            {"params": dit_params, "lr": 3e-6},
            {"params": llm_params, "lr": 5e-7},
        ]
        
        optimizer = torch.optim.AdamW(param_groups, **opt_kwargs)
        return optimizer
    def _compute_rewards(self, inputs: List[Dict], images: List[Any], completions: List[str]) -> torch.Tensor:
        """
        Compute rewards from all reward functions.
        
        Args:
            inputs: Input batch
            images: Generated images
            
        Returns:
            Tensor of shape (batch_size * num_generations,) with rewards
        """
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(images), len(self.reward_funcs), device=device)
        
        # Extract metadata
        captions = [ex.get("caption", ex.get("target_caption", "")) for ex in inputs]
        
        for i, (func_name, _, reward_func) in enumerate(self.reward_funcs):
            if func_name == "jpeg_compressibility" or func_name == "jpeg_incompressibility":
                rewards_per_func[:, i] = reward_func(images)
            elif func_name in ["pickscore", "hps", "deqa", "image_reward", "aesthetic"]:
                scores = reward_func(
                    images, 
                    [cap for cap in captions for _ in range(self.num_generations)]
                )["scores"]
                rewards_per_func[:, i] = torch.tensor(scores).to(device)
            elif func_name == "format":
                rewards_per_func[:, i] = torch.tensor(reward_func(completions)).to(device)
            elif func_name == "gen_eval":
                meta_files = [ex.get("metadata") for ex in inputs]
                meta_input = {"meta_datas": [m for m in meta_files for _ in range(self.num_generations)]}
                scores = reward_func(
                    images,
                    [cap for cap in captions for _ in range(self.num_generations)],
                    meta_input
                )["scores"]
                rewards_per_func[:, i] = torch.tensor(scores).to(device)
        
        # Aggregate rewards (can be customized)
        return rewards_per_func.sum(dim=1), rewards_per_func
    
    def _compute_cot_loss(
        self,
        model: PreTrainedModel,
        prompt_completion_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        advantages: torch.Tensor,
        prompt_length: int
    ):
        """
        Compute CoT loss on text generation.
        
        Args:
            model: Policy model
            prompt_completion_ids: Full prompt+completion token IDs
            completion_ids: Completion-only token IDs
            advantages: Advantage values for each generation
            prompt_length: Length of prompt tokens
            
        Returns:
            Tuple of (cot_loss, mean_kl, completion_mask)
        """
        def get_per_token_logps(model_class, input_ids):
            """Get log probabilities for each token."""
            logits = model_class(input_ids).logits[:, :-1, :]
            input_ids = input_ids[:, 1:]
            
            per_token_logps = []
            for logits_row, input_ids_row in zip(logits, input_ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(
                    log_probs, dim=1, index=input_ids_row.unsqueeze(1)
                ).squeeze(1)
                per_token_logps.append(token_log_prob)
            
            return torch.stack(per_token_logps)
        
        # Policy log probs
        per_token_logps = get_per_token_logps(model, prompt_completion_ids)
        per_token_logps = per_token_logps[:, prompt_length - 1:]
        
        # Reference log probs
        with torch.inference_mode():
            ref_per_token_logps = get_per_token_logps(self.ref_model, prompt_completion_ids)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
        
        # KL divergence
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - \
                      (ref_per_token_logps - per_token_logps) - 1
        
        # Mask everything after first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # Compute loss
        advantages_cot = advantages.unsqueeze(1)
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages_cot
        per_token_loss = -(per_token_loss - 0.01 * per_token_kl)
        cot_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        return cot_loss, mean_kl, completion_mask
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute CoT + T2I GRPO loss."""
        if return_outputs:
            raise ValueError("GRPOTrainer does not support returning outputs")
        
        device = self.accelerator.device
        
        # Prepare prompts for CoT generation
        prompts_text = [
            maybe_apply_chat_template(ex, self.processing_class)["prompt"] 
            for ex in inputs
        ]
        
        prompt_inputs = self.processing_class.tokenizer(
            prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        ).to(device)
        
        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length:]
        
        # Generate CoT completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            with torch.no_grad():
                from transformers import GenerationConfig
                gen_config = GenerationConfig(**self.generation_config)
                unwrapped_model.generation_config = gen_config
                completion = unwrapped_model.generate(**prompt_inputs, generation_config=gen_config)
                
                # Pad if needed
                max_length = completion.size(1)
                prompt_completion_ids = completion
        
        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        
        # Decode completions for T2I
        completions = self.processing_class.tokenizer.batch_decode(
            prompt_completion_ids, skip_special_tokens=True
        )
        
        # Generate images from CoT completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            with torch.no_grad():
                images, diff_log_probs_traj, prev_latents, pred_latents, ts = \
                    unwrapped_model.generate_image(
                        text=completions,
                        tokenizer=self.processing_class.tokenizer,
                        diffusion_kwargs=self.diffusion_config,
                        use_sde=True,
                    )
        
        # Compute rewards and advantages
        rewards, rewards_per_func = self._compute_rewards(inputs, images, completions)
        reshaped_rewards = rewards.view(-1, self.num_generations)
        mean_rewards = reshaped_rewards.mean(dim=1).repeat_interleave(self.num_generations)
        std_rewards = reshaped_rewards.std(dim=1).repeat_interleave(self.num_generations)
        advantages = (rewards - mean_rewards) / (std_rewards + 1e-4)
        advantages = torch.clamp(advantages, -5, 5)
        
        
        self._log_step(images, prompts_text, advantages, completions)

        # CoT loss computation
        cot_loss, mean_kl_cot, completion_mask = self._compute_cot_loss(
            model, prompt_completion_ids, completion_ids, advantages, prompt_length
        )
        
        # Diffusion loss computation
        model_pred = self._compute_diffusion_loss(
            model, completions, prev_latents, pred_latents, ts, "t2i_queries", 1
        )
        
        with torch.no_grad():
            ref_model_pred = self._compute_diffusion_loss(
                self.ref_model, completions, prev_latents, pred_latents, ts, "t2i_queries", 1
            )
        
        # Compute log probs and KL for diffusion
        _, log_prob_diff, mean_diff, std_diff = compute_log_prob(
            model_pred, self.scheduler, prev_latents, pred_latents, ts
        )
        _, _, mean_ref_diff, std_ref_diff = compute_log_prob(
            ref_model_pred, self.scheduler, prev_latents, pred_latents, ts
        )
        
        kl_diff = (mean_diff - mean_ref_diff)**2 / (2 * std_diff**2)
        kl_diff = kl_diff.mean(dim=tuple(range(1, kl_diff.ndim)))
        
        # Diffusion GRPO loss
        advantages_diff = advantages.repeat_interleave(
            self.diffusion_config["num_inference_steps"], dim=0
        )
        ratio_diff = torch.exp(log_prob_diff - diff_log_probs_traj)
        assert (ratio_diff == 1).all(), f"{ratio_diff}"

        unclipped_loss_diff = -advantages_diff * ratio_diff
        clipped_loss_diff = -advantages_diff * torch.clamp(
            ratio_diff, 1.0 - 1e-4, 1.0 + 1e-4
        )
        diff_loss = torch.mean(torch.maximum(unclipped_loss_diff, clipped_loss_diff))
        diff_loss = diff_loss + self.beta * kl_diff.mean()
        
        # Combined loss
        loss = cot_loss + diff_loss
        
        # Logging
        completion_length = self.accelerator.gather_for_metrics(
            completion_mask.sum(1)
        ).float().mean().item()
        
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["cot_loss"].append(self.accelerator.gather_for_metrics(cot_loss).mean().item())
        self._metrics["cot_kl"].append(self.accelerator.gather_for_metrics(mean_kl_cot).mean().item())
        self._metrics["diff_loss"].append(self.accelerator.gather_for_metrics(diff_loss).mean().item())
        self._metrics["diff_kl"].append(self.accelerator.gather_for_metrics(kl_diff).mean().item())
        self._metrics["completion_length"].append(completion_length)
        
        for i, (func_name, _, _) in enumerate(self.reward_funcs):
            self._metrics[f"reward/{func_name}"].append(
                self.accelerator.gather_for_metrics(
                    rewards_per_func[:, i]
                ).mean().item()
            )
        
        return loss
    
    
    
    
class I2IGRPOTrainer(BaseBLIP3oGRPOTrainer):
    """GRPO Trainer for Image-to-Image editing."""
    
    def __init__(self, *args, **kwargs):
        kwargs["task_type"] = "i2i"
        super().__init__(*args, **kwargs)
        
        self.diffusion_generation_config = dict(
            guidance_scale=3.0,
            num_inference_steps=10,
            num_images_per_prompt=self.num_generations,
            guidance_scale_ref=1.0
        )
        self.reverse_diffusion_generation_config = dict(
            guidance_scale=3.0,
            num_inference_steps=10,
            num_images_per_prompt=1,
            guidance_scale_ref=1.0
        )
        self.scheduler.set_timesteps(self.diffusion_generation_config["num_inference_steps"])
        
        if any(name == "clip_sim" for name, _, _ in self.reward_funcs) or any(name == "sim_direction" for name, _, _ in self.reward_funcs):
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.accelerator.device)
            if self.is_deepspeed_enabled:
                self.clip_model = prepare_deepspeed(self.clip_model, self.accelerator)
            else:
                self.clip_model = self.accelerator.prepare_model(self.clip_model, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "reverse_prompt", "image", "caption", "target_caption"]

    def _log_step(self, und_images, edited_images, recon_images, prompts_text, reverse_prompts_text, advantages):
        global_step = self.state.global_step
        if global_step % 10 != 0:
            return

        device_id = str(self.model.device).replace(":", "")
        log_dir = self.log_dir

        text_content = f"Prompt: {prompts_text[0]}\nReverse Prompt: {reverse_prompts_text[0]}"
        if os.path.exists(os.path.join(log_dir, f"step_{global_step}_{device_id}.txt")):
            return 
        with open(os.path.join(log_dir, f"step_{global_step}_{device_id}.txt"), "w", encoding="utf-8") as f:
            f.write(text_content)

        for idx in range(self.num_generations):
            orig_img = und_images[0]
            rev_img = edited_images[idx]
            recon_img = recon_images[idx]
            orig_img_pil = orig_img
            rev_img_pil = rev_img
            recon_img_pil = recon_img
            orig_img_pil.save(os.path.join(log_dir, f"step_{global_step}_{device_id}_orig_{idx}.jpg"))
            rev_img_pil.save(os.path.join(log_dir, f"step_{global_step}_{device_id}_reverse_{idx}.jpg"))
            recon_img_pil.save(os.path.join(log_dir, f"step_{global_step}_{device_id}_recon_{idx}_{advantages[idx].item()}.jpg"))

    def _compute_rewards(self, inputs: List[Dict], edited_images: List[Any], recon_images: List[Any]) -> torch.Tensor:
        """
        Compute rewards from all reward functions.
        
        Args:
            inputs: Input batch
            edited_images: Generated edited images
            recon_images: Reconstructed images
            
        Returns:
            Tuple of (rewards, rewards_per_func)
        """
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(recon_images), len(self.reward_funcs), device=device)
        src_captions = [x["caption"] for x in inputs]
        target_captions = [x["target_caption"] for x in inputs]

        for i, (func_name, reward_processing, reward_func) in enumerate(self.reward_funcs):
            if func_name == "recon":
                ref_images_inputs = torch.cat([self.ref_trsf(x["image"]).unsqueeze(0) for x in inputs], dim=0)
                ref_images_inputs = (ref_images_inputs.repeat_interleave(self.num_generations, dim=0) + 1) / 2
                recon_images_inputs = torch.cat([reward_processing(image).unsqueeze(0) for image in recon_images], dim=0)
                rewards_per_func[:, i] = reward_func(ref_images_inputs, recon_images_inputs)
            elif func_name == "jpeg_compressibility":
                rewards_per_func[:, i] = reward_func(edited_images)
            elif func_name in ["pickscore", "hps", "deqa", "unifiedreward_sglang", "ocr", "image_reward", "aesthetic"]:
                rewards_per_func[:, i] = torch.tensor(
                    reward_func(edited_images, [target_caption for target_caption in target_captions for _ in range(self.num_generations)])["scores"]
                ).to(device)
            elif func_name == "clip_sim":
                rewards_per_func[:, i] = reward_func(
                    recon_images, [example["image"] for example in inputs for _ in range(self.num_generations)],
                    self.clip_model, reward_processing
                ).to(device)
            elif func_name == "sim_direction":
                rewards_per_func[:, i] = reward_func(
                    [example["image"] for example in inputs for _ in range(self.num_generations)],
                    edited_images,
                    [src_caption for src_caption in src_captions for _ in range(self.num_generations)],
                    [target_caption for target_caption in target_captions for _ in range(self.num_generations)],
                    self.clip_model,
                    reward_processing
                ).to(device)
            else:
                raise NotImplementedError(f"Reward function {func_name} not implemented")

        rewards = rewards_per_func[:, 0] * 4 + rewards_per_func[:, 1] * 1
        return rewards, rewards_per_func

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        src_captions = [x["caption"] for x in inputs]
        target_captions = [x["target_caption"] for x in inputs]
        prompts = [x["prompt"] for x in inputs]
        reverse_prompts = [x["reverse_prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"].replace(
                UND_IMAGE_TOKEN, UND_IMAGE_TOKEN * (672 * 672 // 28 // 28)
            ) for example in inputs
        ]
        reverse_inputs = [{"prompt": reverse_prompt} for reverse_prompt in reverse_prompts]
        reverse_prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"].replace(
                UND_IMAGE_TOKEN, UND_IMAGE_TOKEN * (672 * 672 // 28 // 28)
            ) for example in reverse_inputs
        ]

        und_images = [self.und_trsf(x["image"]) for x in inputs]
        und_image_inputs = self.processing_class.image_processor(images=und_images, return_tensors="pt")
        und_image_grid_thw = und_image_inputs.image_grid_thw.to(self.model.device)
        und_pixel_values = und_image_inputs.pixel_values.to(self.model.device)

        ref_images = torch.cat([self.ref_trsf(x["image"]).unsqueeze(0) for x in inputs], dim=0)
        with torch.no_grad():
            ref_latents = self.model.get_model().gen_vision_tower(ref_images.to(self.model.device))

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            with torch.no_grad():
                edited_images, log_probs, prev_latents, pred_latents, ts, noisy_ref_latents = unwrapped_model.generate_image(
                    text=prompts_text,
                    tokenizer=self.processing_class.tokenizer,
                    pixel_values=und_pixel_values,
                    ref_latents=ref_latents,
                    image_grid_thw=und_image_grid_thw,
                    diffusion_kwargs=self.diffusion_generation_config,
                    use_sde = True
                )
                und_edited_images = [self.und_trsf(x) for x in edited_images]
                ref_edited_images = torch.cat([self.ref_trsf(x).unsqueeze(0) for x in edited_images], dim=0)
                ref_edited_latents = self.model.get_model().gen_vision_tower(ref_edited_images.to(self.model.device))

                und_edited_image_inputs = self.processing_class.image_processor(images=und_edited_images, return_tensors="pt")
                und_edited_image_grid_thw = und_edited_image_inputs.image_grid_thw.to(self.model.device)
                und_edited_pixel_values = und_edited_image_inputs.pixel_values.to(self.model.device)

                recon_images, reverse_log_probs, reverse_prev_latents, reverse_pred_latents, reverse_ts, reverse_noisy_ref_latents = unwrapped_model.generate_image(
                    text=[reverse_prompt_text for reverse_prompt_text in reverse_prompts_text for _ in range(self.num_generations)],
                    tokenizer=self.processing_class.tokenizer,
                    pixel_values=und_edited_pixel_values,
                    ref_latents=ref_edited_latents,
                    image_grid_thw=und_edited_image_grid_thw,
                    diffusion_kwargs=self.reverse_diffusion_generation_config,
                    use_sde = True
                )

        rewards, rewards_per_func = self._compute_rewards(inputs, edited_images, recon_images)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        advantages = torch.clamp(advantages, -5, 5)

        self._log_step(und_images, edited_images, recon_images, prompts_text, reverse_prompts_text, advantages)

        def loss_computation(prompts_text, und_pixel_values, und_image_grid_thw, ref_latents, prev_latents, pred_latents, ts, repeat_num=1):
            model_pred = self._compute_diffusion_loss(
                model, prompts_text, prev_latents, pred_latents, ts, "i2i_queries",
                pixel_values=und_pixel_values, image_grid_thw=und_image_grid_thw, 
                ref_latents=ref_latents, embedding_repeat_num=repeat_num
            )
            with torch.no_grad():
                ref_model_pred = self._compute_diffusion_loss(
                    self.ref_model, prompts_text, prev_latents, pred_latents, ts, "i2i_queries",
                    pixel_values=und_pixel_values, image_grid_thw=und_image_grid_thw,
                    ref_latents=ref_latents, embedding_repeat_num=repeat_num
                )

            prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(
                model_pred, self.scheduler, prev_latents, pred_latents, ts
            )
            _, _, ref_prev_sample_mean, ref_std_dev_t = compute_log_prob(
                ref_model_pred, self.scheduler, prev_latents, pred_latents, ts
            )

            assert (std_dev_t == ref_std_dev_t).all()
            kl = (prev_sample_mean - ref_prev_sample_mean) ** 2 / (2 * std_dev_t ** 2)
            kl = kl.mean(dim=tuple(range(1, kl.ndim)))

            return log_prob, kl

        log_probs_1, ref_loss_1 = loss_computation(
            prompts_text, und_pixel_values, und_image_grid_thw, noisy_ref_latents, prev_latents, pred_latents, ts, repeat_num=self.num_generations
        )
        log_probs_2, ref_loss_2 = loss_computation(
            [reverse_prompt_text for reverse_prompt_text in reverse_prompts_text for _ in range(self.num_generations)],
            und_edited_pixel_values, und_edited_image_grid_thw, reverse_noisy_ref_latents, reverse_prev_latents, reverse_pred_latents, reverse_ts, repeat_num=1
        )

        advantages = advantages.repeat_interleave(self.diffusion_generation_config["num_inference_steps"], dim=0)
        ratio_1 = torch.exp(log_probs_1 - log_probs)
        unclipped_loss_1 = -advantages * ratio_1
        clipped_loss_1 = -advantages * torch.clamp(ratio_1, 1.0 - 1e-4, 1.0 + 1e-4)
        policy_loss_1 = torch.mean(torch.maximum(unclipped_loss_1, clipped_loss_1))

        ratio_2 = torch.exp(log_probs_2 - reverse_log_probs)
        unclipped_loss_2 = -advantages * ratio_2
        clipped_loss_2 = -advantages * torch.clamp(ratio_2, 1.0 - 1e-4, 1.0 + 1e-4)
        policy_loss_2 = torch.mean(torch.maximum(unclipped_loss_2, clipped_loss_2))

        loss = (policy_loss_1 + policy_loss_2).mean() + self.beta * ref_loss_1.mean() + self.beta * ref_loss_2.mean()

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["ref_loss"].append(self.accelerator.gather_for_metrics(ref_loss_1 + ref_loss_2).mean().item())
        for i, (func_name, _, _) in enumerate(self.reward_funcs):
            self._metrics["reward/" + func_name].append(self.accelerator.gather_for_metrics(rewards_per_func[:, i]).mean().item())

        return loss
