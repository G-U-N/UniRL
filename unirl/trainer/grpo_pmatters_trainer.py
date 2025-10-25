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
GRPO (Group Relative Policy Optimization) Trainer implementations for multimodal models.

This module provides trainer classes for optimizing vision-language models using GRPO,
with support for language model optimization, diffusion model optimization, and joint optimization.
"""

import sys
import os
import io
import pickle
import requests
import torch
import torch.nn as nn
import torch.utils.data
import transformers
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datetime import datetime
from PIL import Image

from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoProcessor,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
    CLIPModel,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from accelerate.utils import DistributedType

from trl.data_utils import maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unimodel.qwenflux.qwenflux_inference import QwenFluxForInferenceLM
from unimodel.qwenflux.fluxpipeline import sde_step_with_logprob
from unimodel.qwenkontext.qwenkontext_inference import QwenKontextForInferenceLM
from unimodel.qwensd3.qwensd3_inference import QwenSD3ForInferenceLM
from unimodel.qwensana.qwensana_inference import QwenSanaForInferenceLM
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from qwen_vl_utils import process_vision_info


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb


# Type aliases
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


def compute_log_prob(
    model_pred: torch.Tensor,
    scheduler: FlowMatchEulerDiscreteScheduler,
    prev_latents: torch.Tensor,
    pred_latents: torch.Tensor,
    ts: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the log probability of predicted latents given previous latents.
    
    Args:
        model_pred: Model prediction output
        scheduler: Flow matching scheduler
        prev_latents: Previous latent states
        pred_latents: Predicted latent states
        ts: Timesteps
        
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


class BaseGRPOTrainer(Trainer):
    """
    Base trainer class for Group Relative Policy Optimization (GRPO).
    
    This base class provides common functionality for GRPO training including:
    - Model and reference model initialization
    - Generation configuration
    - Reward computation
    - Logging utilities
    
    Args:
        model: Model to be trained (string path or PreTrainedModel)
        reward_funcs: Reward functions for computing rewards
        args: Training configuration (GRPOConfig)
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        processing_class: Tokenizer/processor for data processing
        reward_processing_classes: Processing classes for reward models
        callbacks: Training callbacks
        optimizers: Tuple of (optimizer, scheduler)
        peft_config: PEFT configuration for parameter-efficient fine-tuning
        max_pixels: Maximum pixels for image processing
        min_pixels: Minimum pixels for image processing
        attn_implementation: Attention implementation type
    """
    
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, List[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, Dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, List[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):
        # Initialize configuration
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        
        # Load or validate model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        
        if isinstance(model, str):
            self.model_id = model
            model = self._load_model(model, model_init_kwargs, args)
        else:
            self.model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated."
                )
        
        # Apply PEFT if configured
        if peft_config is not None:
            model = get_peft_model(model, peft_config)
        
        # Configure model components
        self._configure_model_components(model)
        
        # Initialize reference model
        self.ref_model = self._create_reference_model(model, model_init_kwargs, args)
        
        # Initialize scheduler
        self.scheduler = self._initialize_scheduler(model)
        
        # Initialize processing class
        if processing_class is None:
            processing_class = self._create_default_processor(max_pixels, min_pixels)
        
        self.processing_class = processing_class
        self.reward_funcs = reward_funcs
        
        # Training parameters
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations

        self.beta = args.beta
        
        # Generation configuration
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=True,
            temperature=1.0,
            num_return_sequences=self.num_generations,
            pad_token_id=processing_class.pad_token_id,
            eos_token_id=processing_class.eos_token_id,
        )
        
        model.generation_config = self.generation_config
        if self.ref_model is not None:
            self.ref_model.generation_config = self.generation_config
        
        model.warnings_issued["estimate_tokens"] = True
        
        # Initialize metrics tracking
        self._metrics = defaultdict(list)
        
        # Data collator (no collation needed for GRPO)
        def data_collator(features):
            return features
        
        # Initialize parent Trainer
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
        
        # Configure loss computation
        self.model_accepts_loss_kwargs = False
        
        # Prepare reference model
        if self.ref_model is not None:
            if self.is_deepspeed_enabled and is_deepspeed_zero3_enabled():
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        
        # Diffusion generation configuration
        self.diffusion_generation_config = self._get_diffusion_config()
        
        # Logging setup
        self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.log_dir = os.path.join(args.output_dir, "training_samples", self.start_time)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def _load_model(self, model_id: str, model_init_kwargs: Dict, args: GRPOConfig) -> PreTrainedModel:
        """Load model based on model ID."""
        # Handle torch_dtype
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, str) and torch_dtype != "auto":
            model_init_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
        
        # Disable caching if gradient checkpointing is enabled
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
        
        # Load appropriate model type
        if "qwenflux" in model_id.lower():
            return QwenFluxForInferenceLM.from_pretrained(model_id, **model_init_kwargs)
        elif "qwenkontext" in model_id.lower():
            return QwenKontextForInferenceLM.from_pretrained(model_id, **model_init_kwargs)
        elif "qwensd3" in model_id.lower():
            return QwenSD3ForInferenceLM.from_pretrained(model_id, **model_init_kwargs)
        elif "qwensana" in model_id.lower():
            return QwenSanaForInferenceLM.from_pretrained(model_id, **model_init_kwargs)
        
        else:
            raise ValueError(f"Unsupported model type: {model_id}")
    
    def _configure_model_components(self, model: PreTrainedModel):
        """Configure model components and freeze/unfreeze parameters."""
        raise NotImplementedError("Subclasses must implement _configure_model_components")
    
    def _create_reference_model(
        self, model: PreTrainedModel, model_init_kwargs: Dict, args
    ) -> Optional[PreTrainedModel]:
        """Create and configure reference model."""
        if is_deepspeed_zero3_enabled():
            ref_model = self._load_model(self.model_id, model_init_kwargs, args)
        else:
            ref_model = create_reference_model(model)
        
        # Freeze all reference model parameters
        for param in ref_model.parameters():
            param.requires_grad = False
        
        return ref_model
    
    def _initialize_scheduler(self, model: PreTrainedModel) -> FlowMatchEulerDiscreteScheduler:
        """Initialize diffusion scheduler."""
        return FlowMatchEulerDiscreteScheduler(shift=3.0)
    
    def _create_default_processor(self, max_pixels: int, min_pixels: int) -> AutoProcessor:
        """Create default processor for Qwen models."""
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        processor.pad_token_id = processor.tokenizer.pad_token_id
        processor.eos_token_id = processor.tokenizer.eos_token_id
        processor.image_processor.max_pixels = max_pixels
        processor.image_processor.min_pixels = min_pixels
        return processor
    
    def _get_diffusion_config(self) -> Dict[str, Any]:
        """Get diffusion generation configuration."""
        device_id = int(str(self.accelerator.device).split(":")[-1]) if ":" in str(self.accelerator.device) else 0
        return {
            "guidance_scale": 3.5,
            "num_inference_steps": 10,
            "num_images_per_prompt": 1,
            "generator": torch.manual_seed(42 + device_id),
            "height": 512,
            "width": 512
        }
    
    def _set_signature_columns_if_needed(self):
        """Set signature columns for data processing."""
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]
    
    # def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
    #     """Prepare inputs without automatic tensor conversion."""
    #     return inputs
    
    def _log_step(
        self,
        images: List[Image.Image],
        prompts_text: List[str],
        advantages: torch.Tensor,
        completions: List[str]
    ):
        """Log training samples and metrics."""
        global_step = self.state.global_step
        
        if global_step % 5 != 0:
            return
        
        device_id = str(self.model.device).replace(":", "")
        # log_dir = os.path.join("training_samples", self.start_time)
        log_dir = self.log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create text log
        text_content = f"Prompt: {prompts_text[0]}"
        for idx in range(self.num_generations):
            text_content += f"\nCompletion {idx}: {completions[idx]}"
        
        txt_path = os.path.join(log_dir, f"step_{global_step}_{device_id}.txt")
        if not os.path.exists(txt_path):
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text_content)
        
        # Save images
        for idx in range(self.num_generations):
            advantage = advantages[idx]
            img_path = os.path.join(log_dir, f"step_{global_step}_{device_id}_{advantage.item():.4f}_{idx}.jpg")
            images[idx].save(img_path)
    
    def compute_rewards(
        self,
        images: List[Image.Image],
        captions: List[str],
        completions: List[str],
        meta_files: Optional[List[Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rewards from all reward functions.
        
        Args:
            images: Generated images
            captions: Original captions
            completions: Generated completions
            meta_files: Optional metadata files
            
        Returns:
            Tuple of (total_rewards, rewards_per_func)
        """
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(images), len(self.reward_funcs), device=device)
        
        for i, (func_name, reward_processing, reward_func) in enumerate(self.reward_funcs):
            if func_name == "recon":
                continue
            elif func_name == "format":
                rewards_per_func[:, i] = torch.tensor(reward_func(completions)).to(device)
            elif func_name == "jpeg_compressibility" or func_name == "jpeg_incompressibility":
                rewards_per_func[:, i] = reward_func(images)
            elif func_name in ["pickscore", "hps", "deqa", "unifiedreward_sglang", "ocr", "image_reward", "aesthetic"]:
                expanded_captions = [caption for caption in captions for _ in range(self.num_generations)]
                rewards_per_func[:, i] = torch.tensor(reward_func(images, expanded_captions)["scores"]).to(device)
            elif func_name == "gen_eval":
                meta_files_input = {"meta_datas": [mf for mf in meta_files for _ in range(self.num_generations)]}
                expanded_captions = [caption for caption in captions for _ in range(self.num_generations)]
                rewards_per_func[:, i] = torch.tensor(
                    reward_func(images, expanded_captions, meta_files_input)["scores"]
                ).to(device)
            else:
                raise ValueError(f"Unknown reward function: {func_name}")
        
        rewards = rewards_per_func.sum(dim=1)
        return rewards, rewards_per_func
    
    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized advantages from rewards.
        
        Args:
            rewards: Reward tensor
            
        Returns:
            Normalized advantages tensor
        """
        # Compute grouped-wise statistics
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        
        # Expand to match reward shape
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        
        # Normalize and clamp
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        advantages = torch.clamp(advantages, -5, 5)
        
        return advantages
    
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Log metrics with averaging."""
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        
        self._metrics.clear()


class PMattersGRPOTrainer(BaseGRPOTrainer):
    """
    GRPO Trainer for optimizing the language model component.
    
    This trainer focuses on optimizing the chain-of-thought (CoT) generation
    by fine-tuning the language model while keeping the diffusion model frozen.
    """
    
    def _configure_model_components(self, model: PreTrainedModel):
        """Configure model: train LLM, freeze diffusion components."""
        # Enable VAE slicing for memory efficiency
        try:
            model.get_model().diffusion_expert.enable_vae_slicing()
        except AttributeError:
            try:
                model.get_model().diffusion_expert.vae.enable_slicing()
            except AttributeError:
                pass
        
        # Train language model components
        for param in model.get_model().parameters():
            param.requires_grad = True
        for param in model.lm_head.parameters():
            param.requires_grad = True
        
        # Freeze vision and diffusion components
        for param in model.visual.parameters():
            param.requires_grad = False
        
        try:
            for param in model.get_model().transformer.parameters():
                param.requires_grad = False
            for param in model.get_model().vae.parameters():
                param.requires_grad = False
            for param in model.get_model().text_encoder.parameters():
                param.requires_grad = False
        except AttributeError:
            pass
        
        try:
            for param in model.get_model().text_encoder_2.parameters():
                param.requires_grad = False
            for param in model.get_model().text_encoder_3.parameters():
                param.requires_grad = False
        except AttributeError:
            pass
    
    def create_optimizer(self):
        """Create optimizer for LLM parameters only."""
        opt_kwargs = {
            "lr": 1e-6,
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "eps": self.args.adam_epsilon,
            "weight_decay": self.args.weight_decay,
        }
        
        # Collect LLM parameters (excluding diffusion components)
        llm_params = []
        for name, param in self.model.get_model().named_parameters():
            if param.requires_grad and all(
                comp not in name for comp in ["transformer", "text_encoder", "text_encoder_2", "text_encoder_3", "vae"]
            ):
                llm_params.append(param)
        
        return torch.optim.AdamW(llm_params, **opt_kwargs)
    
    def generate_images_with_sd3_api(
        self,
        prompts: List[str],
        endpoint: str,
        timeout: int = 120,
        num_images_per_prompt: int = 1
    ) -> List[Image.Image]:
        """
        Generate images using SD3 API endpoint.
        
        Args:
            prompts: Text prompts for image generation
            endpoint: API endpoint URL
            timeout: Request timeout in seconds
            num_images_per_prompt: Number of images per prompt
            
        Returns:
            List of generated PIL Images
        """
        payload = {
            "prompts": prompts,
            "width": 512,
            "height": 512,
            "num_inference_steps": 10,
            "guidance_scale": 3.5,
            "num_images": num_images_per_prompt
        }
        
        try:
            data = pickle.dumps(payload)
            response = requests.post(endpoint, data=data, timeout=timeout)
            response.raise_for_status()
            
            result = pickle.loads(response.content)
            if result["success"]:
                images = []
                for image_bytes in result["images"]:
                    image = Image.open(io.BytesIO(image_bytes))
                    images.append(image)
                return images
            else:
                raise ValueError(f"SD3 API returned error: {result}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"SD3 API request failed: {e}")
        except (pickle.UnpicklingError, KeyError) as e:
            raise RuntimeError(f"SD3 API response parsing failed: {e}")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute CoT loss with policy gradient."""
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        
        
        # print("model device", model.device)
        # print("accelerator device", self.accelerator.device)
        # assert 0
        # Extract inputs
        captions = [example["caption"] for example in inputs]
        meta_files = [example.get("metadata") for example in inputs if "metadata" in example]
        
        # Prepare prompts
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        
        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length:]
        
        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            with torch.no_grad():
                completion = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
                
                # Pad if necessary
                max_length = completion.size(1)
                padding = torch.full(
                    (completion.size(0), max_length - completion.size(1)),
                    self.processing_class.tokenizer.pad_token_id,
                    dtype=completion.dtype,
                    device=completion.device,
                )
                prompt_completion_ids = torch.cat([completion, padding], dim=1) if padding.size(1) > 0 else completion
        
        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        completions = self.processing_class.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        
        # Extract refined prompts and generate images
        refined_prompts = [self.model.extract_thinking_content(completion) for completion in completions]
        
        if isinstance(self.model, QwenSD3ForInferenceLM):
            images = self.generate_images_with_sd3_api(refined_prompts, endpoint="http://NODE_ADDR:18099/")
        else:
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                with torch.no_grad():
                    images = unwrapped_model.generate_image(texts=refined_prompts, diffusion_kwargs=self.diffusion_generation_config)
        
        # Compute rewards and advantages
        rewards, rewards_per_func = self.compute_rewards(images, captions, completions, meta_files)
        advantages = self.compute_advantages(rewards)
        
        # Log samples
        self._log_step(images, prompts_text, advantages, completions)
        
        # Compute CoT loss
        cot_loss, mean_kl, completion_mask = self._compute_cot_loss(
            model, prompt_completion_ids, completion_ids, prompt_length, advantages
        )
        
        # Log metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["cot_ref_loss"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        self._metrics["cot_loss"].append(self.accelerator.gather_for_metrics(cot_loss).mean().item())
        
        for i, (func_name, _, _) in enumerate(self.reward_funcs):
            self._metrics[f"reward/{func_name}"].append(
                self.accelerator.gather_for_metrics(rewards_per_func[:, i]).mean().item()
            )
        
        return cot_loss
    
    def _compute_cot_loss(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        prompt_length: int,
        advantages: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute chain-of-thought loss with KL regularization."""
        # Get per-token log probabilities
        def get_per_token_logps(model_instance, ids):
            logits = model_instance(ids).logits[:, :-1, :]
            ids = ids[:, 1:]
            per_token_logps = []
            for logits_row, ids_row in zip(logits, ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)
        
        per_token_logps = get_per_token_logps(model, input_ids)[:, prompt_length - 1:]
        
        with torch.inference_mode():
            ref_per_token_logps = get_per_token_logps(self.ref_model, input_ids)[:, prompt_length - 1:]
        
        # Compute KL divergence
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        
        # Create mask for valid tokens
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # Compute policy gradient loss
        advantages_cot = advantages.unsqueeze(1)
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages_cot
        per_token_loss = -(per_token_loss - 0.01 * per_token_kl)
        cot_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        return cot_loss, mean_kl, completion_mask


class PMattersGRPOJointTrainer(BaseGRPOTrainer):
    """
    GRPO Trainer for joint optimization of language and diffusion models.
    
    This trainer optimizes both the CoT generation (language model) and
    the image generation (diffusion model) simultaneously.
    """
    
    def _configure_model_components(self, model: PreTrainedModel):
        """Configure model: train both LLM and DiT components."""
        # Enable VAE slicing
        model.get_model().diffusion_expert.enable_vae_slicing()
        
        # Train all main components
        for param in model.get_model().parameters():
            param.requires_grad = True
        for param in model.lm_head.parameters():
            param.requires_grad = True
        for param in model.get_model().transformer.parameters():
            param.requires_grad = True
        
        # Freeze vision and encoding components
        for param in model.visual.parameters():
            param.requires_grad = False
        for param in model.get_model().vae.parameters():
            param.requires_grad = False
        for param in model.get_model().text_encoder.parameters():
            param.requires_grad = False
        for param in model.get_model().text_encoder_2.parameters():
            param.requires_grad = False
    
    def _initialize_scheduler(self, model: PreTrainedModel) -> FlowMatchEulerDiscreteScheduler:
        """Use the model's existing scheduler."""
        return model.get_model().diffusion_expert.scheduler
    
    def _get_diffusion_config(self) -> Dict[str, Any]:
        """Get diffusion configuration with more inference steps."""
        device_id = int(str(self.accelerator.device).split(":")[-1]) if ":" in str(self.accelerator.device) else 0
        return {
            "guidance_scale": 3.5,
            "num_inference_steps": 20,
            "num_images_per_prompt": 1,
            "generator": torch.manual_seed(42 + device_id),
            "height": 512,
            "width": 512
        }
    
    def create_optimizer(self):
        """Create optimizer with different learning rates for DiT and LLM."""
        opt_kwargs = {
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "eps": self.args.adam_epsilon,
            "weight_decay": self.args.weight_decay,
        }
        
        # Collect DiT parameters
        dit_params = []
        for name, param in self.model.get_model().transformer.named_parameters():
            if param.requires_grad:
                dit_params.append(param)
        
        # Collect LLM parameters (excluding diffusion components)
        llm_params = []
        for name, param in self.model.get_model().named_parameters():
            if param.requires_grad and all(
                comp not in name for comp in ["transformer", "text_encoder", "text_encoder_2", "vae"]
            ):
                llm_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = [
            {"params": dit_params, "lr": 1e-7},
            {"params": llm_params, "lr": 3e-7},
        ]
        
        return torch.optim.AdamW(param_groups, **opt_kwargs)
    
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step with joint optimization.
        
        This method generates samples, computes both CoT and diffusion losses,
        and updates the model parameters using gradient accumulation.
        """
        model.eval()
        self.ref_model.eval()
        
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        
        inputs = self._prepare_inputs(inputs)
        
        def loss_update(loss: torch.Tensor, scale_factor: float = 1.0):
            """Update loss with proper scaling and gradient accumulation."""
            kwargs = {}
            
            if self.args.n_gpu > 1:
                loss = loss.mean()
            
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps
            
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False
            
            loss = loss / scale_factor
            model.backward(loss)
        
        with self.compute_loss_context_manager():
            # Generate samples and compute rewards
            generations = self.generate_samples(model, inputs)
            torch.cuda.empty_cache()
            
            # Compute CoT loss
            cot_loss = self.cot_loss_computation(
                model,
                generations["prompt_completion_ids"],
                generations["completion_ids"],
                generations["prompt_length"],
                advantages=generations["advantages"]
            )
            loss_update(cot_loss, 1.0)
            
            # Compute diffusion loss in batches
            diff_advantages = generations["advantages"].repeat_interleave(
                self.diffusion_generation_config["num_inference_steps"], dim=0
            )
            total_len = diff_advantages.shape[0]
            diff_loss_lst = []
            diff_ref_loss_lst = []
            batch_size = 20
            
            for idx in range(0, total_len, batch_size):
                # Extract batch slice
                batched_states_slice = {}
                for key, value in generations["batched_states"].items():
                    if key in ["img_ids", "txt_ids"]:
                        batched_states_slice[key] = value
                    else:
                        batched_states_slice[key] = value[idx:idx+batch_size]
                
                # Compute diffusion loss for batch
                diff_loss, diff_ref_loss = self.diffusion_loss_computation(
                    generations["prev_latents"][idx:idx+batch_size],
                    generations["diff_sampling_log_probs"][idx:idx+batch_size],
                    generations["pred_latents"][idx:idx+batch_size],
                    generations["ts"][idx:idx+batch_size],
                    batched_states_slice,
                    diff_advantages[idx:idx+batch_size]
                )
                loss_update(diff_loss, float(total_len / batch_size))
                diff_loss_lst.append(diff_loss)
                diff_ref_loss_lst.append(diff_ref_loss)
        
        # Aggregate diffusion losses
        diff_loss = torch.stack(diff_loss_lst).mean()
        diff_ref_loss = torch.stack(diff_ref_loss_lst).mean()
        
        # Total loss
        loss = diff_loss + cot_loss
        
        del inputs
        
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            torch.cuda.empty_cache()
        
        model.step()
        
        # Log metrics
        self._metrics["diff_ref_loss"].append(self.accelerator.gather_for_metrics(diff_ref_loss).mean().item())
        self._metrics["diff_loss"].append(self.accelerator.gather_for_metrics(diff_loss).mean().item())
        
        torch.cuda.empty_cache()
        return loss.detach()
    
    def generate_samples(self, model: nn.Module, inputs: List[Dict]) -> Dict[str, Any]:
        """
        Generate samples including completions and images.
        
        Returns a dictionary containing all necessary tensors for loss computation.
        """
        captions = [example["caption"] for example in inputs]
        meta_files = [example.get("metadata") for example in inputs if "metadata" in example]
        
        # Prepare prompts
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        
        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length:]
        
        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            with torch.no_grad():
                completion = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
                
                # Pad completions
                max_length = completion.size(1)
                padding = torch.full(
                    (completion.size(0), max_length - completion.size(1)),
                    self.processing_class.tokenizer.pad_token_id,
                    dtype=completion.dtype,
                    device=completion.device,
                )
                prompt_completion_ids = torch.cat([completion, padding], dim=1) if padding.size(1) > 0 else completion
        
        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        completions = self.processing_class.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        
        # Extract refined prompts and generate images with SDE sampling
        refined_prompts = [self.model.extract_thinking_content(completion) for completion in completions]
        
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            with torch.no_grad():
                images, prev_latents, diff_sampling_log_probs, pred_latents, ts, batched_states = (
                    unwrapped_model.generate_image(
                        texts=refined_prompts,
                        diffusion_kwargs=self.diffusion_generation_config,
                        sde_sampling=True
                    )
                )
        
        # Compute rewards and advantages
        rewards, rewards_per_func = self.compute_rewards(images, captions, completions, meta_files)
        advantages = self.compute_advantages(rewards)
        
        # Log metrics
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        for i, (func_name, _, _) in enumerate(self.reward_funcs):
            self._metrics[f"reward/{func_name}"].append(
                self.accelerator.gather_for_metrics(rewards_per_func[:, i]).mean().item()
            )
        
        self._log_step(images, prompts_text, advantages, completions)
        
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
            "ts": ts
        }
    
    def cot_loss_computation(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        prompt_length: int,
        advantages: torch.Tensor
    ) -> torch.Tensor:
        """Compute chain-of-thought loss with KL regularization."""
        def get_per_token_logps(model_instance, ids):
            logits = model_instance(ids).logits[:, :-1, :]
            ids = ids[:, 1:]
            per_token_logps = []
            for logits_row, ids_row in zip(logits, ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)
        
        per_token_logps = get_per_token_logps(model, input_ids)[:, prompt_length - 1:]
        
        with torch.inference_mode():
            ref_per_token_logps = get_per_token_logps(self.ref_model, input_ids)[:, prompt_length - 1:]
        
        # Compute KL divergence
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        
        # Create completion mask
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # Compute policy gradient loss
        advantages_cot = advantages.unsqueeze(1)
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages_cot
        per_token_loss = -(per_token_loss - 0.01 * per_token_kl)
        cot_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        # Log metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        self._metrics["cot_ref_loss"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        self._metrics["cot_loss"].append(self.accelerator.gather_for_metrics(cot_loss).mean().item())
        
        return cot_loss
    
    def diffusion_loss_computation(
        self,
        prev_latents: torch.Tensor,
        diff_sampling_log_probs: torch.Tensor,
        pred_latents: torch.Tensor,
        ts: torch.Tensor,
        batched_states: Dict[str, torch.Tensor],
        advantages: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute diffusion model loss using PPO-style clipped objective.
        
        Args:
            prev_latents: Previous latent states
            diff_sampling_log_probs: Log probabilities from sampling
            pred_latents: Predicted latent states
            ts: Timesteps
            batched_states: Batched state information
            advantages: Advantage values
            
        Returns:
            Tuple of (diff_loss, diff_ref_loss)
        """
        # Forward pass through transformer
        model_pred = self.model.get_model().transformer(
            hidden_states=prev_latents.to(self.model.device),
            **batched_states,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]
        
        # Reference model forward pass
        with torch.no_grad():
            ref_model_pred = self.ref_model.get_model().transformer(
                hidden_states=prev_latents.to(self.model.device),
                **batched_states,
                joint_attention_kwargs={},
                return_dict=False,
            )[0]
        
        # Compute log probabilities
        _, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(
            model_pred, self.scheduler, prev_latents, pred_latents, ts
        )
        _, ref_log_prob, ref_prev_sample_mean, ref_std_dev_t = compute_log_prob(
            ref_model_pred, self.scheduler, prev_latents, pred_latents, ts
        )
        
        assert (std_dev_t == ref_std_dev_t).all()
        
        # Compute KL divergence
        kl = (prev_sample_mean - ref_prev_sample_mean)**2 / (2 * std_dev_t**2)
        kl = kl.mean(dim=tuple(range(1, kl.ndim)))
        
        # Compute PPO-style clipped loss
        ratio = torch.exp(log_prob - diff_sampling_log_probs)
        print(ratio)  # Debug output
        
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * torch.clamp(ratio, 1.0 - 1e-4, 1.0 + 1e-4)
        diff_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
        diff_loss = diff_loss.mean() + self.beta * kl.mean()
        
        return diff_loss, kl


class PMattersGRPODiffusionTrainer(BaseGRPOTrainer):
    """
    GRPO Trainer for optimizing only the diffusion model component.
    
    This trainer focuses on optimizing the diffusion transformer while
    keeping the language model frozen.
    """
    
    def _configure_model_components(self, model: PreTrainedModel):
        """Configure model: train DiT only, freeze LLM."""
        # Enable VAE slicing
        model.get_model().diffusion_expert.enable_vae_slicing()
        model.get_model().transformer.eval()
        
        # Freeze all components initially
        for param in model.get_model().parameters():
            param.requires_grad = False
        for param in model.visual.parameters():
            param.requires_grad = False
        for param in model.lm_head.parameters():
            param.requires_grad = False
        
        # Train only the diffusion transformer
        for param in model.get_model().transformer.parameters():
            param.requires_grad = True
        
        # Ensure encoding components are frozen
        for param in model.get_model().vae.parameters():
            param.requires_grad = False
        for param in model.get_model().text_encoder.parameters():
            param.requires_grad = False
        for param in model.get_model().text_encoder_2.parameters():
            param.requires_grad = False
    
    def _initialize_scheduler(self, model: PreTrainedModel) -> FlowMatchEulerDiscreteScheduler:
        """Use the model's existing scheduler."""
        return model.get_model().diffusion_expert.scheduler
    
    def _get_diffusion_config(self) -> Dict[str, Any]:
        """Get diffusion configuration with more inference steps."""
        device_id = int(str(self.accelerator.device).split(":")[-1]) if ":" in str(self.accelerator.device) else 0
        return {
            "guidance_scale": 3.5,
            "num_inference_steps": 20,
            "num_images_per_prompt": 1,
            "generator": torch.manual_seed(42 + device_id),
            "height": 512,
            "width": 512
        }
    
    def create_optimizer(self):
        """Create optimizer for DiT parameters only."""
        opt_kwargs = {
            "lr": 1e-5,
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "eps": self.args.adam_epsilon,
            "weight_decay": self.args.weight_decay,
        }
        
        # Collect DiT parameters
        dit_params = []
        for name, param in self.model.get_model().transformer.named_parameters():
            if param.requires_grad:
                dit_params.append(param)
        
        return torch.optim.AdamW(dit_params, **opt_kwargs)
    
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """Perform a training step optimizing only the diffusion model."""
        # model.eval()
        # self.ref_model.eval()
        
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        
        inputs = self._prepare_inputs(inputs)
        
        def loss_update(loss: torch.Tensor, scale_factor: float = 1.0):
            """Update loss with proper scaling."""
            kwargs = {}
            
            if self.args.n_gpu > 1:
                loss = loss.mean()
            
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps
            
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False
            
            loss = loss / scale_factor
            model.backward(loss)
        
        with self.compute_loss_context_manager():
            # Generate samples
            generations = self.generate_samples(model, inputs)
            # torch.cuda.empty_cache()
            
            # Compute diffusion loss in batches
            diff_advantages = generations["advantages"].repeat_interleave(
                self.diffusion_generation_config["num_inference_steps"], dim=0
            )
            total_len = diff_advantages.shape[0]
            diff_loss_lst = []
            diff_ref_loss_lst = []
            batch_size = 20
            
            for idx in range(0, total_len, batch_size):
                batched_states_slice = {}
                for key, value in generations["batched_states"].items():
                    if key in ["img_ids", "txt_ids"]:
                        batched_states_slice[key] = value
                    else:
                        batched_states_slice[key] = value[idx:idx+batch_size]
                
                diff_loss, diff_ref_loss = self.diffusion_loss_computation(
                    generations["prev_latents"][idx:idx+batch_size],
                    generations["diff_sampling_log_probs"][idx:idx+batch_size],
                    generations["pred_latents"][idx:idx+batch_size],
                    generations["ts"][idx:idx+batch_size],
                    batched_states_slice,
                    diff_advantages[idx:idx+batch_size]
                )
                loss_update(diff_loss, float(total_len / batch_size))
                diff_loss_lst.append(diff_loss)
                diff_ref_loss_lst.append(diff_ref_loss)
        
        diff_loss = torch.stack(diff_loss_lst).mean()
        diff_ref_loss = torch.stack(diff_ref_loss_lst).mean()
        
        loss = diff_loss
        
        del inputs
        
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            torch.cuda.empty_cache()
        
        model.step()
        
        self._metrics["diff_ref_loss"].append(self.accelerator.gather_for_metrics(diff_ref_loss).mean().item())
        self._metrics["diff_loss"].append(self.accelerator.gather_for_metrics(diff_loss).mean().item())
        
        torch.cuda.empty_cache()
        return loss.detach()
    
    def generate_samples(self, model: nn.Module, inputs: List[Dict]) -> Dict[str, Any]:
        """Generate samples using pre-defined captions (no CoT generation)."""
        captions = [example["caption"] for example in inputs]
        meta_files = [example.get("metadata") for example in inputs if "metadata" in example]
        
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        
        # Use captions directly as refined prompts
        completions = [caption for caption in captions for _ in range(self.num_generations)]
        refined_prompts = completions
        
        # Generate images with SDE sampling
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            with torch.no_grad():
                images, prev_latents, diff_sampling_log_probs, pred_latents, ts, batched_states = (
                    unwrapped_model.generate_image(
                        texts=refined_prompts,
                        diffusion_kwargs=self.diffusion_generation_config,
                        sde_sampling=True
                    )
                )
        
        # Compute rewards and advantages
        rewards, rewards_per_func = self.compute_rewards(images, captions, completions, meta_files)
        advantages = self.compute_advantages(rewards)
        
        # Log metrics
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        for i, (func_name, _, _) in enumerate(self.reward_funcs):
            self._metrics[f"reward/{func_name}"].append(
                self.accelerator.gather_for_metrics(rewards_per_func[:, i]).mean().item()
            )
        
        self._log_step(images, prompts_text, advantages, completions)
        
        return {
            "images": images,
            "prev_latents": prev_latents,
            "diff_sampling_log_probs": diff_sampling_log_probs,
            "pred_latents": pred_latents,
            "batched_states": batched_states,
            "advantages": advantages,
            "ts": ts
        }
    
    def diffusion_loss_computation(
        self,
        prev_latents: torch.Tensor,
        diff_sampling_log_probs: torch.Tensor,
        pred_latents: torch.Tensor,
        ts: torch.Tensor,
        batched_states: Dict[str, torch.Tensor],
        advantages: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute diffusion model loss using PPO-style clipped objective."""
        # Forward pass
        # with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
        #     model_pred = unwrapped_model.get_model().transformer(
        #         hidden_states=prev_latents.to(self.model.device),
        #         **batched_states,
        #         joint_attention_kwargs={},
        #         return_dict=False,
        #     )[0]
        model_pred = self.model.get_model().transformer(
            hidden_states=prev_latents.to(self.model.device),
            **batched_states,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]
            
        with torch.no_grad():
            ref_model_pred = self.ref_model.get_model().transformer(
                hidden_states=prev_latents.to(self.model.device),
                **batched_states,
                joint_attention_kwargs={},
                return_dict=False,
            )[0]
        
        # Compute log probabilities and KL
        _, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(
            model_pred, self.scheduler, prev_latents, pred_latents, ts
        )
        _, ref_log_prob, ref_prev_sample_mean, ref_std_dev_t = compute_log_prob(
            ref_model_pred, self.scheduler, prev_latents, pred_latents, ts
        )
        
        assert (std_dev_t == ref_std_dev_t).all()
        
        kl = (prev_sample_mean - ref_prev_sample_mean)**2 / (2 * std_dev_t**2)
        kl = kl.mean(dim=tuple(range(1, kl.ndim)))
        
        # PPO-style clipped loss
        ratio = torch.exp(log_prob - diff_sampling_log_probs)
        print(ratio) # need to check
        
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * torch.clamp(ratio, 1.0 - 1e-4, 1.0 + 1e-4)
        diff_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
        diff_loss = diff_loss.mean() + self.beta * kl.mean()
        
        return diff_loss, kl
    
    

class QwenKontextGRPOTrainer(BaseGRPOTrainer):
    """
    GRPO Trainer for optimizing the language model component.
    
    This trainer focuses on optimizing the chain-of-thought (CoT) generation
    by fine-tuning the language model while keeping the diffusion model frozen.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)    
        
        if any(name == "clip_sim" for name, _, _ in self.reward_funcs) or any(name == "sim_direction" for name, _, _ in self.reward_funcs):
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.accelerator.device)
            if self.is_deepspeed_enabled:
                self.clip_model = prepare_deepspeed(self.clip_model, self.accelerator)
            else:
                self.clip_model = self.accelerator.prepare_model(self.clip_model, evaluation_mode=True)
    
    def _configure_model_components(self, model: PreTrainedModel):
        """Configure model: train LLM, freeze diffusion components."""
        # Enable VAE slicing for memory efficiency
        try:
            model.get_model().diffusion_expert.enable_vae_slicing()
        except AttributeError:
            try:
                model.get_model().diffusion_expert.vae.enable_slicing()
            except AttributeError:
                pass
        
        # Train language model components
        for param in model.get_model().parameters():
            param.requires_grad = True
        for param in model.lm_head.parameters():
            param.requires_grad = True
        
        # Freeze vision and diffusion components
        for param in model.visual.parameters():
            param.requires_grad = False
        
        try:
            for param in model.get_model().transformer.parameters():
                param.requires_grad = False
            for param in model.get_model().vae.parameters():
                param.requires_grad = False
            for param in model.get_model().text_encoder.parameters():
                param.requires_grad = False
        except AttributeError:
            pass
        
        try:
            for param in model.get_model().text_encoder_2.parameters():
                param.requires_grad = False
            for param in model.get_model().text_encoder_3.parameters():
                param.requires_grad = False
        except AttributeError:
            pass
    def compute_rewards(
        self,
        inputs: List[Dict],
        images: List[Image.Image],
        completions: List[str],
        recon_images: Optional[List[Image.Image]] = None,
        
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rewards from all reward functions.
        
        Args:
            images: Generated images
            captions: Original captions
            completions: Generated completions
            meta_files: Optional metadata files
            
        Returns:
            Tuple of (total_rewards, rewards_per_func)
        """
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(images), len(self.reward_funcs), device=device)
        
        for i, (func_name, reward_processing, reward_func) in enumerate(self.reward_funcs):
            if func_name == "recon":
                continue
            elif func_name == "format":
                rewards_per_func[:, i] = torch.tensor(reward_func(completions)).to(device)
            elif func_name == "jpeg_compressibility" or func_name == "jpeg_incompressibility":
                rewards_per_func[:, i] = reward_func(images)
            elif func_name in ["pickscore", "hps", "deqa", "unifiedreward_sglang", "ocr", "image_reward", "aesthetic"]:
                expanded_captions = [caption for caption in inputs["caption"] for _ in range(self.num_generations)]
                rewards_per_func[:, i] = torch.tensor(reward_func(images, expanded_captions)["scores"]).to(device)
            elif func_name == "gen_eval":
                meta_files = [ex.get("metadata") for ex in inputs]
                meta_files_input = {"meta_datas": [m for m in meta_files for _ in range(self.num_generations)]}
                expanded_captions = [caption for caption in inputs["caption"] for _ in range(self.num_generations)]
                rewards_per_func[:, i] = torch.tensor(
                    reward_func(images, expanded_captions, meta_files_input)["scores"]
                ).to(device)
            elif func_name == "clip_sim":
                rewards_per_func[:, i] = reward_func(
                    recon_images, [example["image"] for example in inputs for _ in range(self.num_generations)],
                    self.clip_model, reward_processing
                ).to(device)
            elif func_name == "sim_direction":
                rewards_per_func[:, i] = reward_func(
                    [example["image"] for example in inputs for _ in range(self.num_generations)],
                    images,
                    [example["caption"] for example in inputs for _ in range(self.num_generations)],
                    [example["target_caption"] for example in inputs for _ in range(self.num_generations)],
                    self.clip_model,
                    reward_processing
                ).to(device)
            elif func_name == "editreward":
                images_input = dict(source=[example["image"] for example in inputs for _ in range(self.num_generations)], edited=images)
                rewards_per_func[:, i] = torch.tensor(reward_func(
                    images_input,
                    [example["editing_instruction"] for example in inputs for _ in range(self.num_generations)],
                )["scores"]).to(device)
            else:
                raise ValueError(f"Unknown reward function: {func_name}")
        
        rewards = rewards_per_func.sum(dim=1)
        return rewards, rewards_per_func
    def create_optimizer(self):
        """Create optimizer for LLM parameters only."""
        opt_kwargs = {
            "lr": 1e-6,
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "eps": self.args.adam_epsilon,
            "weight_decay": self.args.weight_decay,
        }
        
        # Collect LLM parameters (excluding diffusion components)
        llm_params = []
        for name, param in self.model.get_model().named_parameters():
            if param.requires_grad and all(
                comp not in name for comp in ["transformer", "text_encoder", "text_encoder_2", "text_encoder_3", "vae"]
            ):
                llm_params.append(param)
        
        return torch.optim.AdamW(llm_params, **opt_kwargs)
    
    def _get_diffusion_config(self) -> Dict[str, Any]:
        """Get diffusion generation configuration."""
        device_id = int(str(self.accelerator.device).split(":")[-1]) if ":" in str(self.accelerator.device) else 0
        return {
            "guidance_scale": 2.5,
            "num_inference_steps": 10,
            "num_images_per_prompt": 1,
            "generator": torch.manual_seed(42 + device_id),
            "height": 512,
            "width": 512
        }
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute CoT loss with policy gradient."""
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        
        ref_images = [x["image"] for x in inputs]
        
        # Prepare prompts
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            images = ref_images,
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        
        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length:]
        
        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            with torch.no_grad():
                completion = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
                
                # Pad if necessary
                max_length = completion.size(1)
                padding = torch.full(
                    (completion.size(0), max_length - completion.size(1)),
                    self.processing_class.tokenizer.pad_token_id,
                    dtype=completion.dtype,
                    device=completion.device,
                )
                prompt_completion_ids = torch.cat([completion, padding], dim=1) if padding.size(1) > 0 else completion
        
        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        completions = self.processing_class.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        
        # Extract refined prompts and generate images
        refined_prompts = [self.model.extract_thinking_content(completion) for completion in completions]
        

        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            with torch.no_grad():
                images = unwrapped_model.generate_image(images = ref_images, texts=refined_prompts, diffusion_kwargs=self.diffusion_generation_config)
        
        
        # Compute rewards and advantages
        rewards, rewards_per_func = self.compute_rewards(inputs, images, completions)
        advantages = self.compute_advantages(rewards)
        
        # Log samples
        self._log_step(ref_images, images, refined_prompts, advantages)
        
        # Compute CoT loss
        cot_loss, mean_kl, completion_mask = self._compute_cot_loss(
            model, prompt_completion_ids, completion_ids, prompt_length, advantages
        )
        
        # Log metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["cot_ref_loss"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        self._metrics["cot_loss"].append(self.accelerator.gather_for_metrics(cot_loss).mean().item())
        
        for i, (func_name, _, _) in enumerate(self.reward_funcs):
            self._metrics[f"reward/{func_name}"].append(
                self.accelerator.gather_for_metrics(rewards_per_func[:, i]).mean().item()
            )
        
        return cot_loss
    
    def _log_step(self, und_images, edited_images, prompts_text, advantages):
        global_step = self.state.global_step
        if global_step % 10 != 0:
            return

        device_id = str(self.model.device).replace(":", "")
        log_dir = self.log_dir

        text_contet = ""
        for idx in range(self.num_generations):
            text_content += f"Prompt: {prompts_text[idx]}\n"
        if os.path.exists(os.path.join(log_dir, f"step_{global_step}_{device_id}.txt")):
            return 
        with open(os.path.join(log_dir, f"step_{global_step}_{device_id}.txt"), "w", encoding="utf-8") as f:
            f.write(text_content)

        for idx in range(self.num_generations):
            orig_img = und_images[0]
            rev_img = edited_images[idx]
            orig_img_pil = orig_img
            rev_img_pil = rev_img
            orig_img_pil.save(os.path.join(log_dir, f"step_{global_step}_{device_id}_orig_{idx}.jpg"))
            rev_img_pil.save(os.path.join(log_dir, f"step_{global_step}_{device_id}_edited_{idx}_{advantages[idx].item():.5f}.jpg"))


    def _compute_cot_loss(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        prompt_length: int,
        advantages: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute chain-of-thought loss with KL regularization."""
        # Get per-token log probabilities
        def get_per_token_logps(model_instance, ids):
            logits = model_instance(ids).logits[:, :-1, :]
            ids = ids[:, 1:]
            per_token_logps = []
            for logits_row, ids_row in zip(logits, ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)
        
        per_token_logps = get_per_token_logps(model, input_ids)[:, prompt_length - 1:]
        
        with torch.inference_mode():
            ref_per_token_logps = get_per_token_logps(self.ref_model, input_ids)[:, prompt_length - 1:]
        
        # Compute KL divergence
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        
        # Create mask for valid tokens
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # Compute policy gradient loss
        advantages_cot = advantages.unsqueeze(1)
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages_cot
        per_token_loss = -(per_token_loss - 0.01 * per_token_kl)
        cot_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        return cot_loss, mean_kl, completion_mask
    

class QwenKontextCycleGRPOTrainer(QwenKontextGRPOTrainer):
    """
    GRPO Trainer for optimizing the language model component.
    
    This trainer focuses on optimizing the chain-of-thought (CoT) generation
    by fine-tuning the language model while keeping the diffusion model frozen.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.reverse_diffusion_generation_config = self.diffusion_generation_config
    
        
        if any(name == "clip_sim" for name, _, _ in self.reward_funcs) or any(name == "sim_direction" for name, _, _ in self.reward_funcs):
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.accelerator.device)
            if self.is_deepspeed_enabled:
                self.clip_model = prepare_deepspeed(self.clip_model, self.accelerator)
            else:
                self.clip_model = self.accelerator.prepare_model(self.clip_model, evaluation_mode=True)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute CoT loss with policy gradient."""
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        
        # Extract inputs
        captions = [example["caption"] for example in inputs]
        meta_files = [example.get("metadata") for example in inputs if "metadata" in example]
        ref_images = [x["image"] for x in inputs]
        
        # hacked; only thinking for the first round 
        reverse_prompts = [x["reverse_editing_instruction"] for x in inputs]

        
        # Prepare prompts
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            images = ref_images,
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        
        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length:]
        
        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            with torch.no_grad():
                completion = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
                
                # Pad if necessary
                max_length = completion.size(1)
                padding = torch.full(
                    (completion.size(0), max_length - completion.size(1)),
                    self.processing_class.tokenizer.pad_token_id,
                    dtype=completion.dtype,
                    device=completion.device,
                )
                prompt_completion_ids = torch.cat([completion, padding], dim=1) if padding.size(1) > 0 else completion
        
        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        completions = self.processing_class.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        
        # Extract refined prompts and generate images
        refined_prompts = [self.model.extract_thinking_content(completion) for completion in completions]
        

        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            with torch.no_grad():
                images = unwrapped_model.generate_image(images = ref_images, texts=refined_prompts, diffusion_kwargs=self.diffusion_generation_config)
        
        
        
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            with torch.no_grad():
                recon_images = unwrapped_model.generate_image(images = images, texts=[reverse_prompt for reverse_prompt in reverse_prompts for _ in range(self.num_generations)], diffusion_kwargs=self.diffusion_generation_config)
        
        
        
        # Compute rewards and advantages
        rewards, rewards_per_func = self.compute_rewards(inputs, images, completions, recon_images)
        advantages = self.compute_advantages(rewards)
        
        # Log samples
        self._log_step(ref_images, images, recon_images,  refined_prompts, reverse_prompts, advantages)
        
        # Compute CoT loss
        cot_loss, mean_kl, completion_mask = self._compute_cot_loss(
            model, prompt_completion_ids, completion_ids, prompt_length, advantages
        )
        
        # Log metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["cot_ref_loss"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        self._metrics["cot_loss"].append(self.accelerator.gather_for_metrics(cot_loss).mean().item())
        
        for i, (func_name, _, _) in enumerate(self.reward_funcs):
            self._metrics[f"reward/{func_name}"].append(
                self.accelerator.gather_for_metrics(rewards_per_func[:, i]).mean().item()
            )
        
        return cot_loss
    
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
            recon_img_pil.save(os.path.join(log_dir, f"step_{global_step}_{device_id}_recon_{idx}_{advantages[idx].item():.5f}.jpg"))
