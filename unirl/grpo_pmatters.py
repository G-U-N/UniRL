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

import os
import re
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import io
import numpy as np
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, CLIPModel, CLIPProcessor, CLIPImageProcessor
from unimodel.blip3o.constants import UND_IMAGE_TOKEN
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from .trainer import PMattersGRPOTrainer, PMattersGRPOJointTrainer, PMattersGRPODiffusionTrainer, QwenKontextGRPOTrainer
from .reward_evaluator.reward_evaluator import RewardEvaluatorClient


reward_client = RewardEvaluatorClient()

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'recon', 'jpeg_compressibility', 
            'jpeg_incompressibility', 'pickscore', 'deqa', 'gen_eval', 'unifiedreward_sglang', 'ocr', 
            'image_reward', 'aesthetic', 'hps', 'clip_sim', 'sim_direction'.
        max_pixels (`int`, optional):
            Maximum number of pixels for the image.
        min_pixels (`int`, optional):
            Minimum number of pixels for the image.
        prompts_file (`str`, optional):
            Path to the file containing prompts (.txt, .jsonl, or .parquet).
        task (`str`, optional):
            Task type: 't2i' for text-to-image, 't2icot' for text-to-image with CoT, 'i2i' for image-to-image.
    """
    reward_funcs: list[str] = field(
        default_factory=lambda: ["jpeg_compressibility"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format', 'recon', 'jpeg_compressibility', 'jpeg_incompressibility', 'pickscore', 'deqa', 'gen_eval', 'unifiedreward_sglang', 'ocr', 'image_reward', 'aesthetic', 'hps', 'clip_sim', 'sim_direction'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    prompts_file: Optional[str] = field(
        default="prompts.txt",
        metadata={"help": "Path to the .txt, .jsonl, or .parquet file containing prompts"},
    )
    task: Optional[str] = field(
        default="t2i",
        metadata={"help": "Task type: 't2i' for text-to-image, 't2icot' for text-to-image with CoT, 'i2i' for image-to-image"}
    )

def recon_reward(orig_images, recon_images):
    """
    Compute rewards based on PSNR between original and reconstructed tensor images.
    Images are assumed to have values normalized between 0 and 1, with maximum value exactly 1.0.
    
    Args:
        orig_images (torch.Tensor): Batch of original images [B, C, H, W], values in [0, 1].
        recon_images (torch.Tensor): Batch of reconstructed images [B, C, H, W], values in [0, 1].
    
    Returns:
        list: List of PSNR values as rewards for each image pair.
    """
    def calculate_psnr(img1, img2):
        if img1.shape != img2.shape:
            raise ValueError("Image dimensions must match")
        
        mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
        perfect_match = mse == 0
        psnr = torch.zeros_like(mse)
        valid = ~perfect_match
        psnr[valid] = -10 * torch.log10(mse[valid])
        psnr[perfect_match] = float('inf')
        return psnr
    
    orig_images = orig_images.float().clamp(0, 1)
    recon_images = recon_images.float().clamp(0, 1)
    rewards = calculate_psnr(orig_images, recon_images)
    return rewards

def jpeg_incompressibility(images):
    buffers = [io.BytesIO() for _ in images]
    for image, buffer in zip(images, buffers):
        image.save(buffer, format="JPEG", quality=95)
    sizes = [buffer.tell() / 1000 for buffer in buffers]
    return torch.tensor(sizes)

def jpeg_compressibility(images):
    return jpeg_incompressibility(images) * -1.0 / 500

def sim_direction(orig_images, edited_images, original_caption, edited_caption, clip_model, clip_processor):
    """
    Compute sim_direction using HuggingFace CLIP by calculating cosine similarity 
    between the difference of image features and the difference of text features.
    
    Args:
        orig_images: Original image or list of images (PIL.Image)
        edited_images: Edited image or list of images (PIL.Image)
        original_caption: List of original text prompts
        edited_caption: List of edited text prompts
        clip_model: Preloaded HuggingFace CLIP model
        clip_processor: Preloaded HuggingFace CLIP processor
    
    Returns:
        torch.Tensor: Cosine similarity of image and text feature differences (sim_direction)
    """
    device = clip_model.device
    inputs_orig = clip_processor(images=orig_images, text=original_caption, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
    inputs_edited = clip_processor(images=edited_images, text=edited_caption, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
    inputs_orig = {k: v.to(device) for k, v in inputs_orig.items()}
    inputs_edited = {k: v.to(device) for k, v in inputs_edited.items()}
    
    with torch.no_grad():
        image_features_orig = clip_model(**inputs_orig)
        image_features_edited = clip_model(**inputs_edited)
    
    image_features_orig, text_features_orig = image_features_orig.image_embeds, image_features_orig.text_embeds
    image_features_edited, text_features_edited = image_features_edited.image_embeds, image_features_edited.text_embeds
    
    image_features_orig = image_features_orig / image_features_orig.norm(dim=-1, keepdim=True)
    image_features_edited = image_features_edited / image_features_edited.norm(dim=-1, keepdim=True)
    text_features_orig = text_features_orig / text_features_orig.norm(dim=-1, keepdim=True)
    text_features_edited = text_features_edited / text_features_edited.norm(dim=-1, keepdim=True)
    
    sim_direction = F.cosine_similarity(
        image_features_edited - image_features_orig, 
        text_features_edited - text_features_orig
    )
    return sim_direction

def clip_similarity(orig_images, recon_images, clip_model, clip_processor):
    """
    Compute cosine similarity between images and reference images using CLIP embeddings.
    
    Args:
        orig_images (list[PIL.Image]): List of original PIL images.
        recon_images (list[PIL.Image]): List of reconstructed PIL images.
        clip_model (CLIPModel): Preloaded CLIP model for computing embeddings.
        clip_processor (CLIPProcessor): Preloaded CLIP processor for image preprocessing.
    
    Returns:
        torch.Tensor: Cosine similarity scores for each image pair.
    """
    ref_inputs = clip_processor(images=orig_images, return_tensors="pt")
    recon_inputs = clip_processor(images=recon_images, return_tensors="pt")
    ref_inputs = {k: v.to(clip_model.device) for k, v in ref_inputs.items()}
    recon_inputs = {k: v.to(clip_model.device) for k, v in recon_inputs.items()}
    
    with torch.no_grad():
        ref_embeddings = clip_model.get_image_features(**ref_inputs)
        recon_embeddings = clip_model.get_image_features(**recon_inputs)
    
    similarity = F.cosine_similarity(ref_embeddings, recon_embeddings, dim=-1)
    return similarity

def pickscore(images, prompts):
    return reward_client.evaluate("pickscore", images, prompts)

def deqa(images, prompts):
    return reward_client.evaluate("deqa", images, prompts)

def gen_eval(images, prompts, meta_files):
    return reward_client.evaluate("gen_eval", images, prompts, meta_files)

def unifiedreward_sglang(images, prompts):
    return reward_client.evaluate("unifiedreward_sglang", images, prompts)

def ocr(images, prompts):
    return reward_client.evaluate("ocr", images, prompts)

def image_reward(images, prompts):
    return reward_client.evaluate("image_reward", images, prompts)

def aesthetic(images, prompts):
    return reward_client.evaluate("aesthetic", images, prompts)

def hps(images, prompts):
    return reward_client.evaluate("hps", images, prompts)

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    
    pattern = r'<answer>.*?</answer>'
    matches = list(map(lambda x: re.match(pattern, x), completions))
    return [1.0 if match else 0.0 for match in matches]

def editreward(images, prompts):
    return reward_client.evaluate("editreward", images, prompts)

reward_funcs_registry = {
    "recon": recon_reward,
    "jpeg_compressibility": jpeg_compressibility,
    "jpeg_incompressibility": jpeg_incompressibility,
    "pickscore": pickscore,
    "deqa": deqa,
    "gen_eval": gen_eval,
    "unifiedreward_sglang": unifiedreward_sglang,
    "ocr": ocr,
    "image_reward": image_reward,
    "aesthetic": aesthetic,
    "hps": hps,
    "clip_sim": clip_similarity,
    "sim_direction": sim_direction,
    "format": format_reward,
    "editreward": editreward,
}

reward_processing_registry = {
    "recon": T.Compose([T.ToTensor()]),
    "jpeg_compressibility": None,
    "jpeg_incompressibility": None,
    "pickscore": None,
    "deqa": None,
    "gen_eval": None,
    "unifiedreward_sglang": None,
    "ocr": None,
    "image_reward": None,
    "aesthetic": None,
    "hps": None,
    "clip_sim": CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14"),
    "sim_direction": CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14"),
    "format": None,
    "editreward": None,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

class PromptDataset(Dataset):
    def __init__(self, prompts_file, question_template):
        self.question_template = question_template
        self.prompts = []
        self.transform = T.Compose([
        T.Resize(512, interpolation=T.InterpolationMode.BILINEAR),  
        T.CenterCrop((512, 512))  
    ])
        
        if prompts_file.endswith(".txt"):
            if not os.path.exists(prompts_file):
                raise FileNotFoundError(f"Prompts file {prompts_file} not found")
            with open(prompts_file, 'r', encoding='utf-8') as f:
                for line in f:
                    prompt = line.strip()
                    if prompt:
                        formatted_prompt = self.question_template.format(Question=prompt)
                        self.prompts.append({
                            "prompt": [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": formatted_prompt}
                                    ]
                                }
                            ],
                            "caption": prompt
                        })
        
        elif prompts_file.endswith(".jsonl"):
            with open(prompts_file, 'r', encoding='utf-8') as f:
                metadatas = [json.loads(line) for line in f]
                captions = [item['prompt'] for item in metadatas]
                self.prompts = [
                    {
                        "prompt": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": self.question_template.format(Question=caption)}
                                ]
                            }
                        ],
                        "metadata": metadata,
                        "caption": caption
                    } for caption, metadata in zip(captions, metadatas)
                ]
        
        elif prompts_file.endswith(".parquet"):
            dataset = load_dataset("parquet", data_files={"train": prompts_file})["train"]
            for item in dataset:
                prompt = item.get("prompt", "")
                reverse_prompt = item.get("reverse_prompt", "")
                image = item.get("image", None)
                if not prompt or not image:
                    continue
                formatted_prompt = self.question_template.format(Question=prompt)
                formatted_reverse_prompt = self.question_template.format(Question=reverse_prompt)
                self.prompts.append({
                    "image": self.transform(item.get("image", None)),
                    "caption": item.get("caption", ""),
                    "target_caption": item.get("target_caption", ""),
                    "editing_instruction": item.get("prompt", ""),
                    "reverse_editing_instruction": item.get("reverse_prompt", ""),
                    "prompt": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": formatted_prompt}
                            ]
                        }
                    ],
                    "reverse_prompt": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": formatted_reverse_prompt}
                            ]
                        }
                    ]
                })
        else:
            raise ValueError("Unsupported file format. Use .txt, .jsonl, or .parquet")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

def get_last_checkpoint(output_dir):
    from transformers.trainer_utils import get_last_checkpoint
    if os.path.isdir(output_dir):
        return get_last_checkpoint(output_dir)
    return None

def main(script_args, training_args, model_args):
    reward_funcs = [(func, reward_processing_registry[func], reward_funcs_registry[func]) for func in script_args.reward_funcs]

    if script_args.task == "t2i-text" or script_args.task == "t2i-joint" or script_args.task == "t2i-diff":
        QUESTION_TEMPLATE = """Please provide an enhanced prompt for the following image generation prompt to make the image more realistic, detailed, with clear separation and precise alignment of all entities.
            Original prompt: {Question}.  Directly provide the improved prompt in <answer> </answer> tags."""
    elif script_args.task == "i2i-text":
        QUESTION_TEMPLATE = """Please provide an enhanced prompt for the following image editing prompt. 
            Ensure the revised prompt is clear, specific, and includes detailed instructions to achieve the desired outcome while maintaining the original intent. 
            Original prompt: {Question}. Directly provide the improved prompt in <answer> </answer> tags."""""
    else: 
        assert 0, "Unsupported task. Choose from 't2i-text', 't2i-diff', 't2i-joint', or 'i2i'."

    train_dataset = PromptDataset(
        prompts_file=script_args.prompts_file,
        question_template=QUESTION_TEMPLATE
    )

    trainer_cls = {
        "t2i-text":  PMattersGRPOTrainer,
        "t2i-diff":  PMattersGRPODiffusionTrainer,
        "t2i-joint": PMattersGRPOJointTrainer,
        "i2i-text":  QwenKontextGRPOTrainer, 
    }.get(script_args.task, PMattersGRPOTrainer)

    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    trainer.train(resume_from_checkpoint=get_last_checkpoint(training_args.output_dir))

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)