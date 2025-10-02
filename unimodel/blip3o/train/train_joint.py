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
import io
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List, Tuple
import time
import torch, gc
import glob
import transformers
import tokenizers
import random
from ..constants import IGNORE_INDEX, T2I_TOKEN_IDX, I2I_TOKEN_IDX, DEFAULT_IMAGE_TOKEN, UND_IMAGE_TOKEN

from torch.utils.data import Dataset
from .blip3o_trainer import blip3oTrainer
from ...blip3o import conversation as  conversation_lib
from ..model import * # Imports blip3oQwenForCausalLM and related models
from PIL import Image, ImageFile
from datasets import load_dataset, concatenate_datasets
from pathlib import Path
from datasets.utils.logging import set_verbosity_info
from transformers import logging as tf_logging
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoProcessor
from packaging import version

# Set global configurations
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Image preprocessing for underspecified (contextual) images
transform_und_images = T.Compose([
    T.Resize(672, interpolation=InterpolationMode.BICUBIC, antialias=True), 
    T.CenterCrop(672) # Target size for a fixed number of tokens
]) 

set_verbosity_info()
tf_logging.set_verbosity_info()

local_rank = None

# Check tokenizer version for compatibility
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")


def rank0_print(*args):
    """Prints only on the main process (rank 0)."""
    if local_rank == 0:
        print(*args)

# --------------------------------------------------------------------------
# --- Data Classes for Arguments ---
# --------------------------------------------------------------------------

@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/resources we are going to load."""
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0", metadata={"help": "Conversation template version."})
    freeze_backbone: bool = field(default=True, metadata={"help": "Whether to freeze the LLM and base vision tower."})
    vision_tower: Optional[str] = field(default=None, metadata={"help": "Path/name of the base vision tower."})
    gen_vision_tower: Optional[str] = field(default=None, metadata={"help": "Path/name of the generative VAE/vision tower."})
    vision_tower_pretrained: Optional[str] = field(default=None, metadata={"help": "Path to pretrained weights for the base vision tower."})
    down_projector_type: Optional[str] = field(default="linear", metadata={"help": "Type of projector for connecting vision to LLM."})
    n_query: Optional[int] = field(default=729, metadata={"help": "Number of query tokens for T2I/I2I generation."})
    n_und_query: Optional[int] = field(default=729, metadata={"help": "Number of tokens for underspecified image grounding."})


@dataclass
class DataArguments:
    """Arguments pertaining to the data to be used for training."""
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = field(default=False, metadata={"help": "Whether to use lazy preprocessing."})
    is_multimodal: bool = field(default=False, metadata={"help": "Flag indicating multimodal training."})
    image_folder: Optional[str] = field(default=None, metadata={"help": "Root folder containing image data."})
    data_type: Optional[str] = field(default="mix", metadata={"help": "Type of dataset (e.g., 'mix')."})
    # These fields are initialized dynamically in train()
    gen_image_processor: Optional[object] = field(default=None)
    image_processor: Optional[object] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Arguments pertaining to the training process."""
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."},
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."},
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"


# --------------------------------------------------------------------------
# --- Preprocessing and Utilities ---
# --------------------------------------------------------------------------

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """
    Resize the tokenizer and the model's token embeddings to accommodate new special tokens.
    Initializes new embeddings by averaging existing ones.

    Args:
        special_tokens_dict (Dict): Dictionary containing new special tokens to add.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer instance.
        model (transformers.PreTrainedModel): The model instance.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        # Initialize new token embeddings by averaging the existing ones
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg


def preprocess_multimodal(sources: Sequence[List[Dict]], data_args: DataArguments) -> Tuple[Sequence[List[Dict]], Optional[str]]:
    """
    Replaces generic image placeholder tokens with model-specific query tokens 
    (T2I, I2I, or UND_IMAGE) in the conversation structure.

    Args:
        sources (Sequence[List[Dict]]): A sequence of conversations, where each conversation 
                                        is a list of message dictionaries.
        data_args (DataArguments): Data arguments containing token length (n_und_query).

    Returns:
        Tuple[Sequence[List[Dict]], Optional[str]]: 
            - sources: The modified conversation structure.
            - inst_type: The type of image task ('und' for context, 'gen' for generation), 
                         or None if no image token is found.
    """
    if not data_args.is_multimodal:
        return sources, None
        
    # Placeholder for the underspecified (contextual) image
    und_placeholder = "<|vision_start|>" + UND_IMAGE_TOKEN * data_args.n_und_query + "<|vision_end|>"
    gen_placeholder = ""
    inst_type = None

    for source in sources:  # Iterate through batch/instance
        for sentence in source: # Iterate through conversation turns
            if sentence["from"] == "human" and DEFAULT_IMAGE_TOKEN in sentence["value"]:
                # Replace default image token with the multi-token placeholder for *contextual* image
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, und_placeholder).strip()
                inst_type = "und"
            elif sentence["from"] == "gpt" and DEFAULT_IMAGE_TOKEN in sentence["value"]:
                # The generative token will be replaced by T2I/I2I query tokens later in data collator
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, gen_placeholder).strip()
                inst_type = "gen"
                
    return sources, inst_type


def preprocess_qwen(sources: Sequence[List[Dict]], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    """
    Tokenizes the conversation data according to the Qwen chat template, 
    creating input_ids and attention mask/labels.

    Args:
        sources (Sequence[List[Dict]]): A sequence of preprocessed conversations.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer instance.
        has_image (bool): Flag indicating if the instance contains an image.
        system_message (str): The initial system instruction.

    Returns:
        Dict: Dictionary containing 'input_ids' and 'labels' tensors.
    """
    roles = {"human": "user", "gpt": "assistant"}

    # Use a deep copy to avoid modifying the global tokenizer chat template if parallel processing is used
    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # 1. Build system message
        system_conv = [{"role" : "system", "content" : system_message}]
        input_id += tokenizer.apply_chat_template(system_conv)
        target += [IGNORE_INDEX] * len(input_id)

        # 2. Process conversation turns
        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except: # Handle older format
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            
            conv_turn = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv_turn)
            input_id += encode_id
            
            # Mask out user/system input for loss calculation
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id # Label the GPT responses
        

                    
        assert len(input_id) == len(target), f"Input length mismatch: {len(input_id)} != {len(target)}"

        input_ids.append(input_id)
        targets.append(target)
        
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def parse_txt(example: Dict) -> Dict:
    """Extracts text prompt from the 'edited_prompt_list' field (for I2I data)."""
    example["txt"] = example["edited_prompt_list"][-1]
    return example

def parse_json(example: Dict) -> Dict:
    """Extracts text prompt from the 'json' field (for T2I data)."""
    if isinstance(example["json"], str):
        try:
            parsed = json.loads(example["json"])
            example["txt"] = parsed.get("prompt", "") 
        except json.JSONDecodeError:
            example["txt"] = "" 
    else:
        example["txt"] = example["json"].get("prompt", "")  
    return example

def get_t2i_data_files(num_files: int = 10) -> List[str]:
    """Generate list of T2I dataset URLs."""
    
    num_files = min(46, num_files)
    base_url = "https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_{:06d}.tar"
    return [base_url.format(i) for i in range(num_files)]

def get_edit_data_files(num_files: int = 10) -> List[str]:
    """Generate list of image editing dataset URLs."""
    num_files = min(571, num_files)
    base_url = "https://huggingface.co/datasets/TIGER-Lab/OmniEdit-Filtered-1.2M/resolve/main/data/train-{:05d}-of-00571.parquet"
    return [base_url.format(i) for i in range(num_files)]  # 00000 to 00570

# --------------------------------------------------------------------------
# --- Dataset and Collation ---
# --------------------------------------------------------------------------

class LazySupervisedMixDataset(Dataset):
    """
    Dataset for supervised fine-tuning, loading and mixing I2I and T2I datasets 
    from Parquet and WebDataset formats.
    """

    def __init__(
        self,
        data_path: str, # Unused but kept for signature consistency
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        super(LazySupervisedMixDataset, self).__init__()

        self.data_args = data_args
        list_data_dict = []

        # 1. Load and process Image-to-Image (I2I) dataset
        rank0_print("Loading I2I dataset...")
        data_files = get_edit_data_files(num_files=1)  # Load first 1 files for example
        i2i_dataset = (load_dataset("parquet", data_files=data_files, split="train", num_proc=16)
                    .map(parse_txt, num_proc=16)
                    .rename_columns({"src_img": "ref_image", "edited_img": "image"}))
        i2i_dataset = (i2i_dataset.add_column('type', len(i2i_dataset) * ['I2I'])
                       .select_columns(["image", "txt", "type", "ref_image"]))
        
        rank0_print(f"Finished loading I2I dataset with {len(i2i_dataset)} samples")

        # 2. Load and process Text-to-Image (T2I) dataset
        rank0_print("Loading T2I dataset...")
        data_files = get_t2i_data_files(num_files=1)  # Load first 1 files for example
        t2i_dataset = (load_dataset("webdataset", data_files=data_files, split="train", num_proc=64)
                    .map(parse_json, num_proc=64)
                    .rename_columns({"jpg": "image"}))
        t2i_dataset = (t2i_dataset.add_column("type", ["T2I"] * len(t2i_dataset))
                    .select_columns(["image", "txt", "type"]))
        rank0_print(f"Finished loading T2I dataset with {len(t2i_dataset)} samples")

        list_data_dict.append(i2i_dataset)
        list_data_dict.append(t2i_dataset)
        

        # 3. Concatenate and Shuffle
        if len(list_data_dict) > 1:
            list_data_dict = concatenate_datasets(list_data_dict)
        else:
            list_data_dict = list_data_dict[0]
            
        list_data_dict = list_data_dict.shuffle(seed=42)

        rank0_print(f"Totoal number of training instance: {len(list_data_dict)}")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """
        Retrieves a single data instance, constructs the conversation, 
        and applies preprocessing.
        """
        while True: # Loop to handle data loading errors
            sources = self.list_data_dict[i]
            task_type = sources["type"]

            # 1. Construct the conversation based on task type
            if task_type == "T2I":
                # T2I: Human asks for generation, GPT replies with <image>
                sources["conversations"] = [
                    {"from": "human", "value": f"Please generate image based on the following caption: {sources['txt']}"},
                    {"from": "gpt", "value": DEFAULT_IMAGE_TOKEN}, # DEFAULT_IMAGE_TOKEN will be replaced by empty string in preprocess_multimodal
                ]
                image_files = sources["image"]   
                ref_image_files = None  
                
            elif task_type == "I2I":
                # I2I: Human provides image (<image>) and instruction, GPT replies with generation (<image>)
                sources["conversations"] = [
                    {
                        "from": "human",
                        "value": f"{DEFAULT_IMAGE_TOKEN}\nPlease edit the given image based on the folllowing instruction: {sources['txt']}.",
                    },
                    {"from": "gpt", "value": DEFAULT_IMAGE_TOKEN},
                ]
                image_files = sources["image"]
                ref_image_files = sources["ref_image"]
            else:
                raise ValueError("Unknown source type. Please check the 'type' in 'sources'.")

            # 2. Load images (handles bytes/PIL images from dataset)
            if not isinstance(image_files, list):
                image_files = [image_files]
            if not isinstance(ref_image_files, list) and ref_image_files is not None:
                ref_image_files = [ref_image_files]
                
            images, ref_images = [], []

            # Load target image(s)
            if task_type == "T2I" or task_type == "I2I":
                for img in image_files:
                    try:
                        # Ensure image is in RGB format
                        if isinstance(img, bytes): img = Image.open(io.BytesIO(img))
                        img = img.convert("RGB")
                        images.append(img)
                    except Exception as e:
                        print(f"Error opening image: {e}. Skipping sample.")
                        images = None; break
            
            # Load reference image(s) for I2I
            if task_type == "I2I" and images is not None:
                for img in ref_image_files:
                    try:
                        if isinstance(img, bytes): img = Image.open(io.BytesIO(img))
                        img = img.convert("RGB")
                        ref_images.append(img)
                    except Exception as e:
                        print(f"Error opening ref image: {e}. Skipping sample.")
                        ref_images = None; break
            
            if images is None or (task_type == "I2I" and ref_images is None):
                i = random.randint(0, len(self.list_data_dict) - 1)
                continue # Skip to the next random index if image loading failed

            # 3. Apply preprocessing to conversation and tokenize
            # Replaces DEFAULT_IMAGE_TOKEN with UND_IMAGE_TOKEN/empty string
            sources, inst_type = preprocess_multimodal(copy.deepcopy([sources["conversations"]]), self.data_args)
            data_dict = preprocess_qwen(sources, self.tokenizer, has_image=("image" in self.list_data_dict[i]))
            
            if isinstance(i, int):
                data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
            
            data_dict["task_types"] = task_type

            # 4. Process images using VAE/Vision Tower processors
            if inst_type == "gen": # T2I or I2I (where the LLM output involves generation)
                # Target image for VAE (latents for diffusion)
                data_dict["gen_image"] = self.data_args.gen_image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
                
                # I2I reference image (for diffusion conditioning)
                if task_type == "I2I":
                    data_dict["gen_ref_image"] = self.data_args.gen_image_processor.preprocess(ref_images, return_tensors="pt")["pixel_values"]
                    # ref_mask is 1 for I2I
                    b = data_dict["gen_ref_image"].shape[0]
                    data_dict["ref_mask"] = torch.ones((b, 1, 1, 1), dtype=data_dict["gen_image"].dtype, device=data_dict["gen_image"].device)
                else: # T2I uses a masked/zeroed reference
                    data_dict["gen_ref_image"] = torch.zeros_like(data_dict["gen_image"])
                    b = data_dict["gen_ref_image"].shape[0]
                    # ref_mask is 0 for T2I (no reference image conditioning)
                    data_dict["ref_mask"] = torch.zeros((b, 1, 1, 1), dtype=data_dict["gen_image"].dtype, device=data_dict["gen_image"].device)
                
            elif inst_type == "und": # I2I (where the LLM input involves image grounding)
                # Underspecified image (resized for LLM context tokens)
                resized_images = [transform_und_images(img) for img in ref_images]
                image_inputs = self.data_args.image_processor(resized_images, return_tensors="pt")
                data_dict["und_image"] = image_inputs.pixel_values
                data_dict["grid_thw"] = image_inputs.image_grid_thw # Grid info for the visual encoder

                # Target and Reference images for VAE (diffusion loss)
                data_dict["gen_image"] = self.data_args.gen_image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
                data_dict["gen_ref_image"] = self.data_args.gen_image_processor.preprocess(ref_images, return_tensors="pt")["pixel_values"]
                
                b = data_dict["gen_ref_image"].shape[0]
                data_dict["ref_mask"] = torch.ones((b, 1, 1, 1), dtype=data_dict["gen_image"].dtype, device=data_dict["gen_image"].device)

            data_dict["ids"] = self.list_data_dict[i]["id"] if "id" in self.list_data_dict[i] else "unk"
            
            return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """
    Collate examples for supervised fine-tuning, primarily handling padding 
    and appending the T2I/I2I query tokens to the sequence.
    """

    tokenizer: transformers.PreTrainedTokenizer
    n_query: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, ids, task_types = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "ids", "task_types"))
        multi_input_ids = []
        multi_labels = []
        i_s_pos = []

        # 1. Truncate and Append Image Query Tokens
        for input_id, label, task_type in zip(input_ids, labels, task_types):
            # Truncate to make space for the image query tokens (n_query)
            # The final length will be truncated to model_max_length in step 2.
            input_id = input_id[: self.tokenizer.model_max_length - (self.n_query)]
            label = label[: self.tokenizer.model_max_length - (self.n_query)]
            
            # i_s_pos records the starting index of the appended image query tokens
            i_s_pos.append(input_id.shape[0])
            
            # Create the sequence of image query tokens
            img_id = torch.full((self.n_query,), T2I_TOKEN_IDX, dtype=input_id.dtype, device=input_id.device) if task_type == "T2I" else torch.full((self.n_query,), I2I_TOKEN_IDX, dtype=input_id.dtype, device=input_id.device)
            
            # Append tokens to input_id and label
            input_id = torch.cat([input_id, img_id])
            img_label = torch.full((self.n_query,), T2I_TOKEN_IDX, dtype=label.dtype, device=label.device) if task_type == "T2I" else torch.full((self.n_query,), I2I_TOKEN_IDX, dtype=label.dtype, device=label.device)
            label = torch.cat([label, img_label])
            
            multi_input_ids.append(input_id)
            multi_labels.append(label)

        input_ids = multi_input_ids
        labels = multi_labels

        # 2. Pad Sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) 
        
        # Final truncation to max length
        if input_ids.shape[1] > self.tokenizer.model_max_length:
            rank0_print(f"Warning: input with length {input_ids.shape[1]} is longer than max length {self.tokenizer.model_max_length}. Truncating.")
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            # Attention mask ignores the padding tokens
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id), 
        )

        # 3. Collate Image Data (VAE latents and contextual images)
        batch_gen_images = []
        batch_ref_gen_images = []
        batch_und_images = []
        batch_grid_thw = []
        batch_ref_mask = []
        
        # Collate gen_image (target VAE latents)
        for instance in instances:
            if "gen_image" in instance:
                batch_gen_images.append(instance["gen_image"])

        if len(batch_gen_images) > 0:
            # Check for consistent shape and concatenate
            if all(x is not None and y.shape == batch_gen_images[0][0].shape for x in batch_gen_images for y in x):
                batch["gen_image"] = torch.cat([images for images in batch_gen_images], dim=0)
            else:
                batch["gen_image"] = batch_gen_images # Keep as list if shapes vary (unlikely for VAE)

        # Collate gen_ref_image (reference VAE latents) and ref_mask
        for instance in instances:
            if "gen_ref_image" in instance:
                batch_ref_gen_images.append(instance["gen_ref_image"])
            if "ref_mask" in instance:
                batch_ref_mask.append(instance["ref_mask"])

        if len(batch_ref_gen_images) > 0:
            if all(x is not None and y.shape == batch_ref_gen_images[0][0].shape for x in batch_ref_gen_images for y in x):
                batch["gen_ref_image"] = torch.cat([images for images in batch_ref_gen_images], dim=0)
            else:
                batch["gen_ref_image"] = batch_ref_gen_images
        
        if len(batch_ref_mask) > 0:
            batch["ref_mask"] = torch.cat([images for images in batch_ref_mask], dim=0)


        # Collate und_image (contextual image pixels) and grid_thw
        for instance in instances:
            if "und_image" in instance:
                batch_und_images.append(instance["und_image"].unsqueeze(0))
                batch_grid_thw.append(instance["grid_thw"]) 

        if len(batch_und_images) > 0:
            batch["und_image"] = torch.cat([images for images in batch_und_images], dim=0)
            batch["grid_thw"] = torch.cat([images for images in batch_grid_thw], dim=0)
        else:
            batch["und_image"] = None
            batch["grid_thw"] = None

        # Final metadata
        batch["ids"] = ids
        batch["i_s_pos"] = i_s_pos # Start positions of the image query tokens

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments) -> Dict:
    """
    Creates the training dataset and data collator based on data arguments.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The model tokenizer.
        data_args (DataArguments): Configuration for data loading.

    Returns:
        Dict: Dictionary containing 'train_dataset', 'eval_dataset', and 'data_collator'.
    """

    if data_args.data_type == "mix":
        train_dataset = LazySupervisedMixDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    else:
        raise ValueError(f"Unknown data type: {data_args.data_type}.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, n_query=data_args.n_query)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


# --------------------------------------------------------------------------
# --- Main Training Function ---
# --------------------------------------------------------------------------

def train(attn_implementation=None):
    """
    Main function to parse arguments, set up model, tokenizer, and start training.
    """
    global local_rank

    # 1. Parse Arguments
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    rank0_print(model_args, data_args, training_args)
    local_rank = training_args.local_rank
    
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    # 2. Configure Quantization (if bits is 4 or 8)
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"], # Skip projection layer from quantization
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,
                ),
            )
        )

    # 3. Load Model (blip3oQwenForCausalLM)
    model = blip3oQwenForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        **bnb_model_from_pretrained_args,
    )

    model.config.use_cache = False

    # 4. Apply Freezing
    if model_args.freeze_backbone:
        rank0_print("Freezing LLM backbone, base visual encoder, and LM head.")
        for (n, p) in model.get_model().named_parameters():
            p.requires_grad = False
        for (n, p) in model.visual.named_parameters():
            p.requires_grad = False
        for (n, p) in model.lm_head.named_parameters():
            p.requires_grad = False
    
    # Enable gradient checkpointing requirements for proper training
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    # 5. Load and Prepare Tokenizer
    try:
        tokenizer = AutoProcessor.from_pretrained(model_args.model_name_or_path).tokenizer
    except Exception as e:
        # Handle case where processor might not have a .tokenizer attribute directly
        tokenizer = AutoProcessor.from_pretrained(model_args.model_name_or_path) 
        
    tokenizer.model_max_length = training_args.model_max_length

    # Resize embeddings to include new special tokens for T2I/I2I
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(additional_special_tokens=["<image>", "<T2I>", "<I2I>"]),
        tokenizer=tokenizer,
        model=model,
    )
    
    # Verification of token indices
    rank0_print(f"Token ID for <T2I>: {tokenizer.encode('<T2I>', return_tensors='pt').cpu().item()}")
    rank0_print(f"Token ID for <I2I>: {tokenizer.encode('<I2I>', return_tensors='pt').cpu().item()}")
    assert tokenizer.encode("<T2I>", return_tensors="pt").cpu().item() == T2I_TOKEN_IDX
    assert tokenizer.encode("<I2I>", return_tensors="pt").cpu().item() == I2I_TOKEN_IDX
    
    
    # 6. Set Conversation Template
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["llama3"]
    rank0_print(f"Using conversation format: {conversation_lib.default_conversation.version}")

    # 7. Initialize Vision Modules (Projectors, DiT)
    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

    # 8. Configure Generative Vision Tower (VAE)
    gen_vision_tower = model.get_gen_vision_tower()
    # Move VAE to device and set to non-trainable
    gen_vision_tower.to(
        dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        device=training_args.device,
    )
    gen_vision_tower.requires_grad_(False)

    # Set image processors for the DataArguments
    data_args.gen_image_processor = gen_vision_tower.image_processor # For VAE input
    data_args.image_processor = AutoProcessor.from_pretrained(model_args.model_name_or_path).image_processor # For contextual image input

    # Final data arguments configuration
    data_args.is_multimodal = True
    data_args.n_query = model_args.n_query
    data_args.n_und_query = model_args.n_und_query
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.pad_token_id = tokenizer.pad_token_id

    # 9. Log Trainable Parameters
    rank0_print("Trainable parameters:")
    for name, param in model.get_model().named_parameters():
        if param.requires_grad:
            rank0_print(f"  - {name}, shape: {param.shape}")

    total_params = sum(p.numel() for p in model.get_model().parameters())
    trainable_params = sum(p.numel() for p in model.get_model().parameters() if p.requires_grad)

    rank0_print(f"\nTotal parameters: {total_params}")
    rank0_print(f"Trainable parameters: {trainable_params}")

    # 10. Prepare Data Module
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # 11. Initialize and Start Trainer
    trainer = blip3oTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )
    
    # Print parameter table (optional, for debugging)
    if trainer.is_world_process_zero():
        from tabulate import tabulate
        stat = []
        for i, (n, p) in enumerate(trainer.model.named_parameters()):
            stat.append([i, n, p.shape, p.requires_grad])
        rank0_print(tabulate(stat, headers=["idx", "name", "shape", "trainable"]))
        
    # Start training, potentially resuming from a checkpoint
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

if __name__ == "__main__":
    train()