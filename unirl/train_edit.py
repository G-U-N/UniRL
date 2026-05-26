import json
import logging
import os
import re
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional


# Silence the per-step CLIP-truncation warnings emitted by the FluxKontext pipeline's
# CLIP path. Long Qwen-refined prompts are expected to overflow CLIP's 77-token limit;
# the T5 encoder still receives the full text, so the truncation is benign here. Two
# loggers fire: the pipeline's own "CLIP can only handle sequences..." warning, and the
# upstream tokenizer's "Token indices sequence length is longer..." warning.
# (Klein has no CLIP encoder, so these are no-ops for the QwenKlein backbone.)
logging.getLogger("unimodel.qwenkontext.fluxkontext_pipeline").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from datasets import load_dataset
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from .reward_evaluator import RewardEvaluatorClient
from .trainer import EditJointGRPOTrainer


class PromptRLConfig(GRPOConfig):
    """GRPOConfig variant for the edit trainer.

    trl's GRPOConfig.__post_init__ requires the effective per-step batch to be divisible by
    `num_generations` because it assumes a trl-style rollout where the batch already contains
    the N generations. In this trainer each example is expanded internally to
    `num_generations` samples (see EditJointGRPOTrainer), so the divisibility check is a
    false positive. We temporarily set `num_generations` to the effective batch size (which
    always divides itself and is >= 2 in any realistic distributed setup) while running the
    parent's post-init, then restore the real value.
    """

    def __post_init__(self):
        real_num_generations = self.num_generations
        world_size = int(os.environ.get("WORLD_SIZE", "1") or 1)
        effective_batch = (
            world_size
            * max(self.per_device_train_batch_size, 1)
            * max(self.gradient_accumulation_steps, 1)
        )
        self.num_generations = max(2, effective_batch)
        try:
            super().__post_init__()
        finally:
            self.num_generations = real_num_generations


DEFAULT_EDIT_DATASET = "https://huggingface.co/wangfuyun/PrompRL/resolve/main/data/omni_edit_train_50k.parquet"

EDIT_QUESTION_TEMPLATE = """Please provide an enhanced prompt for the following image editing prompt.
Ensure the revised prompt is clear, specific, and includes detailed instructions to achieve the desired outcome while maintaining the original intent.
Original prompt: {Question}. Directly provide the improved prompt in <answer> </answer> tags."""


@dataclass
class EditGRPOScriptArguments(ScriptArguments):
    reward_funcs: List[str] = field(
        default_factory=lambda: ["editreward", "format"],
        metadata={"help": "Reward functions to use. Edit training supports only: editreward, format."},
    )
    prompts_file: str = field(
        default=DEFAULT_EDIT_DATASET,
        metadata={"help": "Path or URL to a .parquet or .jsonl edit-training dataset."},
    )
    image_column: str = field(default="image", metadata={"help": "Dataset column containing the source image."})
    prompt_column: str = field(default="prompt", metadata={"help": "Dataset column containing the edit instruction."})
    caption_column: Optional[str] = field(default="caption", metadata={"help": "Optional source-caption column."})
    target_caption_column: Optional[str] = field(
        default="target_caption",
        metadata={"help": "Optional target-caption column."},
    )
    image_size: int = field(default=512, metadata={"help": "Center-cropped source image size used for editing."})
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional Hugging Face datasets cache dir. Defaults to HF_DATASETS_CACHE."},
    )
    editreward_url: str = field(
        default="http://127.0.0.1:18088/",
        metadata={"help": "HTTP URL of the EditReward scorer service."},
    )
    processor_name_or_path: str = field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        metadata={"help": "Processor used for Qwen2.5-VL chat formatting and image preprocessing."},
    )
    max_pixels: int = field(default=200704, metadata={"help": "Maximum pixels passed to the Qwen-VL processor."})
    min_pixels: int = field(default=200704, metadata={"help": "Minimum pixels passed to the Qwen-VL processor."})
    num_skip_refinement: int = field(
        default=2,
        metadata={"help": "Generations per input that use the original edit prompt instead of a Qwen-refined prompt."},
    )
    num_sde: int = field(
        default=4,
        metadata={"help": "Number of FLUX denoising steps sampled with SDE log-prob tracking for diffusion GRPO."},
    )


class EditPromptDataset(Dataset):
    """Loads image-edit instructions from parquet or jsonl files."""

    def __init__(
        self,
        prompts_file: str,
        question_template: str,
        image_column: str = "image",
        prompt_column: str = "prompt",
        caption_column: Optional[str] = "caption",
        target_caption_column: Optional[str] = "target_caption",
        image_size: int = 512,
        cache_dir: Optional[str] = None,
    ):
        self.prompts_file = normalize_data_path(prompts_file)
        self.question_template = question_template
        self.image_column = image_column
        self.prompt_column = prompt_column
        self.caption_column = caption_column
        self.target_caption_column = target_caption_column
        self.image_size = image_size
        self.base_dir = (
            os.path.dirname(os.path.abspath(self.prompts_file))
            if not is_remote_path(self.prompts_file)
            else os.getcwd()
        )

        if self.prompts_file.endswith(".parquet"):
            self.records = load_dataset(
                "parquet",
                data_files={"train": self.prompts_file},
                split="train",
                cache_dir=cache_dir or os.getenv("HF_DATASETS_CACHE"),
            )
        elif self.prompts_file.endswith(".jsonl") or self.prompts_file.endswith(".json"):
            if is_remote_path(self.prompts_file):
                self.records = load_dataset(
                    "json",
                    data_files={"train": self.prompts_file},
                    split="train",
                    cache_dir=cache_dir or os.getenv("HF_DATASETS_CACHE"),
                )
            else:
                with open(self.prompts_file, "r", encoding="utf-8") as file:
                    self.records = [json.loads(line) for line in file if line.strip()]
        else:
            raise ValueError("Edit training datasets must be .parquet or .jsonl files.")

        if len(self.records) == 0:
            raise ValueError(f"No training records found in {prompts_file}.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.records[index]
        instruction = self._read_text(item, self.prompt_column)
        image = self._read_image(item, self.image_column)
        formatted_prompt = self.question_template.format(Question=instruction)

        return {
            "image": image,
            "caption": self._read_optional_text(item, self.caption_column),
            "target_caption": self._read_optional_text(item, self.target_caption_column),
            "editing_instruction": instruction,
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": formatted_prompt},
                    ],
                }
            ],
        }

    def _read_text(self, item: Dict[str, Any], column: str) -> str:
        if column not in item:
            raise KeyError(f"Missing required column '{column}' in {self.prompts_file}.")
        value = item[column]
        if isinstance(value, list):
            value = value[-1] if value else ""
        if value is None or not str(value).strip():
            raise ValueError(f"Empty edit instruction in column '{column}'.")
        return str(value).strip()

    def _read_optional_text(self, item: Dict[str, Any], column: Optional[str]) -> str:
        if not column or column not in item or item[column] is None:
            return ""
        value = item[column]
        if isinstance(value, list):
            value = value[-1] if value else ""
        return str(value).strip()

    def _read_image(self, item: Dict[str, Any], column: str) -> Image.Image:
        if column not in item:
            raise KeyError(f"Missing required image column '{column}' in {self.prompts_file}.")
        image = self._coerce_image(item[column])
        if self.image_size > 0:
            image = ImageOps.fit(image, (self.image_size, self.image_size), method=Image.Resampling.BICUBIC)
        return image.convert("RGB")

    def _coerce_image(self, value: Any) -> Image.Image:
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, str):
            image_path = value if os.path.isabs(value) else os.path.join(self.base_dir, value)
            return Image.open(image_path).convert("RGB")
        if isinstance(value, bytes):
            return Image.open(BytesIO(value)).convert("RGB")
        if isinstance(value, dict):
            if value.get("bytes") is not None:
                return Image.open(BytesIO(value["bytes"])).convert("RGB")
            if value.get("path") is not None:
                image_path = value["path"] if os.path.isabs(value["path"]) else os.path.join(self.base_dir, value["path"])
                return Image.open(image_path).convert("RGB")
        raise TypeError(f"Unsupported image value type: {type(value)!r}")


def is_remote_path(path: str) -> bool:
    return path.startswith(("http://", "https://", "hf://"))


def normalize_data_path(path: str) -> str:
    if path.startswith("hf://"):
        parts = path[len("hf://") :].split("/", 2)
        if len(parts) != 3:
            raise ValueError("hf:// dataset paths must look like hf://owner/repo/path/to/file.parquet")
        repo_id = f"{parts[0]}/{parts[1]}"
        file_path = parts[2]
        return f"https://huggingface.co/{repo_id}/resolve/main/{file_path}"

    if "huggingface.co/" in path and "/blob/" in path:
        return path.replace("/blob/", "/resolve/", 1)
    return path


def format_reward(completions: List[str]) -> List[float]:
    pattern = re.compile(r"<answer>.*?</answer>", re.DOTALL)
    return [1.0 if pattern.search(completion) else 0.0 for completion in completions]


def build_editreward_func(editreward_url: str):
    reward_client = RewardEvaluatorClient(editreward_url=editreward_url)

    def editreward(source_images, edited_images, prompts):
        return reward_client.evaluate_editreward(source_images, edited_images, prompts)

    return editreward


def main(script_args: EditGRPOScriptArguments, training_args: PromptRLConfig, model_args: ModelConfig) -> None:
    supported_rewards = {"editreward", "format"}
    unsupported_rewards = sorted(set(script_args.reward_funcs) - supported_rewards)
    if unsupported_rewards:
        raise ValueError(f"Edit training supports only {sorted(supported_rewards)}, got {unsupported_rewards}.")

    reward_registry = {
        "editreward": build_editreward_func(script_args.editreward_url),
        "format": format_reward,
    }
    reward_funcs = [(name, None, reward_registry[name]) for name in script_args.reward_funcs]

    train_dataset = EditPromptDataset(
        prompts_file=script_args.prompts_file,
        question_template=EDIT_QUESTION_TEMPLATE,
        image_column=script_args.image_column,
        prompt_column=script_args.prompt_column,
        caption_column=script_args.caption_column,
        target_caption_column=script_args.target_caption_column,
        image_size=script_args.image_size,
        cache_dir=script_args.dataset_cache_dir,
    )

    trainer = EditJointGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=get_peft_config(model_args),
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        processor_name_or_path=script_args.processor_name_or_path,
        attn_implementation=model_args.attn_implementation,
        num_skip_refinement=script_args.num_skip_refinement,
        num_sde=script_args.num_sde,
    )

    checkpoint = get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((EditGRPOScriptArguments, PromptRLConfig, ModelConfig))
    parsed_script_args, parsed_training_args, parsed_model_args = parser.parse_args_and_config()
    main(parsed_script_args, parsed_training_args, parsed_model_args)
