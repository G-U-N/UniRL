"""GenEval prompt-refinement GRPO entrypoint.

Selects between `GenEvalPromptGRPOTrainer` (LLM-only) and
`GenEvalJointGRPOTrainer` (LLM + DiT) via the ``--joint`` flag. Backbone
(qwenflux / qwenklein) is auto-detected from ``model.model_name_or_path``.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Silence CLIP-truncation warnings on the FLUX.1-dev path. The pipeline's CLIP
# encoder truncates Qwen-refined prompts past 77 tokens; T5 still receives the
# full text so the truncation is benign here. Klein has no CLIP, so this is a no-op
# for the qwenklein backbone.
logging.getLogger("unimodel.qwenflux.flux_pipeline").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from .reward_evaluator import RewardEvaluatorClient
from .trainer import GenEvalJointGRPOTrainer, GenEvalPromptGRPOTrainer


class PromptRLConfig(GRPOConfig):
    """GRPOConfig variant that bypasses trl's batch/num_generations divisibility check.

    See `unirl.train_edit.PromptRLConfig` for the rationale: in PrompRL each example
    is internally expanded to ``num_generations`` samples, so the upstream check is
    a false positive.
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


DEFAULT_GENEVAL_DATASET = "assets/rl_datasets/train_metadata.jsonl"

GENEVAL_QUESTION_TEMPLATE = (
    "Please provide an enhanced prompt for the following image generation prompt to "
    "make the image more realistic, detailed, with clear separation and precise alignment "
    "of all entities.\nOriginal prompt: {Question}. "
    "Directly provide the improved prompt in <answer> </answer> tags."
)


@dataclass
class GenEvalGRPOScriptArguments(ScriptArguments):
    reward_funcs: List[str] = field(
        default_factory=lambda: ["gen_eval", "format"],
        metadata={"help": "Reward functions to use. GenEval training supports: gen_eval, format."},
    )
    prompts_file: str = field(
        default=DEFAULT_GENEVAL_DATASET,
        metadata={"help": "Path or URL to a .jsonl GenEval prompt metadata file."},
    )
    prompt_column: str = field(
        default="prompt",
        metadata={"help": "JSONL field containing the original (unrefined) generation prompt."},
    )
    geneval_url: str = field(
        default="http://127.0.0.1:18085/",
        metadata={"help": "HTTP URL of the GenEval scorer service."},
    )
    geneval_only_strict: bool = field(
        default=False,
        metadata={"help": "If true, request only strict (binary) GenEval rewards from the scorer."},
    )
    processor_name_or_path: str = field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        metadata={"help": "Processor used for Qwen2.5-VL chat formatting."},
    )
    joint: bool = field(
        default=False,
        metadata={"help": "If true, jointly update the FLUX transformer (SDE-tracked PPO loss)."},
    )
    num_sde: int = field(
        default=8,
        metadata={
            "help": "Number of leading denoising steps tracked with SDE log-probs (used only with --joint)."
        },
    )
    num_inference_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Override the default per-backbone num_inference_steps (20 for both)."},
    )
    guidance_scale: Optional[float] = field(
        default=None,
        metadata={"help": "Override the default per-backbone guidance scale."},
    )
    image_height: Optional[int] = field(
        default=None,
        metadata={"help": "Generated image height (defaults to 1024)."},
    )
    image_width: Optional[int] = field(
        default=None,
        metadata={"help": "Generated image width (defaults to 1024)."},
    )


class GenEvalPromptDataset(Dataset):
    """Loads GenEval prompt metadata records from a local .jsonl or remote URL."""

    def __init__(
        self,
        prompts_file: str,
        question_template: str,
        prompt_column: str = "prompt",
        cache_dir: Optional[str] = None,
    ):
        self.prompts_file = normalize_data_path(prompts_file)
        self.question_template = question_template
        self.prompt_column = prompt_column

        if self.prompts_file.endswith(".jsonl") or self.prompts_file.endswith(".json"):
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
        elif self.prompts_file.endswith(".parquet"):
            self.records = load_dataset(
                "parquet",
                data_files={"train": self.prompts_file},
                split="train",
                cache_dir=cache_dir or os.getenv("HF_DATASETS_CACHE"),
            )
        else:
            raise ValueError("GenEval datasets must be .jsonl, .json, or .parquet files.")

        if len(self.records) == 0:
            raise ValueError(f"No GenEval prompt records found in {prompts_file}.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        if self.prompt_column not in record:
            raise KeyError(
                f"Record {index} missing required column '{self.prompt_column}': keys={list(record.keys())}"
            )
        original_prompt = str(record[self.prompt_column]).strip()
        formatted = self.question_template.format(Question=original_prompt)

        # GenEval metadata = the whole record minus the bare prompt field; the
        # scorer service consumes {"tag", "include", "exclude"} but we forward
        # everything for forward compatibility with future metadata fields.
        metadata = {key: value for key, value in record.items() if key != self.prompt_column}
        metadata.setdefault("prompt", original_prompt)

        return {
            "caption": original_prompt,
            "metadata": metadata,
            "prompt": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": formatted}],
                }
            ],
        }


def is_remote_path(path: str) -> bool:
    return path.startswith(("http://", "https://", "hf://"))


def normalize_data_path(path: str) -> str:
    if path.startswith("hf://"):
        parts = path[len("hf://") :].split("/", 2)
        if len(parts) != 3:
            raise ValueError("hf:// dataset paths must look like hf://owner/repo/path/to/file.jsonl")
        repo_id = f"{parts[0]}/{parts[1]}"
        file_path = parts[2]
        return f"https://huggingface.co/{repo_id}/resolve/main/{file_path}"
    if "huggingface.co/" in path and "/blob/" in path:
        return path.replace("/blob/", "/resolve/", 1)
    return path


def format_reward(completions: List[str]) -> List[float]:
    pattern = re.compile(r"<answer>.*?</answer>", re.DOTALL)
    return [1.0 if pattern.search(completion) else 0.0 for completion in completions]


def build_geneval_func(geneval_url: str, only_strict: bool):
    client = RewardEvaluatorClient(geneval_url=geneval_url)

    def gen_eval(images, prompts, meta_datas):
        return client.evaluate_geneval(images, prompts, meta_datas, only_strict=only_strict)

    return gen_eval


def main(
    script_args: GenEvalGRPOScriptArguments,
    training_args: PromptRLConfig,
    model_args: ModelConfig,
) -> None:
    supported_rewards = {"gen_eval", "format"}
    unsupported = sorted(set(script_args.reward_funcs) - supported_rewards)
    if unsupported:
        raise ValueError(
            f"GenEval training supports only {sorted(supported_rewards)}, got {unsupported}."
        )

    reward_registry = {
        "gen_eval": build_geneval_func(script_args.geneval_url, script_args.geneval_only_strict),
        "format": format_reward,
    }
    reward_funcs = [(name, None, reward_registry[name]) for name in script_args.reward_funcs]

    train_dataset = GenEvalPromptDataset(
        prompts_file=script_args.prompts_file,
        question_template=GENEVAL_QUESTION_TEMPLATE,
        prompt_column=script_args.prompt_column,
    )

    trainer_cls = GenEvalJointGRPOTrainer if script_args.joint else GenEvalPromptGRPOTrainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=get_peft_config(model_args),
        processor_name_or_path=script_args.processor_name_or_path,
        attn_implementation=model_args.attn_implementation,
        num_sde=script_args.num_sde,
        num_inference_steps=script_args.num_inference_steps,
        guidance_scale=script_args.guidance_scale,
        height=script_args.image_height,
        width=script_args.image_width,
    )

    checkpoint = (
        get_last_checkpoint(training_args.output_dir)
        if os.path.isdir(training_args.output_dir)
        else None
    )
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GenEvalGRPOScriptArguments, PromptRLConfig, ModelConfig))
    parsed_script_args, parsed_training_args, parsed_model_args = parser.parse_args_and_config()
    main(parsed_script_args, parsed_training_args, parsed_model_args)
