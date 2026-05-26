import os
import shutil
import sys
import tempfile
from typing import List

import torch
from PIL import Image


class EditRewardScorer(torch.nn.Module):
    def __init__(
        self,
        repo_dir: str,
        config_path: str,
        checkpoint_path: str,
        reward_dim: str = "overall_detail",
        rm_head_type: str = "ranknet_multi_head",
        device: str = "cuda",
    ):
        super().__init__()
        if not os.path.isdir(repo_dir):
            raise FileNotFoundError(
                f"EditReward repository not found at {repo_dir}. "
                "Clone https://github.com/TIGER-AI-Lab/EditReward.git or set EDITREWARD_REPO_DIR."
            )
        sys.path.insert(0, repo_dir)
        from EditReward import EditRewardInferencer

        self.inferencer = EditRewardInferencer(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
            reward_dim=reward_dim,
            rm_head_type=rm_head_type,
        )
        self.device = device
        self.eval()

    @torch.no_grad()
    def __call__(self, prompts: List[str], source_images: List[Image.Image], edited_images: List[Image.Image]):
        if not (len(prompts) == len(source_images) == len(edited_images)):
            raise ValueError("prompts, source_images, and edited_images must have the same length.")

        temp_dir = tempfile.mkdtemp(prefix="editreward_")
        try:
            source_paths = []
            edited_paths = []
            for index, (source_image, edited_image) in enumerate(zip(source_images, edited_images)):
                source_path = os.path.join(temp_dir, f"source_{index}.png")
                edited_path = os.path.join(temp_dir, f"edited_{index}.png")
                source_image.convert("RGB").save(source_path)
                edited_image.convert("RGB").save(edited_path)
                source_paths.append(source_path)
                edited_paths.append(edited_path)

            rewards = self.inferencer.reward(
                prompts=prompts,
                image_src=source_paths,
                image_paths=edited_paths,
            )
            return [reward[0].item() if hasattr(reward[0], "item") else float(reward[0]) for reward in rewards]
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

