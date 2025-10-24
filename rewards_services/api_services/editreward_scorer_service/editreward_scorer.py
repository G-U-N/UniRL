import os
import torch
import tempfile
import shutil
from PIL import Image
import sys
sys.path.append("./EditReward")

from EditReward import EditRewardInferencer

class EditRewardScorer(torch.nn.Module):
    def __init__(self, config_path, checkpoint_path, dtype, device="cuda"):
        super().__init__()
        self.inferencer = EditRewardInferencer(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
            reward_dim="overall_detail",
            rm_head_type="ranknet_multi_head"
        )
        self.dtype = dtype
        self.device = device
        self.eval()

    @torch.no_grad()
    def __call__(self, prompts, image_src, image_paths):
        if not image_src or not image_paths:
            raise ValueError("Missing source or edited images")
        if len(image_src) != len(image_paths) or len(image_src) != len(prompts):
            raise ValueError("Mismatched number of source images, edited images, and prompts")

        # Create temporary directory for saving images
        temp_dir = tempfile.mkdtemp()
        scores = []

        try:
            # Process each image pair one at a time
            for i, (src_img, edit_img, prompt) in enumerate(zip(image_src, image_paths, prompts)):
                if not isinstance(src_img, Image.Image) or not isinstance(edit_img, Image.Image):
                    raise ValueError(f"Image pair {i} contains non-PIL Image objects")

                # Save PIL Images to temporary files
                src_path = os.path.join(temp_dir, f"source_{i}.jpg")
                edit_path = os.path.join(temp_dir, f"edited_{i}.jpg")
                
                src_img.save(src_path, "JPEG")
                edit_img.save(edit_path, "JPEG")

                # Run inference for a single pair
                rewards = self.inferencer.reward(
                    prompts=[prompt],
                    image_src=[src_path],
                    image_paths=[edit_path]
                )
                scores.append(rewards[0][0].item())

            return scores

        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)