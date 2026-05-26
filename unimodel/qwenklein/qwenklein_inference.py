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

"""Qwen2.5-VL + FLUX.2-klein joint inference model.

Mirrors `unimodel.qwenkontext.qwenkontext_inference.QwenKontextForInferenceLM`
but uses the FLUX.2-klein-base diffusion expert (single Qwen3 text encoder,
no CLIP / no T5).
"""

import os
import re
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2TokenizerFast,
    Qwen3ForCausalLM,
)

from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler

from .flux2_klein_pipeline import Flux2KleinPipelineWithSDE


DEFAULT_KLEIN_CKPT = "black-forest-labs/FLUX.2-klein-base-4B"


class QwenKleinMetaModel:
    def __init__(self, config):
        super(QwenKleinMetaModel, self).__init__(config)

        if hasattr(config, "diffusion_expert"):
            # Build skeleton from configs only; weights are loaded later via
            # AutoModelForCausalLM.from_pretrained(fused-checkpoint).
            transformer_config = Flux2Transformer2DModel.load_config(
                DEFAULT_KLEIN_CKPT, subfolder="transformer"
            )
            vae_config = AutoencoderKLFlux2.load_config(DEFAULT_KLEIN_CKPT, subfolder="vae")
            text_encoder_config = AutoConfig.from_pretrained(
                DEFAULT_KLEIN_CKPT, subfolder="text_encoder"
            )

            self.transformer = Flux2Transformer2DModel.from_config(transformer_config)
            self.vae = AutoencoderKLFlux2.from_config(vae_config)
            self.text_encoder = Qwen3ForCausalLM(text_encoder_config)
            self.tokenizer = Qwen2TokenizerFast.from_pretrained(
                DEFAULT_KLEIN_CKPT, subfolder="tokenizer"
            )
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                DEFAULT_KLEIN_CKPT, subfolder="scheduler"
            )

            self.diffusion_expert = Flux2KleinPipelineWithSDE(
                transformer=self.transformer,
                scheduler=self.scheduler,
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
            )

    def initialize_diffusion_expert(self, fsdp=None):
        if getattr(self, "diffusion_expert", None) is None:
            print("random initiation the diffusion expert !!!")
            self.diffusion_expert = Flux2KleinPipelineWithSDE.from_pretrained(
                DEFAULT_KLEIN_CKPT, revision="main", torch_dtype=torch.bfloat16
            ).to(torch.bfloat16)
            self.text_encoder = self.diffusion_expert.text_encoder
            self.tokenizer = self.diffusion_expert.tokenizer
            self.vae = self.diffusion_expert.vae
            self.transformer = self.diffusion_expert.transformer
            self.scheduler = self.diffusion_expert.scheduler

            self.config.diffusion_expert = "flux2-klein"


class QwenKleinConfig(Qwen2_5_VLConfig):
    model_type = "QwenKlein"


class QwenKleinModel(QwenKleinMetaModel, Qwen2_5_VLModel):
    config_class = QwenKleinConfig

    def __init__(self, config: Qwen2_5_VLConfig):
        super(QwenKleinModel, self).__init__(config)


class QwenKleinForInferenceLM(Qwen2_5_VLForConditionalGeneration):
    config_class = QwenKleinConfig

    def __init__(self, config):
        Qwen2_5_VLForConditionalGeneration.__init__(self, config)
        config.model_type = "QwenKlein"

        self.model = QwenKleinModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    @torch.no_grad()
    def generate_image(
        self,
        images: Optional[List[Image.Image]] = None,
        texts: Optional[List[str]] = None,
        diffusion_kwargs: Optional[Dict] = None,
        sde_sampling: Optional[bool] = False,
    ):
        if diffusion_kwargs is None:
            diffusion_kwargs = dict(guidance_scale=4.0, num_inference_steps=28)

        if isinstance(texts, str):
            texts = [texts]

        if not sde_sampling:
            output_img = self.model.diffusion_expert(
                image=images,
                prompt=texts,
                max_sequence_length=128,
                **diffusion_kwargs,
            ).images
            return output_img
        else:
            return self.model.diffusion_expert.sde_sampling(
                image=images,
                prompt=texts,
                max_sequence_length=128,
                **diffusion_kwargs,
            )

    def extract_thinking_content(self, text: str) -> str:
        pattern = r"<answer>(.*?)</answer>"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip().replace("<answer>", "").replace("</answer>", "")
        return text.strip().replace("<answer>", "").replace("</answer>", "")

    @torch.no_grad()
    def generate_image_cot(
        self,
        images: Optional[List[Image.Image]] = None,
        texts: Optional[List[str]] = None,
        processor: Optional[object] = None,
        diffusion_kwargs: Optional[Dict] = None,
        llm_kwargs: Optional[Dict] = None,
        cot_prompt_template: Optional[str] = None,
    ):
        if diffusion_kwargs is None:
            diffusion_kwargs = dict(guidance_scale=4.0, num_inference_steps=28)
        if llm_kwargs is None:
            llm_kwargs = dict(max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True)

        if isinstance(texts, str):
            texts = [texts]
        if cot_prompt_template is None:
            cot_prompt_template = """Please provide an enhanced prompt for the following image editing prompt.
            Ensure the revised prompt is clear, specific, and includes detailed instructions to achieve the desired outcome while maintaining the original intent.
            Original prompt: {original_prompt}. Directly provide the improved prompt in <answer> </answer> tags."""

        improved_prompts: List[str] = []

        # When `images` is None we run the LLM on text-only inputs and skip vision processing.
        if images is None:
            for text in texts:
                cot_input = cot_prompt_template.format(original_prompt=text)
                messages = [{"role": "user", "content": [{"type": "text", "text": cot_input}]}]
                input_text_formatted = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                model_inputs = processor(
                    text=[input_text_formatted], return_tensors="pt"
                ).to(self.device)
                generated_ids = self.generate(
                    **model_inputs,
                    **llm_kwargs,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
                generated_text = processor.batch_decode(
                    generated_ids[:, model_inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
                improved_prompts.extend(
                    [self.extract_thinking_content(decode_text) for decode_text in generated_text]
                )
        else:
            for text, image in zip(texts, images):
                cot_input = cot_prompt_template.format(original_prompt=text)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": cot_input},
                        ],
                    }
                ]
                input_text_formatted = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                model_inputs = processor(
                    images=image_inputs,
                    text=[input_text_formatted],
                    return_tensors="pt",
                ).to(self.device)
                generated_ids = self.generate(
                    **model_inputs,
                    **llm_kwargs,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
                generated_text = processor.batch_decode(
                    generated_ids[:, model_inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
                improved_prompts.extend(
                    [self.extract_thinking_content(decode_text) for decode_text in generated_text]
                )

        output_images = self.generate_image(images, improved_prompts, diffusion_kwargs)

        return {
            "ref_images": images,
            "images": output_images,
            "original_prompts": texts,
            "improved_prompts": improved_prompts,
        }


AutoConfig.register("QwenKlein", QwenKleinConfig)
AutoModelForCausalLM.register(QwenKleinConfig, QwenKleinForInferenceLM)


if __name__ == "__main__":
    model = QwenKleinForInferenceLM.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.bfloat16
    )
    model.model.initialize_diffusion_expert()
    model.model.diffusion_expert.to("cuda:0")
    model.to("cuda:0")
    AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    output_dir = os.environ.get("QWENKLEIN_SAVE_DIR", "outputs/pretrain/qwenklein")

    sanity_image = os.environ.get("QWENKLEIN_SANITY_IMAGE")
    if sanity_image and os.path.isfile(sanity_image):
        try:
            text = ["add a hat to him"]
            ref_image = [Image.open(sanity_image).convert("RGB")]
            images = model.generate_image(ref_image, text)
            images[0].save("test_klein.jpg")
            print(f"Sanity edit saved to test_klein.jpg using {sanity_image}.")
        except Exception as exc:
            print(f"Sanity edit failed ({exc}); proceeding to save the fused checkpoint anyway.")
    else:
        print(
            "Skipping inference sanity check; set QWENKLEIN_SANITY_IMAGE=/path/to/image.jpg "
            "to verify FLUX.2-klein-base inference end-to-end."
        )

    model.save_pretrained(output_dir)
    print(f"Fused Qwen-Klein checkpoint saved to {output_dir}.")
