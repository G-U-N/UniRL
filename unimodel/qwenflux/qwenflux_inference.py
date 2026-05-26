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

"""Qwen2.5-VL + FLUX.1-dev joint inference model for text-to-image GRPO.

Mirrors `unimodel.qwenklein.qwenklein_inference.QwenKleinForInferenceLM` but uses
the FLUX.1-dev diffusion expert (CLIP + T5 dual text encoders, embedded
guidance). Intended for GenEval-style prompt-refinement RL where the model
generates an image from a (possibly Qwen-refined) text prompt only.
"""

import os
import re
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers import AutoencoderKL, FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler

from .flux_pipeline import FluxPipelineWithSDE


DEFAULT_FLUX_CKPT = "black-forest-labs/FLUX.1-dev"


class QwenFluxMetaModel:
    def __init__(self, config):
        super(QwenFluxMetaModel, self).__init__(config)

        if hasattr(config, "diffusion_expert"):
            # Build skeleton from configs only; weights are loaded later via
            # AutoModelForCausalLM.from_pretrained(fused-checkpoint).
            transformer_config = FluxTransformer2DModel.load_config(
                DEFAULT_FLUX_CKPT, subfolder="transformer"
            )
            vae_config = AutoencoderKL.load_config(DEFAULT_FLUX_CKPT, subfolder="vae")
            text_encoder_config = AutoConfig.from_pretrained(
                DEFAULT_FLUX_CKPT, subfolder="text_encoder"
            )
            text_encoder_2_config = AutoConfig.from_pretrained(
                DEFAULT_FLUX_CKPT, subfolder="text_encoder_2"
            )

            self.transformer = FluxTransformer2DModel.from_config(transformer_config)
            self.vae = AutoencoderKL.from_config(vae_config)
            self.text_encoder = CLIPTextModel(text_encoder_config)
            self.text_encoder_2 = T5EncoderModel(text_encoder_2_config)
            self.tokenizer = CLIPTokenizer.from_pretrained(
                DEFAULT_FLUX_CKPT, subfolder="tokenizer"
            )
            self.tokenizer_2 = T5TokenizerFast.from_pretrained(
                DEFAULT_FLUX_CKPT, subfolder="tokenizer_2"
            )
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                DEFAULT_FLUX_CKPT, subfolder="scheduler"
            )

            self.diffusion_expert = FluxPipelineWithSDE(
                transformer=self.transformer,
                scheduler=self.scheduler,
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                text_encoder_2=self.text_encoder_2,
                tokenizer_2=self.tokenizer_2,
            )

    def initialize_diffusion_expert(self, fsdp=None):
        if getattr(self, "diffusion_expert", None) is None:
            print("random initiation the diffusion expert !!!")
            self.diffusion_expert = FluxPipelineWithSDE.from_pretrained(
                DEFAULT_FLUX_CKPT, revision="main", torch_dtype=torch.bfloat16
            ).to(torch.bfloat16)
            self.text_encoder = self.diffusion_expert.text_encoder
            self.text_encoder_2 = self.diffusion_expert.text_encoder_2
            self.tokenizer = self.diffusion_expert.tokenizer
            self.tokenizer_2 = self.diffusion_expert.tokenizer_2
            self.vae = self.diffusion_expert.vae
            self.transformer = self.diffusion_expert.transformer
            self.scheduler = self.diffusion_expert.scheduler

            self.config.diffusion_expert = "flux-dev"


class QwenFluxConfig(Qwen2_5_VLConfig):
    model_type = "QwenFlux"


class QwenFluxModel(QwenFluxMetaModel, Qwen2_5_VLModel):
    config_class = QwenFluxConfig

    def __init__(self, config: Qwen2_5_VLConfig):
        super(QwenFluxModel, self).__init__(config)


class QwenFluxForInferenceLM(Qwen2_5_VLForConditionalGeneration):
    config_class = QwenFluxConfig

    def __init__(self, config):
        Qwen2_5_VLForConditionalGeneration.__init__(self, config)
        config.model_type = "QwenFlux"

        self.model = QwenFluxModel(config)
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
        """T2I generation. ``images`` is accepted for API symmetry with the edit
        backbones but is ignored — FLUX.1-dev does not condition on source images.
        """
        if diffusion_kwargs is None:
            diffusion_kwargs = dict(guidance_scale=3.5, num_inference_steps=28)

        if isinstance(texts, str):
            texts = [texts]

        if not sde_sampling:
            output_img = self.model.diffusion_expert(
                prompt=texts,
                max_sequence_length=512,
                **diffusion_kwargs,
            ).images
            return output_img
        else:
            return self.model.diffusion_expert.sde_sampling(
                prompt=texts,
                max_sequence_length=512,
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
        texts: Optional[List[str]] = None,
        processor: Optional[object] = None,
        diffusion_kwargs: Optional[Dict] = None,
        llm_kwargs: Optional[Dict] = None,
        cot_prompt_template: Optional[str] = None,
    ):
        if diffusion_kwargs is None:
            diffusion_kwargs = dict(guidance_scale=3.5, num_inference_steps=28)
        if llm_kwargs is None:
            llm_kwargs = dict(max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True)

        if isinstance(texts, str):
            texts = [texts]
        if cot_prompt_template is None:
            cot_prompt_template = (
                "Please provide an enhanced prompt for the following image generation prompt to "
                "make the image more realistic, detailed, with clear separation and precise alignment "
                "of all entities.\nOriginal prompt: {original_prompt}. "
                "Directly provide the improved prompt in <answer> </answer> tags."
            )

        improved_prompts: List[str] = []
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

        output_images = self.generate_image(texts=improved_prompts, diffusion_kwargs=diffusion_kwargs)

        return {
            "images": output_images,
            "original_prompts": texts,
            "improved_prompts": improved_prompts,
        }


AutoConfig.register("QwenFlux", QwenFluxConfig)
AutoModelForCausalLM.register(QwenFluxConfig, QwenFluxForInferenceLM)


if __name__ == "__main__":
    model = QwenFluxForInferenceLM.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.bfloat16
    )
    model.model.initialize_diffusion_expert()
    model.model.diffusion_expert.to("cuda:0")
    model.to("cuda:0")
    AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    output_dir = os.environ.get("QWENFLUX_SAVE_DIR", "outputs/pretrain/qwenflux")

    try:
        images = model.generate_image(
            texts=["a photo of a red apple on a wooden table, studio lighting"]
        )
        images[0].save("test_flux.jpg")
        print("Sanity T2I saved to test_flux.jpg.")
    except Exception as exc:
        print(f"Sanity T2I failed ({exc}); proceeding to save the fused checkpoint anyway.")

    model.save_pretrained(output_dir)
    print(f"Fused Qwen-Flux checkpoint saved to {output_dir}.")
