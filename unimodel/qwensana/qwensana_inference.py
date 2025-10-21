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


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLModel, Qwen2_5_VLForConditionalGeneration, T5Config, Gemma2Model, GemmaTokenizer, GemmaTokenizerFast, Gemma2Config, AutoConfig
from diffusers import SanaPipeline, AutoencoderDC, FlowMatchEulerDiscreteScheduler, SanaTransformer2DModel, DPMSolverMultistepScheduler
import re
import datetime
import os


def save_grid_image(prompt, images, rows, cols):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("samples", timestamp, prompt[:100])
    os.makedirs(base_dir, exist_ok=True)
    
    filename = os.path.join(base_dir, "grid.jpg")
    grid_image = create_image_grid(images, rows, cols)
    grid_image.save(filename)
    
    print(f"Saved: {filename}")

def create_image_grid(images, rows, cols):
    """Creates a grid of images and returns a single PIL Image."""
    assert len(images) == rows * cols

    width, height = images[0].size
    grid_width = width * cols
    grid_height = height * rows

    grid_image = Image.new('RGB', (grid_width, grid_height))

    for i, image in enumerate(images):
        x = (i % cols) * width
        y = (i // cols) * height
        grid_image.paste(image, (x, y))

    return grid_image


class QwenSanaMetaModel:

    def __init__(self, config):
        super(QwenSanaMetaModel, self).__init__(config)
        if hasattr(config, "diffusion_expert"):
            ckpt_id = "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"

            # Load configuration for each component
            transformer_config = SanaTransformer2DModel.load_config(ckpt_id, subfolder="transformer")
            vae_config = AutoencoderDC.load_config(ckpt_id, subfolder="vae")
            text_encoder_config = Gemma2Config.from_pretrained(ckpt_id, subfolder="text_encoder")
            scheduler_config = DPMSolverMultistepScheduler.load_config(ckpt_id, subfolder="scheduler")
            # Initialize components from their configurations
            self.transformer = SanaTransformer2DModel.from_config(transformer_config)
            self.vae = AutoencoderDC.from_config(vae_config)
            self.text_encoder = Gemma2Model(text_encoder_config)
            self.scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)

            # Initialize tokenizer
            self.tokenizer = GemmaTokenizerFast.from_pretrained(ckpt_id, subfolder="tokenizer")

            # Create the pipeline configuration dictionary
            pipeline_config = {
                "transformer": self.transformer,
                "scheduler": self.scheduler,
                "vae": self.vae,
                "text_encoder": self.text_encoder,
                "tokenizer": self.tokenizer,
            }

            self.diffusion_expert = SanaPipeline(**pipeline_config)

    def initialize_diffusion_expert(self, fsdp=None):
        
        if getattr(self, 'diffusion_expert', None) is None:
            print("Random initiation the Sana diffusion expert !!!")
            self.diffusion_expert = SanaPipeline.from_pretrained(
                "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
                torch_dtype=torch.bfloat16
            )
            
            # Store references to components for easier access
            self.transformer = self.diffusion_expert.transformer
            self.vae = self.diffusion_expert.vae
            self.text_encoder = self.diffusion_expert.text_encoder
            self.tokenizer = self.diffusion_expert.tokenizer
            self.scheduler = self.diffusion_expert.scheduler
            
            self.config.diffusion_expert = "Sana"


class QwenSanaConfig(Qwen2_5_VLConfig):
    model_type = "QwenSana"


class QwenSanaModel(QwenSanaMetaModel, Qwen2_5_VLModel):
    config_class = QwenSanaConfig

    def __init__(self, config: Qwen2_5_VLConfig):
        super(QwenSanaModel, self).__init__(config)


class QwenSanaForInferenceLM(Qwen2_5_VLForConditionalGeneration):
    config_class = QwenSanaConfig

    def __init__(self, config):
        Qwen2_5_VLForConditionalGeneration.__init__(self, config)
        config.model_type = "QwenSana"

        self.model = QwenSanaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    @torch.no_grad()
    def generate_image(
        self,
        texts: List[str],
        diffusion_kwargs: Optional[Dict] = None,
    ):  
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Default parameters for Sana
        default_kwargs = dict(
            guidance_scale=3.5, 
            num_inference_steps=20,
            height=1024,
            width=1024
        )
        
        if diffusion_kwargs:
            default_kwargs.update(diffusion_kwargs)

        output_img = self.model.diffusion_expert(
            texts,
            **default_kwargs,
        ).images        
        
        return output_img

    def extract_thinking_content(self, text: str) -> str:
        pattern = r'<answer>(.*?)</answer>'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            return matches[-1].strip().replace("<answer>", "").replace("</answer>", "")
        else:
            return text.strip().replace("<answer>", "").replace("</answer>", "")

    @torch.no_grad()
    def generate_image_cot(
        self,
        texts: List[str],
        processor: Optional[object] = None,
        diffusion_kwargs: Optional[Dict] = None,
        llm_kwargs: Optional[Dict] = None,
        cot_prompt_template: Optional[str] = None,
    ):
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Default parameters
        default_diffusion_kwargs = dict(
            guidance_scale=5.0, 
            num_inference_steps=20,
            height=1024,
            width=1024
        )
        if diffusion_kwargs:
            default_diffusion_kwargs.update(diffusion_kwargs)
        
        default_llm_kwargs = dict(
            max_new_tokens=256, 
            temperature=0.7, 
            top_p=0.9, 
            do_sample=True
        )
        if llm_kwargs:
            default_llm_kwargs.update(llm_kwargs)
            
        if cot_prompt_template is None:
            cot_prompt_template = """Please provide an enhanced prompt for the following image generation prompt to make the image more realistic, detailed, with clear separation and precise alignment of all entities.
            Original prompt: {original_prompt}. Directly provide the improved prompt in <answer> </answer> tags."""

        improved_prompts = []
        
        for text in texts:
            cot_input = cot_prompt_template.format(original_prompt=text)
            
            messages = [{"role": "user", "content": cot_input}]
            input_text_formatted = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = processor(
                text=[input_text_formatted], 
                return_tensors="pt"
            ).to(self.device)

            generated_ids = self.generate(
                **model_inputs,
                **default_llm_kwargs,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id
            )
            
            generated_text = processor.batch_decode(
                generated_ids[:, model_inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            improved_prompt = [self.extract_thinking_content(decode_text) for decode_text in generated_text]
            improved_prompts.extend(improved_prompt)
            
            print(f"Original prompt: {text}")
            print(f"Improved prompt: {improved_prompt}")
            print("-" * 50)

        output_images = self.generate_image(improved_prompts, default_diffusion_kwargs)
        
        return {
            'images': output_images,
            'original_prompts': texts,
            'improved_prompts': improved_prompts
        }


AutoConfig.register("QwenSana", QwenSanaConfig)
AutoModelForCausalLM.register(QwenSanaConfig, QwenSanaForInferenceLM)


if __name__ == "__main__":
    model = QwenSanaForInferenceLM.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", 
        torch_dtype=torch.bfloat16
    )
    model.model.initialize_diffusion_expert()
    model.model.diffusion_expert.to("cuda:0")
    model.to("cuda:0")
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    
    # Test basic image generation
    text = ["a photo of a cat"]
    diffusion_kwargs = dict(
        guidance_scale=3.5, 
        num_inference_steps=20, 
        width=1024, 
        height=1024, 
        generator=torch.manual_seed(0)
    )
    
    images = model.generate_image(text, diffusion_kwargs=diffusion_kwargs)
    images[0].save("test_Sana.jpg")
    
    # Test chain-of-thought image generation
    outputs = model.generate_image_cot(text, processor=processor, diffusion_kwargs=diffusion_kwargs)
    outputs['images'][0].save("test_Sana_cot.jpg")
    
    # Save the model
    model.save_pretrained("qwenSana-1.5")
    
    # print("Sana model integration completed successfully!")
    
    # model = QwenSanaForInferenceLM.from_pretrained(
    # "qwenSana-1.5", 
    # torch_dtype=torch.bfloat16
    # ).to("cuda")
    
    
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    
    # # Test basic image generation
    # text = ["a photo of a cat"]
    # diffusion_kwargs = dict(
    #     guidance_scale=5.0, 
    #     num_inference_steps=20, 
    #     width=1024, 
    #     height=1024, 
    #     generator=torch.manual_seed(0)
    # )
    
    # images = model.generate_image(text, diffusion_kwargs=diffusion_kwargs)
    # images[0].save("test_Sana.jpg")
    
    # # Test chain-of-thought image generation
    # outputs = model.generate_image_cot(text, processor=processor, diffusion_kwargs=diffusion_kwargs)
    # outputs['images'][0].save("test_Sana_cot.jpg")
    
