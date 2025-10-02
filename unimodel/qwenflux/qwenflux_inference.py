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
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLModel, Qwen2_5_VLForConditionalGeneration


from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import numpy_to_pil
import numpy as np
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteSchedulerOutput
from diffusers.schedulers import DPMSolverMultistepScheduler
import math
from diffusers.utils.torch_utils import randn_tensor
from diffusers import FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast, CLIPTextConfig, T5Config
from .fluxpipeline import FluxPipeline
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


def sde_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
    process from the learned model outputs (most often the predicted velocity).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned flow model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
    """
    step_index = [self.index_for_timestep(t) for t in timestep]
    prev_step_index = [step+1 for step in step_index]
    sigma = self.sigmas[step_index].view(-1, 1, 1).to(model_output.device)
    sigma_prev = self.sigmas[prev_step_index].view(-1, 1, 1).to(model_output.device)
    sigma_max = self.sigmas[1].item()
    dt = sigma_prev - sigma

    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))*1.0
    
    
    # our sde
    prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt)+model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt
    
    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*dt) * variance_noise


    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt))**2))
        - torch.log(std_dev_t * torch.sqrt(-1*dt))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    
    return prev_sample, log_prob, prev_sample_mean, std_dev_t * torch.sqrt(-1*dt)


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

def sde_step_with_logprob_simple(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
    process from the learned model outputs (most often the predicted velocity).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned flow model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
    """
    
    step_index = [self.index_for_timestep(t) for t in timestep]
    prev_step_index = [step+1 for step in step_index]
    sigma = self.sigmas[step_index].view(-1, 1, 1, 1).to(model_output.device)
    sigma_prev = self.sigmas[prev_step_index].view(-1, 1, 1, 1).to(model_output.device)
    sigma_max = self.sigmas[1].item()
    dt = sigma_prev - sigma
    
    
    eta = 0.5
    Dt = - dt * eta
    
    prev_sample_mean = sample * (1 - Dt / (1 - torch.where(sigma == 1, sigma_max, sigma))) + model_output * (dt - Dt)
    
    std_dev_t = torch.sqrt(2 * Dt * (sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))))
    
    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    if prev_sample is None:
        # Generate noise if not provided
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
            
        prev_sample = prev_sample_mean + std_dev_t * variance_noise


    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    
    return prev_sample, log_prob, prev_sample_mean, std_dev_t 

class QwenFluxMetaModel:

    def __init__(self, config):
        super(QwenFluxMetaModel, self).__init__(config)

        if hasattr(config, "diffusion_expert"):
            ckpt_id = "black-forest-labs/FLUX.1-dev"
            # Load configuration for each component
            transformer_config = FluxTransformer2DModel.load_config(ckpt_id, subfolder="transformer")
            vae_config = AutoencoderKL.load_config(ckpt_id, subfolder="vae")
            text_encoder_config = CLIPTextConfig.from_pretrained(ckpt_id, subfolder="text_encoder")
            text_encoder_2_config = T5Config.from_pretrained(ckpt_id, subfolder="text_encoder_2")

            # Initialize components from their configurations
            self.transformer = FluxTransformer2DModel.from_config(transformer_config)
            self.vae = AutoencoderKL.from_config(vae_config)
            self.text_encoder = CLIPTextModel(text_encoder_config)
            self.text_encoder_2 = T5EncoderModel(text_encoder_2_config)

            # Initialize tokenizers (these don't use from_config as they are not models)
            self.tokenizer = CLIPTokenizer.from_pretrained(ckpt_id, subfolder="tokenizer")
            self.tokenizer_2 = T5TokenizerFast.from_pretrained(ckpt_id, subfolder="tokenizer_2")

            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(ckpt_id, subfolder="scheduler")

            # Create the pipeline configuration dictionary
            pipeline_config = {
                "transformer": self.transformer,
                "scheduler": self.scheduler,
                "vae": self.vae,
                "text_encoder": self.text_encoder,
                "text_encoder_2": self.text_encoder_2,
                "tokenizer": self.tokenizer,
                "tokenizer_2": self.tokenizer_2,
            }

            self.diffusion_expert = FluxPipeline(**pipeline_config)


    def initialize_diffusion_expert(self, fsdp=None):
        
        if getattr(self, 'diffusion_expert', None) is None:
            print("random initiation the diffusion expert !!!")
            self.diffusion_expert = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", revision="main", torch_dtype=torch.bfloat16).to(torch.bfloat16)
            self.text_encoder = self.diffusion_expert.text_encoder
            self.text_encoder_2 = self.diffusion_expert.text_encoder_2
            self.tokenizer = self.diffusion_expert.tokenizer
            self.tokenizer_2 = self.diffusion_expert.tokenizer_2
            self.vae = self.diffusion_expert.vae
            self.transformer = self.diffusion_expert.transformer
            self.scheduler = self.diffusion_expert.scheduler
            
            self.config.diffusion_expert =  "flux"
        
            

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
        texts: List[str],
        diffusion_kwargs: Optional[Dict] = dict(guidance_scale = 3.5, num_inference_steps=25),
        sde_sampling: Optional[bool] = False,
    ):  
        
        if isinstance(texts, str):
            texts = [texts]

        if not sde_sampling:
            output_img = self.model.diffusion_expert(
                texts,
                max_sequence_length=512,            
                **diffusion_kwargs,
            ).images        
            return output_img
        else:
            return self.model.diffusion_expert.sde_sampling(
                texts,
                max_sequence_length=512,            
                **diffusion_kwargs,
            )
    

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
        diffusion_kwargs: Optional[Dict] = dict(guidance_scale = 3.5, num_inference_steps=25),
        llm_kwargs: Optional[Dict] = dict(max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True),
        cot_prompt_template: Optional[str] = None,
    ):
        
        if isinstance(texts, str):
            texts = [texts]
            
        if cot_prompt_template is None:
            # cot_prompt_template = """Please improve the following image generation prompt to make it more detailed and specific for better image quality. Think step by step about what visual elements would make this image more compelling. Original prompt: {original_prompt}. Please provide the improved prompt in <thinking> </thinking> tags."""
            cot_prompt_template = """Please provide an enhanced prompt for the following image generation prompt to make the image more realistic, detailed, with clear separation and precise alignment of all entities.
            Original prompt: {original_prompt}.  Directly provide the improved prompt in <answer> </answer> tags."""

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
                **llm_kwargs,
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

        output_images = self.generate_image(improved_prompts, diffusion_kwargs)
        
        return {
            'images': output_images,
            'original_prompts': texts,
            'improved_prompts': improved_prompts
        }

AutoConfig.register("QwenFlux", QwenFluxConfig)
AutoModelForCausalLM.register(QwenFluxConfig, QwenFluxForInferenceLM)


if __name__ == "__main__":
    
    model = QwenFluxForInferenceLM.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct",torch_dtype=torch.bfloat16)
    model.model.initialize_diffusion_expert()
    model.model.diffusion_expert.to("cuda:0")
    model.to("cuda:0")
    AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    text = ["a photo of a cat"]
    images = model.generate_image(text)
    images[0].save("test_flux.png")
    
    model.save_pretrained("qwenflux-test")
    
    
    model = QwenFluxForInferenceLM.from_pretrained("qwenflux-test", torch_dtype=torch.bfloat16)
    model.to("cuda:0")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")    
    text = ["a photo of a cat"]
    images = model.generate_image(text)
    images[0].save("test_flux.jpg")
    
    outputs = model.generate_image_cot(text, processor = processor)
    outputs['images'][0].save("test_flux_cot.jpg")



