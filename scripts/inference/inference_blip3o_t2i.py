import numpy as np
from PIL import Image
from transformers import AutoProcessor 
import torch
from tqdm import tqdm
from unimodel.blip3o.constants import *
from unimodel.blip3o.conversation import conv_templates
from unimodel.blip3o.model.builder import load_pretrained_model
from unimodel.blip3o.utils import disable_torch_init
from transformers import AutoProcessor
from datetime import datetime 
from torchvision import transforms as T
import os
from pathlib import Path

import re, random

model_path = "outputs/pretrain/blip3o/checkpoint-pretrain"


processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")


device_1 = 0


disable_torch_init()

tokenizer, multi_model, context_len = load_pretrained_model(model_path)



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


def add_template(prompt):
   conv = conv_templates['qwen'].copy()
   conv.append_message(conv.roles[0], prompt[0])
   conv.append_message(conv.roles[1], None)
   prompt = conv.get_prompt()
   return [prompt]



def set_global_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



# Create outputs_test folder if it doesn't exist
output_dir = Path("outputs/samples/")
output_dir.mkdir(parents=True, exist_ok=True)
prompts = [
    "a photo of a zebra and a bed",
    "a photo of an oven and a bed",
    "a photo of a baseball bat and a fork",
    "a photo of a vase and a spoon",
    "a photo of a skateboard and a sink",
    "a photo of a pizza and a bench",
    
    "a photo of a black hot dog",
    "a photo of a red scissors",
    "a photo of a white teddy bear",
    "a photo of a black skis",
    "a photo of a blue dining table",
    "a photo of a black refrigerator",
    
    "a photo of an orange potted plant and a black spoon",
    "a photo of a green tennis racket and a black dog",
    "a photo of a yellow handbag and a blue refrigerator",
    "a photo of a pink broccoli and a red sink",
    "a photo of a red bowl and a pink sink",
    "a photo of a white toilet and a red apple",
    "a photo of a pink dining table and a black sandwich",
    "a photo of a black car and a green parking meter",
    "a photo of a yellow bird and a black motorcycle",
    "a photo of a brown giraffe and a white stop sign",
    "a photo of a white banana and a black elephant",
    
    "a photo of four computer keyboards",
    "a photo of three sinks",
    "a photo of two ovens",
    "a photo of two toilets",
    "a photo of two bicycles",
    "a photo of two trains",
    "a photo of three oranges",
]

diffusion_kwargs = dict(num_inference_steps=50, guidance_scale=3.5)

for idx, prompt in enumerate(prompts):
    set_global_seed(seed=42)
    gen_images = []
    for i in range(1):
        gen_img = multi_model.generate_image(text=add_template([f"Please generate image based on the following caption: {prompt}"]), tokenizer=tokenizer, diffusion_kwargs=diffusion_kwargs)
        gen_images.append(gen_img[0])
    gen_img[0].save(f"{output_dir}/image_{idx}_wo.jpg")

# thinking to generate enhanced prompts first
# then use the enhanced prompts to generate images
for idx, prompt in enumerate(prompts):
    set_global_seed(seed=42)
    for i in range(1):
        prompts_text = add_template([f"Please enhance the following prompt to make it more suitable for image generation: {prompt}. Output the final answer (entailed prompt) in <answer> </answer> tags."])
        
        prompt_inputs = tokenizer(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        generation_config = {
            "max_new_tokens": 200,
            "do_sample": False,
            "temperature": 0.7,
            "top_p": 0.9,
            "num_return_sequences": 1,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        with torch.no_grad():
            prompt_inputs = {k: v.to(device_1) for k, v in prompt_inputs.items()}
            prompt_completion_ids = multi_model.generate(**prompt_inputs, **generation_config)
        
        prompt_completion = tokenizer.batch_decode(prompt_completion_ids, skip_special_tokens=True)[0]
        
        with open(output_dir / f"completion_{idx}.txt", "w") as f:
            f.write(prompt_completion)
        
        gen_img = multi_model.generate_image(text=prompt_completion, tokenizer=tokenizer, diffusion_kwargs=diffusion_kwargs)
        gen_img[0].save(f"{output_dir}/image_{idx}.jpg")
    