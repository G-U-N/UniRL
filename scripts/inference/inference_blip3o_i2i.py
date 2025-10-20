import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModel
import torch
import sys
import argparse
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from unimodel.blip3o.constants import *
from unimodel.blip3o.conversation import conv_templates
from unimodel.blip3o.model.builder import load_pretrained_model
from unimodel.blip3o.utils import disable_torch_init
from transformers import AutoProcessor
import torchvision.transforms as T
import re, random
from torchvision.transforms.functional import InterpolationMode

model_path = "outputs/pretrain/blip3o/checkpoint-pretrain"
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

device_1 = 0
n_und_query = 576

disable_torch_init()

tokenizer, multi_model, context_len = load_pretrained_model(model_path)

img_tokens = "<|vision_start|>" + "<|image_pad|>" * n_und_query + "<|vision_end|>"

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

def save_grid_image(prompt, images, rows, cols, output_dir):
    """Saves a grid of images to a timestamped folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = re.sub(r'[^\w\s-]', '', prompt[:100]).replace(' ', '_')
    base_dir = os.path.join(output_dir, timestamp, safe_prompt)
    os.makedirs(base_dir, exist_ok=True)
    filename = os.path.join(base_dir, "grid.jpg")
    grid_image = create_image_grid(images, rows, cols)
    grid_image.save(filename)
    print(f"Saved grid: {filename}")
    return filename

def add_template(prompt):
    """Formats the prompt using the conversation template."""
    conv = conv_templates['qwen'].copy()
    conv.append_message(conv.roles[0], prompt[0])
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return [prompt]

def set_global_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Define prompts for image editing
prompts = [
    "change the setting to spring with blooming trees",
    "change the background to red",
    "put the cat in the beach",
    "add a red hat to it",
    "add sunglasses to the animal",
    "change the cat to a dog, specifically change the cat's head and body to a dog's head and body, but keep the background intact",
    "remove the cat in the photo",
    "add a red scarf to the cat in the photo",
    "make the photo Van Gogh style",
]

# Create output directory
output_dir = Path("outputs/edited_samples/")
output_dir.mkdir(parents=True, exist_ok=True)

# Load reference image
ref_image = Image.open("assets/images/cat.jpg").convert("RGB")
und_trsf = T.Compose([T.Resize(672, interpolation=InterpolationMode.BICUBIC, antialias=True), T.CenterCrop(672)])
# Prepare image inputs
image_inputs = processor.image_processor(images=[und_trsf(ref_image)], return_tensors="pt")
image_grid_thw = image_inputs.image_grid_thw.to(multi_model.get_model().device)
pixel_values = image_inputs.pixel_values.to(multi_model.get_model().device)
trsf = T.Compose([
    T.Resize(size=1024, interpolation=InterpolationMode.BICUBIC, antialias=True),
    T.CenterCrop(size=1024),
    T.ToTensor(),
    T.Lambda(lambda x: x * 2 - 1)
])
ref_latents = multi_model.get_model().gen_vision_tower(trsf(ref_image).unsqueeze(0).to(multi_model.get_model().device))

diffusion_kwargs = {
    "num_inference_steps": 30,
    "guidance_scale": 3.5,
    "guidance_scale_ref": 1.2, 
}

# Process each prompt
for idx, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
    set_global_seed(seed=42)
    gen_images = []
    
    # Prepare text input with image tokens
    text = add_template([f"{img_tokens}\nPlease edit the given image based on the following instruction: {prompt}."])
    
    gen_img = multi_model.generate_image(
        text=text,
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        ref_latents=ref_latents,
        image_grid_thw=image_grid_thw,
        diffusion_kwargs=diffusion_kwargs
    )
    
    # Save individual image
    safe_prompt = re.sub(r'[^\w\s-]', '', prompt[:100]).replace(' ', '_')
    individual_filename = output_dir / f"image_{idx}_{safe_prompt}.jpg"
    gen_img[0].save(individual_filename)
    print(f"Saved individual: {individual_filename}")
    
    print(f"Finished processing prompt: {prompt}")

# # Optional: Enhance prompts and generate images
# for idx, prompt in enumerate(tqdm(prompts, desc="Processing enhanced prompts")):
#     set_global_seed(seed=42)
    
#     # Generate enhanced prompt
#     prompts_text = add_template([f"Please enhance the following prompt to make it more suitable for image editing: {prompt}. Output the final answer (entailed prompt) in <answer> </answer> tags."])
#     prompt_inputs = tokenizer(
#         text=prompts_text,
#         images=und_trsf(ref_image)
#         return_tensors="pt",
#         padding=True,
#         padding_side="left",
#         add_special_tokens=False,
#     )
#     generation_config = {
#         "max_new_tokens": 200,
#         "do_sample": False,
#         "temperature": 0.7,
#         "top_p": 0.9,
#         "num_return_sequences": 1,
#         "pad_token_id": tokenizer.pad_token_id,
#         "eos_token_id": tokenizer.eos_token_id,
#     }
#     with torch.no_grad():
#         prompt_inputs = {k: v.to(device_1) for k, v in prompt_inputs.items()}
#         prompt_completion_ids = multi_model.generate(**prompt_inputs, **generation_config)
    
#     prompt_completion = tokenizer.batch_decode(prompt_completion_ids, skip_special_tokens=True)[0]
    
#     # Save enhanced prompt
#     safe_prompt = re.sub(r'[^\w\s-]', '', prompt[:100]).replace(' ', '_')
#     with open(output_dir / f"completion_{idx}_{safe_prompt}.txt", "w") as f:
#         f.write(prompt_completion)
    
#     # Generate images with enhanced prompt
#     text = add_template([f"{img_tokens}\n{prompt_completion}"])
#     gen_img = multi_model.generate_image(
#         text=text,
#         tokenizer=tokenizer,
#         pixel_values=pixel_values,
#         ref_latents=ref_latents,
#         image_grid_thw=image_grid_thw,
#         diffusion_kwargs=diffusion_kwargs
#     )
            
#     individual_filename = output_dir / f"enhanced_image_{idx}_{safe_prompt}.jpg"
#     gen_img[0].save(individual_filename)
#     print(f"Saved enhanced individual: {individual_filename}")
    
#     print(f"Finished processing enhanced prompt: {prompt}")