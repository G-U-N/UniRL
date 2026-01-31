"""
Unified Inference Script for Multi-Modal Image Generation and Editing

Supports three modes:
1. t2i (Text-to-Image): Generate images from text prompts (txt file)
2. geneval: Generate multiple samples per prompt for evaluation (jsonl file)
3. edit: Edit images based on prompts (parquet file)

Example usage:
    # Text-to-Image
    python unified_inference.py --mode t2i --model_path ./model --model_type flux \
        --prompt_file prompts.txt --output_dir outputs/t2i

    # GenEval
    python unified_inference.py --mode geneval --model_path ./model --model_type flux \
        --metadata_file evaluation_metadata.jsonl --output_dir outputs/geneval --n_samples 4

    # Image Editing
    python unified_inference.py --mode edit --model_path ./model --model_type kontext \
        --data_file data.parquet --output_dir outputs/edit
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import os
import traceback
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor
import random
import multiprocessing as mp
import pandas as pd
from io import BytesIO
import base64
from torchvision import transforms as TF

# Model imports
from unimodel.qwenflux.qwenflux_inference import QwenFluxForInferenceLM
from unimodel.qwenkontext.qwenkontext_inference import QwenKontextForInferenceLM

# Global configuration
NUM_DEVICE = 8
NUM_PROCESSES = 8


# =============================================================================
# CoT Prompt Templates
# =============================================================================
COT_PROMPT_TEMPLATES = {
    # General enhancement
    "geneval": """Please provide an enhanced prompt for the following image generation prompt to make the image more realistic, detailed, with clear separation and precise alignment of all entities.
Original prompt: {original_prompt}. Directly provide the improved prompt in <answer> </answer> tags.""",


    "ocr_clarity_v2": """Please enhance the following image generation prompt with specific focus on TEXT clarity and readability.
Original prompt: {original_prompt}. Directly provide the improved prompt in <answer> </answer> tags.""",


    "quality_purev2": """Rewrite the following image generation prompt to improve its visual quality, detail level, realism, and artistic sophistication. 

Original prompt: {original_prompt}

Directly provide the enhanced version directly in <answer></answer> tags.""",


    "edit_general": """Please provide an enhanced prompt for the following image editing prompt. 
Ensure the revised prompt is clear, specific, and includes detailed instructions to achieve the desired outcome while maintaining the original intent. 
Original prompt: {original_prompt}. Directly provide the improved prompt in <answer> </answer> tags.""",

}


# =============================================================================
# Utility Functions
# =============================================================================
def set_global_seed(seed):
    """Set global random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




# =============================================================================
# Model Loading
# =============================================================================
def load_model_pipeline(model_path, model_type, device):
    """Load model pipeline based on model type."""
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    subfolder = model_path.split('/')[-1]
    model_path = model_path.replace(f"/{subfolder}", "")
    if model_type == "flux":
        model = QwenFluxForInferenceLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, subfolder=subfolder
        )
    elif model_type == "sana":
        model = QwenSanaForInferenceLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, subfolder=subfolder
        )
    elif model_type == "sd3":
        model = QwenSD3ForInferenceLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, subfolder=subfolder
        )
    elif model_type == "kontext":
        model = QwenKontextForInferenceLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, subfolder=subfolder
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    processor.tokenizer.padding_side = "left"  # for batch inference
    model.to(device)
    
    return model, processor


# =============================================================================
# Data Loading Functions
# =============================================================================
def load_prompts_from_txt(txt_file):
    """Load prompts from text file (one per line)."""
    with open(txt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def load_prompts_from_jsonl(metadata_file):
    """Load prompts and metadata from JSONL file."""
    with open(metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]
    prompts = [metadata['prompt'].strip() for metadata in metadatas]
    return prompts, metadatas


def load_data_from_parquet(parquet_file):
    """Load images and prompts from parquet file."""
    df = pd.read_parquet(parquet_file)
    
    # Identify column names
    image_col = None
    prompt_col = None
    id_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'image' in col_lower and image_col is None:
            image_col = col
        elif any(kw in col_lower for kw in ['prompt', 'text', 'caption', 'instruction']) and prompt_col is None:
            prompt_col = col
        elif any(kw in col_lower for kw in ['id', 'index']) and id_col is None:
            id_col = col
    
    if image_col is None or prompt_col is None:
        raise ValueError(
            f"Cannot identify columns. Found: {df.columns.tolist()}\n"
            f"Expected 'image' and 'prompt'/'text'/'caption'"
        )
    
    print(f"Using columns - Image: '{image_col}', Prompt: '{prompt_col}', ID: '{id_col}'")
    
    data_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading parquet"):
        try:
            image_data = row[image_col]["bytes"]
            
            if isinstance(image_data, bytes):
                image = Image.open(BytesIO(image_data)).convert('RGB')
            elif isinstance(image_data, str):
                if image_data.startswith('data:image') or image_data.startswith('/9j/') or image_data.startswith('iVBOR'):
                    if 'base64,' in image_data:
                        image_data = image_data.split('base64,')[1]
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(BytesIO(image_bytes)).convert('RGB')
                else:
                    image = Image.open(image_data).convert('RGB')
            else:
                print(f"Warning: Skipping row {idx} - unsupported image format")
                continue
            
            prompt = str(row[prompt_col])
            item_id = row[id_col] if id_col else idx
            
            data_list.append({
                'image': image,
                'prompt': prompt,
                'id': item_id,
                'index': idx
            })
        except Exception as e:
            print(f"Error loading row {idx}: {e}")
            continue
    
    print(f"Loaded {len(data_list)} samples from parquet")
    return data_list


# =============================================================================
# Image Grid Utility
# =============================================================================
def create_image_grid(images, rows, cols):
    """Create a grid image from a list of images."""
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


# =============================================================================
# Generation Functions
# =============================================================================
def generate_t2i_batch(
    prompts, start_idx, pipeline, processor, output_dir, batch_size,
    guidance_scale, num_inference_steps, seed, use_cot, cot_template_name,
    add_instruction, device_id
):
    """Generate images from text prompts (T2I mode)."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"GPU {device_id} T2I"):
        batch_prompts = prompts[i:i + batch_size]
        batch_start_idx = start_idx + i
        original_prompts = batch_prompts.copy()
        
        if add_instruction:
            batch_prompts = [
                f"Please generate image based on the following caption: {p}"
                for p in batch_prompts
            ]
        
        diffusion_kwargs = dict(
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            generator=torch.Generator("cpu").manual_seed(seed)
        )
        
        try:
            with torch.no_grad():
                if use_cot:
                    llm_kwargs = dict(
                        max_new_tokens=256, temperature=0.7, top_p=0.9,
                        do_sample=False, num_return_sequences=1
                    )
                    cot_template = COT_PROMPT_TEMPLATES.get(cot_template_name)
                    outputs = pipeline.generate_image_cot(
                        texts=batch_prompts,
                        diffusion_kwargs=diffusion_kwargs,
                        processor=processor,
                        llm_kwargs=llm_kwargs,
                        cot_prompt_template=cot_template
                    )
                    images = outputs["images"]
                    thinking_prompts = outputs.get("improved_prompts", [])
                else:
                    images = pipeline.generate_image(
                        texts=batch_prompts,
                        diffusion_kwargs=diffusion_kwargs
                    )
                    thinking_prompts = []
            
            for j, img in enumerate(images):
                img_idx = batch_start_idx + j
                base_name = f"{img_idx:05d}"
                
                img.save(os.path.join(output_dir, f"{base_name}.png"))
                
                with open(os.path.join(output_dir, f"{base_name}_caption.txt"), 'w', encoding='utf-8') as f:
                    f.write(original_prompts[j])
                
                if use_cot and j < len(thinking_prompts):
                    with open(os.path.join(output_dir, f"{base_name}_thinking.txt"), 'w', encoding='utf-8') as f:
                        f.write(thinking_prompts[j])
                        
        except Exception as e:
            print(f"Error at batch {batch_start_idx}: {e}")
            traceback.print_exc()


def generate_geneval_batch(
    prompts, metadatas, start_idx, pipeline, processor, output_dir, batch_size,
    guidance_scale, num_inference_steps, seed, n_samples, use_cot,
    cot_template_name, skip_grid, device_id
):
    """Generate multiple samples per prompt for evaluation (GenEval mode)."""
    for prompt_idx, (prompt, metadata) in enumerate(zip(prompts, metadatas)):
        global_idx = start_idx + prompt_idx
        outpath = os.path.join(output_dir, f"{device_id}_{prompt_idx:0>5}")
        os.makedirs(outpath, exist_ok=True)
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)
        
        sample_count = 0
        all_samples = []
        enhanced_prompts = []
        total_batches = (n_samples + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(total_batches), desc=f"GPU {device_id} prompt {prompt_idx}"):
            num_images = min(batch_size, n_samples - sample_count)
            
            diffusion_kwargs = dict(
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images,
                generator=torch.Generator("cpu").manual_seed(seed)
            )
            
            try:
                with torch.inference_mode():
                    if use_cot:
                        llm_kwargs = dict(
                            max_new_tokens=256, temperature=0.7, top_p=0.9,
                            do_sample=False, num_return_sequences=1
                        )
                        cot_template = COT_PROMPT_TEMPLATES.get(cot_template_name)
                        outputs = pipeline.generate_image_cot(
                            texts=prompt,
                            diffusion_kwargs=diffusion_kwargs,
                            processor=processor,
                            llm_kwargs=llm_kwargs,
                            cot_prompt_template=cot_template
                        )
                        images = outputs["images"]
                        enhanced_prompts.extend(outputs.get("improved_prompts", []))
                    else:
                        images = pipeline.generate_image(
                            texts=prompt,
                            diffusion_kwargs=diffusion_kwargs
                        )
                
                for img in images:
                    img.save(os.path.join(sample_path, f"{sample_count:05}.png"))
                    sample_count += 1
                    if not skip_grid:
                        all_samples.append(img)
                        
            except Exception as e:
                print(f"Error at prompt {prompt_idx}, batch {batch_idx}: {e}")
                traceback.print_exc()
        
        # Save enhanced prompts
        with open(os.path.join(outpath, "thinking_prompts.txt"), "w") as fp:
            for ep in enhanced_prompts:
                fp.write(f"{ep}\n")
        
        # Create grid
        if not skip_grid and all_samples:
            rows = int(np.sqrt(n_samples))
            cols = (n_samples + rows - 1) // rows
            if rows * cols >= len(all_samples):
                grid_image = create_image_grid(all_samples[:rows * cols], rows, cols)
                grid_image.save(os.path.join(outpath, "grid.jpg"))


def generate_edit_batch(
    data_batch, start_idx, pipeline, processor, output_dir, batch_size,
    guidance_scale, num_inference_steps, seed, use_cot, cot_template_name,
    device_id, resolution
):
    """Edit images based on prompts (Edit mode)."""
    os.makedirs(output_dir, exist_ok=True)
    
    transform = TF.Compose([
        TF.Resize(resolution),
        TF.CenterCrop(resolution)
    ])
    
    for i in tqdm(range(0, len(data_batch), batch_size), desc=f"GPU {device_id} Edit"):
        batch_data = data_batch[i:i + batch_size]
        batch_start_idx = start_idx + i
        
        batch_images = [transform(item['image']) for item in batch_data]
        batch_prompts = [item['prompt'] for item in batch_data]
        batch_ids = [item['id'] for item in batch_data]
        
        diffusion_kwargs = dict(
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            generator=torch.Generator("cpu").manual_seed(seed),
            max_area=resolution ** 2
        )
        
        try:
            with torch.no_grad():
                if use_cot:
                    llm_kwargs = dict(
                        max_new_tokens=256, temperature=0.7, top_p=0.9,
                        do_sample=False, num_return_sequences=1
                    )
                    cot_template = COT_PROMPT_TEMPLATES.get(cot_template_name)
                    outputs = pipeline.generate_image_cot(
                        images=batch_images,
                        texts=batch_prompts,
                        diffusion_kwargs=diffusion_kwargs,
                        processor=processor,
                        llm_kwargs=llm_kwargs,
                        cot_prompt_template=cot_template
                    )
                    edited_images = outputs["images"]
                    improved_prompts = outputs.get("improved_prompts", [])
                else:
                    edited_images = pipeline.generate_image(
                        images=batch_images,
                        texts=batch_prompts,
                        diffusion_kwargs=diffusion_kwargs
                    )
                    improved_prompts = []
            
            for j, (edited_img, ref_img) in enumerate(zip(edited_images, batch_images)):
                item_id = batch_ids[j]
                base_name = f"{item_id}"
                
                edited_img.save(os.path.join(output_dir, f"{base_name}_edited.png"))
                ref_img.save(os.path.join(output_dir, f"{base_name}_reference.png"))
                
                with open(os.path.join(output_dir, f"{base_name}_prompt.txt"), 'w', encoding='utf-8') as f:
                    f.write(batch_prompts[j])
                
                if use_cot and j < len(improved_prompts):
                    with open(os.path.join(output_dir, f"{base_name}_improved_prompt.txt"), 'w', encoding='utf-8') as f:
                        f.write(improved_prompts[j])
                        
        except Exception as e:
            print(f"Error at batch {batch_start_idx}: {e}")
            traceback.print_exc()


# =============================================================================
# Worker Process
# =============================================================================
def worker_process(
    device_id, mode, data, start_idx, pipeline, processor, output_dir,
    batch_size, guidance_scale, num_inference_steps, seed, use_cot,
    cot_template_name, add_instruction, n_samples, skip_grid, resolution, metadatas=None
):
    """Single GPU worker process."""
    torch.cuda.set_device(f"cuda:{device_id % NUM_DEVICE}")
    
    print(f"GPU {device_id}: Processing {len(data)} items (indices {start_idx} to {start_idx + len(data) - 1})")
    
    if mode == "t2i":
        generate_t2i_batch(
            prompts=data, start_idx=start_idx, pipeline=pipeline,
            processor=processor, output_dir=output_dir, batch_size=batch_size,
            guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
            seed=seed, use_cot=use_cot, cot_template_name=cot_template_name,
            add_instruction=add_instruction, device_id=device_id
        )
    elif mode == "geneval":
        generate_geneval_batch(
            prompts=data, metadatas=metadatas, start_idx=start_idx,
            pipeline=pipeline, processor=processor, output_dir=output_dir,
            batch_size=batch_size, guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps, seed=seed,
            n_samples=n_samples, use_cot=use_cot, cot_template_name=cot_template_name,
            skip_grid=skip_grid, device_id=device_id
        )
    elif mode == "edit":
        generate_edit_batch(
            data_batch=data, start_idx=start_idx, pipeline=pipeline,
            processor=processor, output_dir=output_dir, batch_size=batch_size,
            guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
            seed=seed, use_cot=use_cot, cot_template_name=cot_template_name,
            device_id=device_id, resolution=resolution
        )
    
    print(f"GPU {device_id}: Completed!")


# =============================================================================
# Argument Parser
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Inference Script for Image Generation and Editing"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["t2i", "geneval", "edit"],
        help="Inference mode: t2i (text-to-image), geneval (evaluation), edit (image editing)"
    )
    
    # Input/Output
    parser.add_argument("--prompt_file", type=str, help="Text file with prompts (for t2i mode)")
    parser.add_argument("--metadata_file", type=str, help="JSONL metadata file (for geneval mode)")
    parser.add_argument("--data_file", type=str, help="Parquet file with images and prompts (for edit mode)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    
    # Model configuration
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument(
        "--model_type", type=str, default="flux",
        choices=["flux", "sana", "sd3", "kontext"],
        help="Model type"
    )
    
    # Generation parameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--resolution", type=int, default=1024, help="Image resolution")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="CFG guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=40, help="Inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # CoT options
    parser.add_argument("--use_cot", action="store_true", help="Use Chain of Thought")
    parser.add_argument(
        "--cot_template", type=str, default="general",
        choices=list(COT_PROMPT_TEMPLATES.keys()),
        help="CoT prompt template"
    )
    parser.add_argument("--add_instruction", action="store_true", help="Add instruction prefix (t2i mode)")
    
    # GenEval specific
    parser.add_argument("--n_samples", type=int, default=4, help="Samples per prompt (geneval mode)")
    parser.add_argument("--skip_grid", action="store_true", help="Skip grid image (geneval mode)")
    
    # Hardware
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process")
    
    return parser.parse_args()


# =============================================================================
# Main Function
# =============================================================================
def main():
    mp.set_start_method('spawn', force=True)
    args = parse_args()
    
    global NUM_PROCESSES
    if args.num_gpus is not None:
        NUM_PROCESSES = min(args.num_gpus, NUM_DEVICE)
    
    # Validate mode-specific arguments
    if args.mode == "t2i" and not args.prompt_file:
        raise ValueError("--prompt_file is required for t2i mode")
    if args.mode == "geneval" and not args.metadata_file:
        raise ValueError("--metadata_file is required for geneval mode")
    if args.mode == "edit" and not args.data_file:
        raise ValueError("--data_file is required for edit mode")
    if args.mode == "edit" and args.model_type != "kontext":
        print(f"Warning: edit mode typically uses kontext model, but got {args.model_type}")
    
    # Load data based on mode
    print(f"Mode: {args.mode}")
    metadatas = None
    
    if args.mode == "t2i":
        print(f"Loading prompts from {args.prompt_file}...")
        data = load_prompts_from_txt(args.prompt_file)
    elif args.mode == "geneval":
        print(f"Loading metadata from {args.metadata_file}...")
        data, metadatas = load_prompts_from_jsonl(args.metadata_file)
    elif args.mode == "edit":
        print(f"Loading data from {args.data_file}...")
        data = load_data_from_parquet(args.data_file)
    
    # Apply max_samples limit
    if args.max_samples is not None:
        if args.mode == "geneval":
            data = data[:args.max_samples]
            metadatas = metadatas[:args.max_samples]
        else:
            data = data[:args.max_samples]
        print(f"Limited to {len(data)} samples")
    
    print(f"Total samples: {len(data)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, "config.json")
    config_dict = vars(args).copy()
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Config saved to {config_path}")
    
    # Load models
    print("Loading models...")
    pipelines = []
    processors = []
    
    for i in range(NUM_DEVICE):
        print(f"Loading model {i+1}/{NUM_DEVICE} on cuda:{i % NUM_DEVICE}...")
        pipeline, processor = load_model_pipeline(
            args.model_path, args.model_type, f"cuda:{i % NUM_DEVICE}"
        )
        pipelines.append(pipeline)
        processors.append(processor)
    
    print("All models loaded!")
    
    # Distribute data across GPUs
    samples_per_gpu = len(data) // NUM_PROCESSES
    
    with ThreadPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        futures = []
        
        for device_id in range(NUM_PROCESSES):
            start_idx = device_id * samples_per_gpu
            end_idx = len(data) if device_id == NUM_PROCESSES - 1 else start_idx + samples_per_gpu
            
            gpu_data = data[start_idx:end_idx]
            gpu_metadatas = metadatas[start_idx:end_idx] if metadatas else None
            
            future = executor.submit(
                worker_process,
                device_id=device_id,
                mode=args.mode,
                data=gpu_data,
                start_idx=start_idx,
                pipeline=pipelines[device_id % NUM_DEVICE],
                processor=processors[device_id % NUM_DEVICE],
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
                use_cot=args.use_cot,
                cot_template_name=args.cot_template,
                add_instruction=args.add_instruction,
                n_samples=args.n_samples,
                skip_grid=args.skip_grid,
                resolution=args.resolution,
                metadatas=gpu_metadatas
            )
            futures.append(future)
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Worker failed: {e}")
                traceback.print_exc()
    
    print(f"\n✓ Done! Results saved to {args.output_dir}")
    print(f"  Total processed: {len(data)}")


if __name__ == "__main__":
    main()