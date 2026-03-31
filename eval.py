#!/usr/bin/env python3
"""Batch image evaluation tool with YAML configuration."""

import requests
import pickle
from PIL import Image
from typing import List, Dict, Any, Union, Optional, Tuple
import sys
import os
import json
import yaml
from io import BytesIO
from tqdm import tqdm
from datetime import datetime


PAIR_SCORERS = {"editreward"}
CAPTION_SUFFIXES = ["_caption.txt", "_prompt.txt"]


class RewardEvaluatorClient:
    def __init__(self, scorer_urls: Dict[str, str]):
        self.scorer_urls = scorer_urls

    def evaluate(self, 
                 model_name: str, 
                 images: Union[List[Image.Image], Dict[str, List[Image.Image]]], 
                 prompts: List[str], 
                 metadata: Dict[str, Any] = None) -> Union[List[float], Dict[str, Any]]:
        url = self.scorer_urls.get(model_name)
        if not url:
            raise ValueError(f"Reward model '{model_name}' URL not configured.")

        payload_bytes = create_payload(images, prompts, metadata)

        try:
            response = requests.post(url, data=payload_bytes, timeout=600)
            response.raise_for_status()
            result = parse_response(response.content)
            
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(f"Scorer '{model_name}' returned error: {result['error']}")
            
            return result

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"HTTP request to '{model_name}' failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to process response from '{model_name}': {e}")


def serialize_images(images: List[Image.Image]) -> List[bytes]:
    images_bytes = []
    for img in images:
        img_byte_arr = BytesIO()
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(img_byte_arr, format="JPEG")
        images_bytes.append(img_byte_arr.getvalue())
    return images_bytes


def create_payload(images: Union[List[Image.Image], Dict[str, List[Image.Image]]], 
                   prompts: List[str], 
                   metadata: Dict[str, Any] = None) -> bytes:
    if isinstance(images, dict):
        serialized_images = {key: serialize_images(value) for key, value in images.items()}
    else:
        serialized_images = serialize_images(images)
    
    return pickle.dumps({
        "images": serialized_images,
        "prompts": prompts,
        "metadata": metadata or {}
    })


def parse_response(response_content: bytes) -> Union[List[float], Dict[str, Any]]:
    return pickle.loads(response_content)


def find_caption_file(base_path: str, base_name: str) -> Optional[str]:
    for suffix in CAPTION_SUFFIXES:
        caption_path = os.path.join(base_path, f"{base_name}{suffix}")
        if os.path.exists(caption_path):
            return caption_path
    return None


def collect_standard_samples(folder_path: str) -> Tuple[List[Image.Image], List[str], List[str]]:
    images, prompts, filenames = [], [], []
    
    for file in sorted(os.listdir(folder_path)):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        if any(suffix in file for suffix in ['_edited', '_reference', '_source']):
            continue
            
        base_name = os.path.splitext(file)[0]
        img_path = os.path.join(folder_path, file)
        caption_path = find_caption_file(folder_path, base_name)
        
        if not caption_path:
            continue
        
        try:
            img = Image.open(img_path)
            with open(caption_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            images.append(img)
            prompts.append(prompt)
            filenames.append(file)
        except Exception as e:
            print(f"  Warning: Failed to process {file}: {e}")
    
    return images, prompts, filenames


def collect_edit_samples(folder_path: str) -> Tuple[Dict[str, List[Image.Image]], List[str], List[str]]:
    source_images, edited_images, prompts, filenames = [], [], [], []
    
    edited_files = [f for f in os.listdir(folder_path) if f.endswith('_edited.png')]
    
    for edited_file in sorted(edited_files):
        base_name = edited_file.replace('_edited.png', '')
        source_file = f"{base_name}_reference.png"
        
        if not os.path.exists(os.path.join(folder_path, source_file)):
            source_file = f"{base_name}_source.png"
        
        source_path = os.path.join(folder_path, source_file)
        edited_path = os.path.join(folder_path, edited_file)
        caption_path = find_caption_file(folder_path, base_name)
        
        if not os.path.exists(source_path) or not caption_path:
            continue
        
        try:
            source_img = Image.open(source_path)
            edited_img = Image.open(edited_path)
            with open(caption_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            
            source_images.append(source_img)
            edited_images.append(edited_img)
            prompts.append(prompt)
            filenames.append(base_name)
        except Exception as e:
            print(f"  Warning: Failed to process {base_name}: {e}")
    
    return {'source': source_images, 'edited': edited_images}, prompts, filenames


def evaluate_folder(folder_path: str, 
                    model_name: str, 
                    batch_size: int,
                    scorer_urls: Dict[str, str],
                    verbose: bool = True) -> Optional[Dict[str, Any]]:
    if not os.path.isdir(folder_path):
        return None
    
    evaluator = RewardEvaluatorClient(scorer_urls)
    is_pair_scorer = model_name in PAIR_SCORERS
    
    if is_pair_scorer:
        images, prompts, filenames = collect_edit_samples(folder_path)
        sample_count = len(prompts)
    else:
        images, prompts, filenames = collect_standard_samples(folder_path)
        sample_count = len(images)
    
    if sample_count == 0:
        if verbose:
            print(f"  Skipped (no valid samples): {folder_path}")
        return None
    
    if verbose:
        print(f"  Evaluating {sample_count} samples: {folder_path}")
    
    all_scores = []
    
    if is_pair_scorer:
        source_images = images['source']
        edited_images = images['edited']
        
        for start_idx in tqdm(range(0, sample_count, batch_size), disable=not verbose):
            end_idx = min(start_idx + batch_size, sample_count)
            batch_images = {
                'source': source_images[start_idx:end_idx],
                'edited': edited_images[start_idx:end_idx]
            }
            batch_prompts = prompts[start_idx:end_idx]
            
            try:
                batch_results = evaluator.evaluate(model_name, batch_images, batch_prompts)
                scores = batch_results.get('scores', batch_results) if isinstance(batch_results, dict) else batch_results
                all_scores.extend(scores)
            except Exception as e:
                print(f"    Batch evaluation failed [{start_idx}:{end_idx}]: {e}")
                return None
    else:
        for start_idx in tqdm(range(0, sample_count, batch_size), disable=not verbose):
            end_idx = min(start_idx + batch_size, sample_count)
            batch_images = images[start_idx:end_idx]
            batch_prompts = prompts[start_idx:end_idx]
            
            try:
                batch_results = evaluator.evaluate(model_name, batch_images, batch_prompts)
                scores = batch_results.get('scores', batch_results) if isinstance(batch_results, dict) else batch_results
                all_scores.extend(scores)
            except Exception as e:
                print(f"    Batch evaluation failed [{start_idx}:{end_idx}]: {e}")
                continue
    
    if not all_scores:
        return None
    
    return {
        'folder': folder_path,
        'model': model_name,
        'average': sum(all_scores) / len(all_scores),
        'scores': all_scores,
        'count': len(all_scores)
    }


def find_leaf_folders(root_path: str, min_depth: int = 0, max_depth: int = -1) -> List[str]:
    result = []
    root_path = os.path.abspath(root_path)
    
    def has_images(folder: str) -> bool:
        for f in os.listdir(folder):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                return True
        return False
    
    def recurse(current_path: str, depth: int):
        if max_depth >= 0 and depth > max_depth:
            return
        
        try:
            entries = os.listdir(current_path)
        except PermissionError:
            return
        
        subdirs = [e for e in entries if os.path.isdir(os.path.join(current_path, e))]
        
        if not subdirs or (max_depth >= 0 and depth == max_depth):
            if depth >= min_depth and has_images(current_path):
                result.append(current_path)
        else:
            for subdir in subdirs:
                recurse(os.path.join(current_path, subdir), depth + 1)
            if depth >= min_depth and has_images(current_path):
                result.append(current_path)
    
    recurse(root_path, 0)
    return sorted(result)


def run(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    scorer_urls = config['scorer_urls']
    defaults = config.get('defaults', {})
    evaluations = config['evaluations']
    output_file = config.get('output')
    verbose = config.get('verbose', True)
    
    default_batch_size = defaults.get('batch_size', 64)
    default_recursive = defaults.get('recursive', False)
    default_min_depth = defaults.get('min_depth', 0)
    default_max_depth = defaults.get('max_depth', -1)
    
    all_results = {}
    
    for eval_item in evaluations:
        path = eval_item.get('path')
        if not path:
            print("Warning: Evaluation item missing 'path', skipping")
            continue
        
        models = eval_item.get('models', [])
        if not models:
            print(f"Warning: No models specified for {path}, skipping")
            continue
        
        batch_size = eval_item.get('batch_size', default_batch_size)
        recursive = eval_item.get('recursive', default_recursive)
        min_depth = eval_item.get('min_depth', default_min_depth)
        max_depth = eval_item.get('max_depth', default_max_depth)
        
        if not recursive:
            max_depth = 0
        
        folders = find_leaf_folders(path, min_depth, max_depth)
        
        if not folders:
            print(f"No image folders found in: {path}")
            continue
        
        print(f"\nProcessing {len(folders)} folder(s) from: {path}")
        print(f"Models: {', '.join(models)}")
        print("-" * 60)
        
        for folder in tqdm(folders, desc="Folders", disable=not verbose):
            folder_results = {}
            
            for model in models:
                if verbose:
                    print(f"\n[{model}] ", end="")
                
                result = evaluate_folder(folder, model, batch_size, scorer_urls, verbose)
                
                if result:
                    folder_results[model] = result
                    if verbose:
                        print(f"    -> Average: {result['average']:.4f} (n={result['count']})")
            
            if folder_results:
                rel_path = os.path.relpath(folder, path)
                key = f"{path}:{rel_path}" if rel_path != "." else path
                all_results[key] = folder_results
    
    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    for folder, results in all_results.items():
        print(f"\n{folder}")
        for model, data in results.items():
            print(f"   [{model}] avg={data['average']:.4f}, n={data['count']}")
    
    # Save results
    if output_file:
        serializable = {
            folder: {
                model: {'average': data['average'], 'count': data['count']}
                for model, data in results.items()
            }
            for folder, results in all_results.items()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': serializable
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")
    
    return all_results


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    results = run(config)
    sys.exit(0 if results else 1)


if __name__ == "__main__":
    main()