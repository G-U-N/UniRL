import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from client.reward_evaluator import RewardEvaluatorClient


def main():
    images_pil = [
        Image.open("examples/1.png").convert("RGB"),
        Image.open("examples/1.png").convert("RGB"),
        Image.open("examples/1.png").convert("RGB"),
        Image.open("examples/1.png").convert("RGB"),

    ]
    
    prompts = [
        "a photo of a brown giraffe and a white \"stop\" sign.",
        "a photo of a brown giraffe and a white stop sign. \"stop\"",
        "a photo of a brown giraffe and a white stop sign. \"stop\"",
        "a photo of a brown giraffe and a white stop sign. \"stop\"",


    ]

    evaluator = RewardEvaluatorClient()

    print("\n--- Evaluating with Aesthetic Scorer ---")
    try:
        aesthetic_results = evaluator.evaluate("aesthetic", images_pil, prompts)
        print(f"Aesthetic Scores: {aesthetic_results.get('scores', 'N/A')}")
    except Exception as e:
        print(f"Aesthetic Scorer Error: {e}")

    print("\n--- Evaluating with ImageReward Scorer ---")
    try:
        image_reward_results = evaluator.evaluate("image_reward", images_pil, prompts)
        print(f"ImageReward Scores: {image_reward_results.get('scores', 'N/A')}")
    except Exception as e:
        print(f"ImageReward Scorer Error: {e}")

    print("\n--- Evaluating with OCR Scorer ---")
    try:
        ocr_results = evaluator.evaluate("ocr", images_pil, prompts) 
        print(f"OCR Scores: {ocr_results.get('scores', 'N/A')}")
    except Exception as e:
        print(f"OCR Scorer Error: {e}")

    print("\n--- Evaluating with PickScore Scorer ---")
    try:
        pickscore_results = evaluator.evaluate("pickscore", images_pil, prompts)
        print(f"PickScore Scores: {pickscore_results.get('scores', 'N/A')}")
    except Exception as e:
        print(f"PickScore Scorer Error: {e}")
        
    print("\n--- Evaluating with DeQA Scorer ---")
    try:
        deqa_results = evaluator.evaluate("deqa", images_pil, prompts)
        print(f"DeQA Scores: {deqa_results.get('scores', 'N/A')}")
    except Exception as e:
        print(f"DeQA Scorer Error: {e}")

    print("\n--- Evaluating with HPSv2 Scorer ---")
    try:
        deqa_results = evaluator.evaluate("hps", images_pil, prompts)
        print(f"HPSv2 Scores: {deqa_results.get('scores', 'N/A')}")
    except Exception as e:
        print(f"HPSv2 Scorer Error: {e}")

    print("\n--- Evaluating with GenEval Scorer ---")
    try:
        geneval_metadata = {
            "meta_datas": [{"tag": "color_attr", "include": [{"class": "giraffe", "count": 1, "color": "brown"}, {"class": "stop sign", "count": 1, "color": "white"}], "prompt": "a photo of a brown giraffe and a white stop sign"}],
            "only_strict": False,
        }
        geneval_results = evaluator.evaluate("gen_eval", images_pil, prompts, geneval_metadata)
        print(f"GenEval Results: {geneval_results}")
    except Exception as e:
        print(f"GenEval Scorer Error: {e}")

    print("\n--- Evaluating with UnifiedReward (SGLang) Scorer ---")
    try:
        unifiedreward_metadata = {
        }
        unifiedreward_results = evaluator.evaluate("unifiedreward_sglang", images_pil, prompts, unifiedreward_metadata)
        print(f"UnifiedReward (SGLang) Scores: {unifiedreward_results.get('scores', 'N/A')}")
    except Exception as e:
        print(f"UnifiedReward (SGLang) Scorer Error: {e}")

    print("\n--- Evaluating Multiple Models ---")
    model_weights_example = {
        "aesthetic": 0.2,
        "image_reward": 0.2,
        "ocr": 0.1,
        "pickscore": 0.2,
        "deqa": 0.1, 
        "gen_eval": 0.1,
        "unifiedreward_sglang": 0.1, 
        "hps": 0.1,
    }
    
    combined_metadata_for_multiple = {
        "aesthetic": {},
        "image_reward": {},
        "ocr": {},
        "pickscore": {},
        "deqa": {},
        "gen_eval": { 
            "meta_datas": [{"tag": "color_attr", "include": [{"class": "giraffe", "count": 1, "color": "brown"}, {"class": "stop sign", "count": 1, "color": "white"}], "prompt": "a photo of a brown giraffe and a white stop sign"}],
            "only_strict": False,
        },
        "unifiedreward_sglang": {},
        "hps": {},
    }

    all_results = evaluator.evaluate_multiple(model_weights_example, images_pil, prompts, combined_metadata_for_multiple)
    
    print(f"Combined Results (raw from each service): {all_results}")

if __name__ == "__main__":
    main()