import requests
import pickle
from PIL import Image
from typing import List, Dict, Any, Union
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../common')))
from utils import create_payload, parse_response

SCORER_URLS = {
    "aesthetic": "http://0.0.0.0:18080/",
    "image_reward": "http://0.0.0.0:18081/",
    "ocr": "http://0.0.0.0:18082/",
    "pickscore": "http://0.0.0.0:18083/",
    "deqa": "http://0.0.0.0:18084/",
    "gen_eval": "http://0.0.0.0:18085/",
    "unifiedreward_sglang": "http://0.0.0.0:18086/", 
    "hps": "http://0.0.0.0:18087/", 
    "editreward": "http://0.0.0.0:18088/", 

}

class RewardEvaluatorClient:
    def __init__(self, scorer_urls: Dict[str, str] = SCORER_URLS):
        self.scorer_urls = scorer_urls

    def evaluate(self, 
                 model_name: str, 
                 images: List[Image.Image], 
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
                raise RuntimeError(f"Scorer '{model_name}' service returned error: {result['error']}")
            
            return result

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"HTTP request to '{model_name}' failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to process response from '{model_name}': {e}")

    def evaluate_multiple(self, 
                          model_weights: Dict[str, float], 
                          images: List[Image.Image], 
                          prompts: List[str], 
                          metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        all_results = {}
        for model_name, weight in model_weights.items():
            if weight == 0: 
                continue
            try:
                if model_name in ["gen_eval", "unifiedreward_sglang"]:
                    specific_metadata = metadata.get(model_name, {})
                    result = self.evaluate(model_name, images, prompts, specific_metadata)
                else:
                    result = self.evaluate(model_name, images, prompts, metadata.get(model_name, {}))

                all_results[model_name] = result
            except Exception as e:
                print(f"Error evaluating model {model_name}: {e}")
                all_results[model_name] = {"error": str(e)}
        return all_results