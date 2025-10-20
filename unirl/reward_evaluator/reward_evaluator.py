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

import requests
import pickle
from PIL import Image
from typing import List, Dict, Any, Union
import sys
import os
import pickle
from io import BytesIO

SCORER_URLS = {
    "aesthetic": "http://[NODE_ADDR]:18080/",
    "image_reward": "http://[NODE_ADDR]:18081/",
    "ocr": "http://[NODE_ADDR]:18082/",
    "pickscore": "http://[NODE_ADDR]:18083/",
    "deqa": "http://[NODE_ADDR]:18084/",
    "gen_eval": "http://[NODE_ADDR]:18085/",
    "unifiedreward_sglang": "http://[NODE_ADDR]:18086/", 
    "hps": "http://[NODE_ADDR]:18087/", 

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
    

def serialize_images(images: List[Image.Image]) -> List[bytes]:
    images_bytes = []
    for img in images:
        img_byte_arr = BytesIO()
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # img.save(img_byte_arr, format="JPEG", quality=95)
        img.save(img_byte_arr, format="JPEG")
        images_bytes.append(img_byte_arr.getvalue())
    return images_bytes

def deserialize_images(images_bytes: List[bytes]) -> List[Image.Image]:
    images = [Image.open(BytesIO(d)) for d in images_bytes]
    return images

def create_payload(images: List[Image.Image], prompts: List[str], metadata: Dict[str, Any] = None) -> bytes:
    serialized_images = serialize_images(images)
    payload = {
        "images": serialized_images,
        "prompts": prompts,
        "metadata": metadata if metadata is not None else {}
    }
    return pickle.dumps(payload)

def parse_response(response_content: bytes) -> Union[List[float], Dict[str, Any]]:
    return pickle.loads(response_content)