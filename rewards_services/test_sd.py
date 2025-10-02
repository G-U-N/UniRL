import pickle
import requests
from PIL import Image
import io

payload = {
    "prompts": ["a majestic dragon flying over a mystical forest, fantasy art, detailed"],
    "negative_prompts": ["blurry, lowres, text, watermark"],
    "width": 512,
    "height": 512,
    "num_inference_steps": 10,
    "guidance_scale": 3.5,
    "num_images": 8
}

data = pickle.dumps(payload)

response = requests.post("http://0.0.0.0:18099/", data=data, timeout=120)
response.raise_for_status() 
print(response)
result = pickle.loads(response.content)
if result["success"]:
    image_bytes = result["images"][0]
    image = Image.open(io.BytesIO(image_bytes))
    image.save("generated_image.png") 