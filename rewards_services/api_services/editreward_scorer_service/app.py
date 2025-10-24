import sys
import os
import torch
from PIL import Image
import pickle
import traceback
from io import BytesIO

from flask import Flask, request, Blueprint

sys.path.append("./EditReward")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../common')))

from utils import deserialize_images

from editreward_scorer import EditRewardScorer

INFERENCE_FN = None

root = Blueprint("root", __name__)

def create_app():
    global INFERENCE_FN
    print("Loading EditReward Scorer model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_path = "EditReward/EditReward/config/EditReward-MiMo-VL-7B-SFT-2508.yaml"
    checkpoint_path = "EditReward/EditReward-MiMo-VL-7B-SFT-2508"
    INFERENCE_FN = EditRewardScorer(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        dtype=torch.float32,
        device=device
    )
    INFERENCE_FN.eval()
    print("EditReward Scorer model loaded.")

    app = Flask(__name__)
    app.register_blueprint(root)
    return app

@root.route("/", methods=["POST"])
def inference():
    print(f"Received POST request from {request.remote_addr}")
    data = request.get_data()

    try:
        payload = pickle.loads(data)
        images_bytes = payload["images"]
        prompts = payload.get("prompts", [])


        # Deserialize images
        image_src = deserialize_images(images_bytes["source"]) if "source" in images_bytes else []
        image_paths = deserialize_images(images_bytes["edited"]) if "edited" in images_bytes else []

        if len(image_src) != len(image_paths) or len(image_src) != len(prompts):
            raise ValueError("Mismatched number of source images, edited images, and prompts")

        print(f"Got {len(image_paths)} images for EditReward Scorer")

        with torch.no_grad():
            rewards = INFERENCE_FN(prompts, image_src, image_paths)
        scores = rewards
        response = {"scores": scores}
        response = pickle.dumps(response)
        return response, 200

    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"Error in EditReward Scorer service: {error_msg}")
        response = {"error": error_msg}
        response = pickle.dumps(response)
        return response, 500

if __name__ == "__main__":
    # For local development/testing only
    # For production, use Gunicorn
    app = create_app()
    app.run(host="0.0.0.0", port=18088, debug=True)