import os
import pickle
import traceback
from io import BytesIO
from typing import Any, Dict, List

import torch
from flask import Blueprint, Flask, request
from PIL import Image

from editreward_scorer import EditRewardScorer


INFERENCE_FN = None
root = Blueprint("root", __name__)


def _deserialize_images(images_bytes: List[bytes]) -> List[Image.Image]:
    return [Image.open(BytesIO(data)).convert("RGB") for data in images_bytes]


def _service_config() -> Dict[str, Any]:
    repo_dir = os.getenv("EDITREWARD_REPO_DIR", os.path.join(os.path.dirname(__file__), "EditReward"))
    return {
        "repo_dir": repo_dir,
        "config_path": os.getenv(
            "EDITREWARD_CONFIG_PATH",
            os.path.join(repo_dir, "EditReward", "config", "EditReward-MiMo-VL-7B-SFT-2508.yaml"),
        ),
        "checkpoint_path": os.getenv(
            "EDITREWARD_CHECKPOINT_PATH",
            os.path.join(repo_dir, "EditReward-MiMo-VL-7B-SFT-2508"),
        ),
        "reward_dim": os.getenv("EDITREWARD_DIM", "overall_detail"),
        "rm_head_type": os.getenv("EDITREWARD_HEAD_TYPE", "ranknet_multi_head"),
    }


def create_app():
    global INFERENCE_FN
    config = _service_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading EditReward scorer on {device} from {config['checkpoint_path']}...")
    INFERENCE_FN = EditRewardScorer(
        repo_dir=config["repo_dir"],
        config_path=config["config_path"],
        checkpoint_path=config["checkpoint_path"],
        reward_dim=config["reward_dim"],
        rm_head_type=config["rm_head_type"],
        device=device,
    )
    INFERENCE_FN.eval()
    print("EditReward scorer loaded.")

    app = Flask(__name__)
    app.register_blueprint(root)
    return app


@root.route("/", methods=["GET"])
def healthcheck():
    return {"status": "ok", "service": "editreward"}, 200


@root.route("/", methods=["POST"])
def inference():
    try:
        payload = pickle.loads(request.get_data())
        images = payload["images"]
        prompts = payload.get("prompts", [])
        source_images = _deserialize_images(images.get("source", []))
        edited_images = _deserialize_images(images.get("edited", []))

        if len(source_images) != len(edited_images) or len(source_images) != len(prompts):
            raise ValueError(
                "Mismatched EditReward inputs: "
                f"{len(source_images)} source images, {len(edited_images)} edited images, {len(prompts)} prompts."
            )

        with torch.no_grad():
            scores = INFERENCE_FN(prompts, source_images, edited_images)
        return pickle.dumps({"scores": [float(score) for score in scores]}), 200
    except Exception:
        error_message = traceback.format_exc()
        print(f"EditReward service error:\n{error_message}")
        return pickle.dumps({"error": error_message}), 500


if __name__ == "__main__":
    port = int(os.getenv("EDITREWARD_PORT", "18088"))
    host = os.getenv("EDITREWARD_HOST", "127.0.0.1")
    app = create_app()
    app.run(host=host, port=port, debug=False)

