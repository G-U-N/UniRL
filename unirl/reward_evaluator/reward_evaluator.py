import pickle
from io import BytesIO
from typing import Any, Dict, List, Mapping, Optional, Union

import requests
from PIL import Image


DEFAULT_EDITREWARD_URL = "http://127.0.0.1:18088/"
DEFAULT_GENEVAL_URL = "http://127.0.0.1:18085/"


def _serialize_image(image: Image.Image) -> bytes:
    buffer = BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _serialize_images(
    images: Union[List[Image.Image], Mapping[str, List[Image.Image]]],
) -> Union[List[bytes], Dict[str, List[bytes]]]:
    if isinstance(images, Mapping):
        return {key: [_serialize_image(image) for image in value] for key, value in images.items()}
    return [_serialize_image(image) for image in images]


def _create_payload(
    images: Union[List[Image.Image], Mapping[str, List[Image.Image]]],
    prompts: List[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> bytes:
    return pickle.dumps(
        {
            "images": _serialize_images(images),
            "prompts": prompts,
            "metadata": metadata or {},
        }
    )


class RewardEvaluatorClient:
    """HTTP client for reward scorer services (EditReward + GenEval)."""

    def __init__(
        self,
        editreward_url: str = DEFAULT_EDITREWARD_URL,
        geneval_url: str = DEFAULT_GENEVAL_URL,
        timeout: int = 600,
    ):
        self.editreward_url = editreward_url
        self.geneval_url = geneval_url
        self.timeout = timeout

    def evaluate_editreward(
        self,
        source_images: List[Image.Image],
        edited_images: List[Image.Image],
        prompts: List[str],
    ) -> Dict[str, Any]:
        if not (len(source_images) == len(edited_images) == len(prompts)):
            raise ValueError(
                "EditReward inputs must have equal lengths: "
                f"{len(source_images)} source images, {len(edited_images)} edited images, {len(prompts)} prompts."
            )

        payload = _create_payload(
            {"source": source_images, "edited": edited_images},
            prompts,
        )
        response = requests.post(self.editreward_url, data=payload, timeout=self.timeout)
        response.raise_for_status()
        result = pickle.loads(response.content)
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"EditReward service returned an error: {result['error']}")
        return result

    def evaluate_geneval(
        self,
        images: List[Image.Image],
        prompts: List[str],
        meta_datas: List[Dict[str, Any]],
        only_strict: bool = False,
    ) -> Dict[str, Any]:
        """Score generated images against GenEval object/attribute spec.

        ``meta_datas[i]`` is the GenEval-style dict (``tag``, ``include``,
        ``exclude``) for the i-th image. The service returns
        ``{scores, rewards, strict_rewards, group_rewards, group_strict_rewards}``;
        callers typically use either ``scores`` (soft, in [0,1]) or
        ``strict_rewards`` (binary) as the GRPO reward.
        """
        if not (len(images) == len(prompts) == len(meta_datas)):
            raise ValueError(
                "GenEval inputs must have equal lengths: "
                f"{len(images)} images, {len(prompts)} prompts, {len(meta_datas)} meta_datas."
            )
        payload = _create_payload(
            images,
            prompts,
            metadata={"meta_datas": meta_datas, "only_strict": only_strict},
        )
        response = requests.post(self.geneval_url, data=payload, timeout=self.timeout)
        response.raise_for_status()
        result = pickle.loads(response.content)
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"GenEval service returned an error: {result['error']}")
        return result
