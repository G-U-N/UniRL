"""Kontext inference smoke test (verifies the _step_index fix).

Runs the fused Qwen-Kontext checkpoint on a single sample from the eval parquet.
The interesting comparison is:
  - deterministic 8-step (sde_sampling=False)         <- ground-truth reference
  - SDE 8-step num_sde=4 noise=0.0 (sde_sampling=True) <- must match deterministic
  - SDE 8-step num_sde=4 noise=0.8 (sde_sampling=True) <- training mode

If the fix is correct, noise=0.0 must be visually identical to deterministic,
and noise=0.8 should be a slightly noisier but still recognizable variant of the
deterministic image (Kontext is distilled so already worked visually; this test
just confirms the fix doesn't regress it).

By default, downloads `omni_edit_dev.parquet` from the `wangfuyun/PrompRL` HF
repo; pass `--data_file` to point at a local parquet instead.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/inference/kontext_smoke_test.py
"""

import argparse
import io
import os
import re
import sys
import time

import pyarrow.parquet as pq
import torch
from PIL import Image
from torchvision import transforms as TF

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from unimodel.qwenkontext.qwenkontext_inference import QwenKontextForInferenceLM

DEFAULT_DATA_FILE = (
    "https://huggingface.co/wangfuyun/PrompRL/resolve/main/data/omni_edit_dev.parquet"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="outputs/pretrain/qwenkontext")
    p.add_argument(
        "--data_file",
        default=DEFAULT_DATA_FILE,
        help="Local parquet path or an https://huggingface.co/.../resolve/... URL.",
    )
    p.add_argument("--row", type=int, default=0)
    p.add_argument("--num_inference_steps", type=int, default=8)
    p.add_argument("--guidance_scale", type=float, default=2.5)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_sde", type=int, default=4)
    p.add_argument("--output_dir", default="outputs/kontext_smoke")
    return p.parse_args()


_HF_URL_RE = re.compile(
    r"^https?://huggingface\.co/(?P<repo>[^/]+/[^/]+)/resolve/(?P<rev>[^/]+)/(?P<path>.+)$"
)


def _resolve_data_file(path_or_url: str) -> str:
    """Return a local parquet path, downloading from HF Hub if a URL was passed."""
    m = _HF_URL_RE.match(path_or_url)
    if m is None:
        return path_or_url
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=m.group("repo"),
        filename=m.group("path"),
        revision=m.group("rev"),
    )


def load_sample(parquet_path, row):
    pf = pq.ParquetFile(_resolve_data_file(parquet_path))
    batch = next(pf.iter_batches(batch_size=max(1, row + 1)))
    record = batch.to_pydict()
    img_bytes = record["image"][row]["bytes"]
    return Image.open(io.BytesIO(img_bytes)).convert("RGB"), record["prompt"][row]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    image, prompt = load_sample(args.data_file, args.row)
    image = TF.Compose([TF.Resize(args.resolution), TF.CenterCrop(args.resolution)])(image)
    image.save(os.path.join(args.output_dir, "source.png"))
    with open(os.path.join(args.output_dir, "prompt.txt"), "w") as fp:
        fp.write(prompt)
    print(f"[input] {image.size} prompt={prompt!r}")

    print(f"[loader] Loading Qwen-Kontext from {args.model_path} ...")
    t0 = time.time()
    model = QwenKontextForInferenceLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    model.to("cuda")
    print(f"[loader] done in {time.time()-t0:.1f}s")

    # Use bytes==0 max_area to disable; pass image size via height/width semantics
    # Kontext's pipeline accepts max_area, so pass that for parity.
    base_kwargs = dict(
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        num_images_per_prompt=1,
        max_area=image.size[0] * image.size[1],
    )

    configs = [
        ("deterministic", dict(sde_sampling=False, extra={})),
        ("sde_ns0.0",     dict(sde_sampling=True,  extra=dict(num_sde=args.num_sde, noise_scale=0.0))),
        ("sde_ns0.8",     dict(sde_sampling=True,  extra=dict(num_sde=args.num_sde, noise_scale=0.8))),
    ]

    for name, cfg in configs:
        diffusion_kwargs = dict(base_kwargs)
        diffusion_kwargs["generator"] = torch.Generator("cpu").manual_seed(args.seed)
        diffusion_kwargs.update(cfg["extra"])

        t0 = time.time()
        result = model.generate_image(
            images=[image],
            texts=[prompt],
            diffusion_kwargs=diffusion_kwargs,
            sde_sampling=cfg["sde_sampling"],
        )
        elapsed = time.time() - t0
        imgs = result[0] if cfg["sde_sampling"] else result
        out_path = os.path.join(
            args.output_dir, f"qwenkontext_steps{args.num_inference_steps:03d}_{name}.png"
        )
        imgs[0].save(out_path)
        print(f"[kontext] {name}: {elapsed:.1f}s -> {out_path}")

    print(f"[done] outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
