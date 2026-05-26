"""Klein inference smoke test.

Runs the fused Qwen-Klein checkpoint (or the raw FLUX.2-klein-base pipeline) on a
single sample from the eval parquet at several `num_inference_steps` counts and
saves all outputs side-by-side for visual comparison. By default, downloads
`omni_edit_dev.parquet` from the `wangfuyun/PrompRL` HF dataset; pass
`--data_file` to point at a local parquet instead.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/inference/klein_smoke_test.py
        [--model_path outputs/pretrain/qwenklein]
        [--num_inference_steps 8,28,50]
        [--guidance_scale 4.0] [--row 0]
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

from unimodel.qwenklein.qwenklein_inference import QwenKleinForInferenceLM

DEFAULT_DATA_FILE = (
    "https://huggingface.co/wangfuyun/PrompRL/resolve/main/data/omni_edit_dev.parquet"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="outputs/pretrain/qwenklein")
    p.add_argument(
        "--data_file",
        default=DEFAULT_DATA_FILE,
        help="Local parquet path or an https://huggingface.co/.../resolve/... URL.",
    )
    p.add_argument("--row", type=int, default=0)
    p.add_argument(
        "--num_inference_steps",
        default="8,28,50",
        help="Comma-separated list of step counts to compare.",
    )
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="outputs/klein_smoke")
    p.add_argument(
        "--raw_pipeline",
        action="store_true",
        help="Skip the Qwen wrapper and call diffusers.Flux2KleinPipeline directly "
        "(useful to isolate whether the wrapper is at fault).",
    )
    p.add_argument(
        "--sde_sampling",
        action="store_true",
        help="Use the GRPO-mode SDE sampling path (noise injected each step). "
        "Useful to reproduce what the trainer sees.",
    )
    p.add_argument(
        "--num_sde",
        type=int,
        default=4,
        help="Number of SDE noise samples per step (matches num_sde used during training).",
    )
    p.add_argument(
        "--noise_scale",
        type=float,
        default=0.8,
        help="SDE noise scale (matches PROMPTRL_EDIT_SDE_NOISE_SCALE used during training).",
    )
    p.add_argument(
        "--prompt_override",
        default=None,
        help="If set, use this editing instruction instead of the parquet prompt.",
    )
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


def load_sample(parquet_path: str, row: int):
    pf = pq.ParquetFile(_resolve_data_file(parquet_path))
    rows_per_batch = max(1, row + 1)
    batch = next(pf.iter_batches(batch_size=rows_per_batch))
    record = batch.to_pydict()
    img_bytes = record["image"][row]["bytes"]
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    prompt = record["prompt"][row]
    return image, prompt


def run_qwenklein(
    model_path,
    image,
    prompt,
    steps_list,
    guidance_scale,
    seed,
    output_dir,
    sde_sampling=False,
    num_sde=4,
    noise_scale=0.8,
):
    print(f"[loader] Loading Qwen-Klein from {model_path} ...")
    t0 = time.time()
    model = QwenKleinForInferenceLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.to("cuda")
    print(f"[loader] done in {time.time()-t0:.1f}s")

    for steps in steps_list:
        diffusion_kwargs = dict(
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            num_images_per_prompt=1,
            generator=torch.Generator("cpu").manual_seed(seed),
            height=image.size[1],
            width=image.size[0],
        )
        if sde_sampling:
            diffusion_kwargs.update(num_sde=num_sde, noise_scale=noise_scale)

        t0 = time.time()
        result = model.generate_image(
            images=[image],
            texts=[prompt],
            diffusion_kwargs=diffusion_kwargs,
            sde_sampling=sde_sampling,
        )
        elapsed = time.time() - t0
        out_imgs = result[0] if sde_sampling else result  # sde_sampling returns 6-tuple
        suffix = f"_sde{num_sde}_ns{noise_scale}" if sde_sampling else ""
        out_path = os.path.join(
            output_dir, f"qwenklein_steps{steps:03d}_g{guidance_scale}{suffix}.png"
        )
        out_imgs[0].save(out_path)
        print(f"[qwenklein] steps={steps} sde={sde_sampling} took {elapsed:.1f}s -> {out_path}")


def run_raw_pipeline(image, prompt, steps_list, guidance_scale, seed, output_dir):
    from diffusers import Flux2KleinPipeline

    repo_id = "black-forest-labs/FLUX.2-klein-base-4B"
    print(f"[loader] Loading raw Flux2KleinPipeline from {repo_id} ...")
    t0 = time.time()
    pipe = Flux2KleinPipeline.from_pretrained(repo_id, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    print(f"[loader] done in {time.time()-t0:.1f}s")

    for steps in steps_list:
        t0 = time.time()
        out = pipe(
            image=image,
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            num_images_per_prompt=1,
            generator=torch.Generator("cpu").manual_seed(seed),
            height=image.size[1],
            width=image.size[0],
        )
        elapsed = time.time() - t0
        out_path = os.path.join(output_dir, f"raw_steps{steps:03d}_g{guidance_scale}.png")
        out.images[0].save(out_path)
        print(f"[raw] steps={steps} took {elapsed:.1f}s -> {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    steps_list = [int(s) for s in args.num_inference_steps.split(",") if s.strip()]

    image, parquet_prompt = load_sample(args.data_file, args.row)
    prompt = args.prompt_override or parquet_prompt

    image = TF.Compose([TF.Resize(args.resolution), TF.CenterCrop(args.resolution)])(image)
    image.save(os.path.join(args.output_dir, "source.png"))
    with open(os.path.join(args.output_dir, "prompt.txt"), "w") as fp:
        fp.write(prompt)
    print(f"[input] resolution={image.size} prompt={prompt!r}")
    print(f"[input] running steps_list={steps_list}, guidance_scale={args.guidance_scale}")

    if args.raw_pipeline:
        run_raw_pipeline(image, prompt, steps_list, args.guidance_scale, args.seed, args.output_dir)
    else:
        run_qwenklein(
            args.model_path,
            image,
            prompt,
            steps_list,
            args.guidance_scale,
            args.seed,
            args.output_dir,
            sde_sampling=args.sde_sampling,
            num_sde=args.num_sde,
            noise_scale=args.noise_scale,
        )

    print(f"[done] All outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
