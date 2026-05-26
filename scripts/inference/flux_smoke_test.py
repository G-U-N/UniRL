"""Flux inference smoke test (verifies the _step_index fix on FLUX.1-dev).

QwenFlux is t2i only — no image conditioning — so this test takes a single
caption and runs three configurations to confirm the SDE bookkeeping fix:
  - deterministic (sde_sampling=False)        <- reference
  - SDE num_sde=4 noise=0.0                   <- must match deterministic
  - SDE num_sde=4 noise=0.8                   <- training mode

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/inference/flux_smoke_test.py
"""

import argparse
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from unimodel.qwenflux.qwenflux_inference import QwenFluxForInferenceLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="outputs/pretrain/qwenflux")
    p.add_argument(
        "--prompt",
        default="A photo of a red fox sitting on a moss-covered rock in a misty forest at dawn.",
    )
    p.add_argument("--num_inference_steps", type=int, default=8)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_sde", type=int, default=4)
    p.add_argument("--output_dir", default="outputs/flux_smoke")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "prompt.txt"), "w") as fp:
        fp.write(args.prompt)
    print(f"[input] prompt={args.prompt!r}  size={args.width}x{args.height}")

    print(f"[loader] Loading Qwen-Flux from {args.model_path} ...")
    t0 = time.time()
    model = QwenFluxForInferenceLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    model.to("cuda")
    print(f"[loader] done in {time.time()-t0:.1f}s")

    base_kwargs = dict(
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        num_images_per_prompt=1,
        height=args.height,
        width=args.width,
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
            texts=[args.prompt],
            diffusion_kwargs=diffusion_kwargs,
            sde_sampling=cfg["sde_sampling"],
        )
        elapsed = time.time() - t0
        imgs = result[0] if cfg["sde_sampling"] else result
        out_path = os.path.join(
            args.output_dir, f"qwenflux_steps{args.num_inference_steps:03d}_{name}.png"
        )
        imgs[0].save(out_path)
        print(f"[flux] {name}: {elapsed:.1f}s -> {out_path}")

    print(f"[done] outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
