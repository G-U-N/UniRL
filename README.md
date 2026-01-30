<p align="center">
  <img src="assets/logo.png" width="25%"><br>
  PromptRL
</p>


## Overview

**PromptRL** is a framework that jointly trains language models (LMs) and flow-matching models (FMs) within a unified reinforcement learning loop for text-to-image generation. By incorporating LMs as adaptive prompt refiners, PromptRL addresses two critical limitations in current flow-based RL pipelines: *exploration collapse* due to insufficient generation diversity, and *prompt overfitting* where models memorize specific training formulations.


## Installation

```bash
conda env create -f environment.yml
conda activate unirl
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/huggingface/diffusers.git
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## Qualitative Results

### Text-to-Image Generation
<p align="center">
  <img src="assets/t2i_comparison.png" width="85%">
</p>

### Instructional Image Editing
<p align="center">
  <img src="assets/edit_comparison.png" width="75%">
</p>


## Key Results

PromptRL achieves **2× sample efficiency** compared to flow-only RL while maintaining robust generalization to diverse prompt formulations.

### Summary

| Benchmark | Metric | PromptRL w/ PE | Best Baseline |
|:---|:---|:---:|:---:|
| GenEval | Avg. Score ↑ | **0.97** | 0.92 (FlowGRPO) |
| Aesthetic | PickScore ↑ | **24.05** | 23.63 (DiffusionNFT) |
| Aesthetic | HPS ↑ | **32.03** | 31.79 (DiffusionNFT) |
| OCR | OCR-1k ↑ | **0.98** | 0.89 (FlowGRPO) |
| Image Editing | EditReward Avg. ↑ | **1.43** | 1.44 (ReasonEdit-Think) |

---

<details>
<summary><b>📊 GenEval Benchmark (Full Results)</b></summary>

<br>

| Model | 1 Obj. | 2 Obj. | Cnt. | Clr. | Pos. | Attr. | Avg.↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Show-o | 0.95 | 0.52 | 0.49 | 0.82 | 0.11 | 0.28 | 0.53 |
| Emu3-Gen | 0.98 | 0.71 | 0.34 | 0.81 | 0.17 | 0.21 | 0.54 |
| SD3 Medium | 0.98 | 0.74 | 0.63 | 0.67 | 0.34 | 0.36 | 0.62 |
| FLUX.1-dev | 0.98 | 0.81 | 0.74 | 0.79 | 0.22 | 0.45 | 0.66 |
| SD3.5 Large | 0.98 | 0.89 | 0.73 | 0.83 | 0.34 | 0.47 | 0.71 |
| JanusFlow | 0.97 | 0.59 | 0.45 | 0.83 | 0.53 | 0.42 | 0.63 |
| Janus-Pro-7B | 0.99 | 0.89 | 0.59 | 0.90 | 0.79 | 0.66 | 0.80 |
| HiDream | 1.00 | 0.98 | 0.79 | 0.91 | 0.60 | 0.72 | 0.83 |
| Seedream 3.0 | 0.99 | 0.96 | 0.91 | 0.93 | 0.47 | 0.80 | 0.84 |
| Qwen-Image | 0.99 | 0.92 | 0.89 | 0.88 | 0.76 | 0.77 | 0.87 |
| *RL-based* |  |  |  |  |  |  |  |
| RePrompt | 0.98 | 0.87 | 0.77 | 0.85 | 0.62 | 0.49 | 0.76 |
| FlowGRPO | 1.00 | 0.99 | 0.91 | 0.89 | 0.95 | 0.80 | 0.92 |
| DiffusionNFT | 1.00 | 0.98 | 0.74 | 0.92 | 0.85 | 0.80 | 0.88 |
| PromptRL w/o PE | 1.00 | 0.96 | 0.95 | 0.95 | 0.93 | 0.85 | 0.94 |
| **PromptRL w/ PE** | **1.00** | **0.99** | **0.99** | **0.96** | **0.99** | **0.90** | **0.97** |

</details>

<details>
<summary><b>🎨 Aesthetic & OCR Metrics (Full Results)</b></summary>

<br>

| Model | P.S. | HPS | U.R. | OCR-1k | TMDB | OpenLib |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| SD1.5 | 20.92 | 23.71 | 2.00 | 0.05 | 0.13 | 0.08 |
| SDXL | 22.14 | 26.67 | 2.78 | 0.13 | 0.20 | 0.09 |
| SD3 Medium | 22.38 | 28.56 | 3.09 | — | 0.44 | 0.33 |
| FLUX.1-schnell | 22.64 | 29.39 | 3.25 | 0.54 | 0.66 | 0.50 |
| FLUX.2-klein | 22.79 | 29.03 | 3.29 | 0.55 | 0.22 | 0.46 |
| Z-Image | 20.14 | 28.22 | 3.51 | 0.70 | 0.71 | 0.83 |
| Qwen-Image | 23.05 | 30.40 | 3.53 | 0.65 | 0.79 | 0.94 |
| Qwen-Image-2512 | 23.16 | 30.79 | 3.40 | 0.72 | 0.81 | 0.87 |
| *RL-based* |  |  |  |  |  |  |
| FlowGRPO | 23.33 | 29.80 | 3.33 | 0.89 | 0.83 | 0.73 |
| DiffusionNFT | 23.63 | 31.79 | 3.39 | 0.89 | 0.91 | 0.86 |
| PromptRL w/o PE | 24.01 | 31.79 | 3.38 | 0.97 | 0.92 | 0.95 |
| **PromptRL w/ PE** | **24.05** | **32.03** | **3.44** | **0.98** | **0.91** | **0.95** |

</details>

<details>
<summary><b>✏️ Image Editing - EditReward (Full Results)</b></summary>

<br>

| Model | Swap | Style | Add. | Attr. | Env. | Removal | Avg.↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| InstructPix2Pix | -0.24 | 0.91 | -0.45 | 0.45 | 0.48 | -0.80 | 0.02 |
| MagicBrush | -0.38 | 0.36 | -0.78 | -0.80 | 0.91 | -0.85 | -0.27 |
| LEDITS++ | -0.81 | -0.32 | -0.30 | -0.60 | -0.37 | -0.97 | -0.60 |
| Qwen-Image-Edit | 1.11 | 1.14 | 0.95 | 0.90 | 1.39 | 0.61 | 1.03 |
| FLUX.2-klein | 1.42 | 1.73 | 1.29 | 1.42 | 1.80 | 0.32 | 1.34 |
| Nano Banana | 1.58 | 1.20 | 1.28 | 1.18 | 1.61 | 1.13 | 1.37 |
| Step1X-Edit | 1.39 | 1.58 | 1.19 | 1.34 | 1.57 | 0.22 | 1.24 |
| ReasonEdit | 1.51 | 1.43 | 1.19 | 1.47 | 1.58 | 1.14 | 1.40 |
| ReasonEdit-Think | 1.52 | 1.47 | 1.19 | 1.44 | 1.69 | 1.27 | 1.44 |
| FLUX.1-Kontext | 1.35 | 1.36 | 1.16 | 1.15 | 1.44 | 0.55 | 1.19 |
| FLUX.1-Kontext w/ PE | 1.35 | 0.97 | 1.04 | 0.48 | 1.22 | 0.65 | 1.01 |
| PromptRL w/o PE | 1.45 | 1.46 | 1.28 | 1.35 | 1.56 | 0.98 | 1.36 |
| **PromptRL w/ PE** | **1.47** | **1.43** | **1.29** | **1.39** | **1.72** | **1.24** | **1.43** |

</details>


## Method [TBD]

## Citation [TBD]

## Acknowledgments

This codebase builds upon [UniRL](https://github.com/G-U-N/UniRL/tree/master).