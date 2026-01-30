# PromptRL: Prompt Matters in RL for Flow-Based Image Generation



## Overview

**PromptRL** is a framework that jointly trains language models (LMs) and flow-matching models (FMs) within a unified reinforcement learning loop for text-to-image generation. By incorporating LMs as adaptive prompt refiners, PromptRL addresses two critical limitations in current flow-based RL pipelines: *exploration collapse* due to insufficient generation diversity, and *prompt overfitting* where models memorize specific training formulations.

### Key Results

<div align="center">

| Benchmark | Score |
|:---:|:---:|
| GenEval | **0.97** |
| OCR Accuracy | **0.98** |
| PickScore | **24.05** |
| EditReward | **1.43** |

</div>

PromptRL achieves **2× sample efficiency** compared to flow-only RL while maintaining robust generalization to diverse prompt formulations.

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
  <img src="assets/edit_comparison.png" width="85%">
</p>

## Method [TBD]
## Citation [TBD]

## Acknowledgments

This codebase builds upon [UniRL](https://github.com/G-U-N/UniRL/tree/master).