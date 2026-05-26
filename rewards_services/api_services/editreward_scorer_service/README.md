# EditReward Scorer Service

This service exposes EditReward over HTTP for edit GRPO training. It accepts a pickled payload with source images, edited images, and edit instructions, then returns `{"scores": [...]}`.

## Setup

```bash
cd rewards_services/api_services/editreward_scorer_service
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install -r requirements.txt
pip install flash-attn --no-build-isolation  # optional, recommended when your CUDA/PyTorch build supports it

git clone https://github.com/TIGER-AI-Lab/EditReward.git
huggingface-cli download TIGER-Lab/EditReward-MiMo-VL-7B-SFT-2508 \
  --local-dir EditReward/EditReward-MiMo-VL-7B-SFT-2508
```

If the repository or checkpoint lives elsewhere, set:

```bash
export EDITREWARD_REPO_DIR=/path/to/EditReward
export EDITREWARD_CHECKPOINT_PATH=/path/to/EditReward-MiMo-VL-7B-SFT-2508
```

## Run

```bash
export EDITREWARD_PORT=18088
export EDITREWARD_CUDA_DEVICES=0,1
export EDITREWARD_WORKERS=2
bash run.sh
```
