
git clone https://github.com/TIGER-AI-Lab/EditReward.git
cd EditReward

conda create -n edit_reward python=3.10 -y
conda activate edit_reward
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install datasets pillow openai -U megfile sentencepiece deepspeed fire omegaconf matplotlib peft trl==0.8.6 tensorboard scipy transformers==4.56.1 accelerate
# Recommend: Install flash-attn
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl


download the EditReward weight locally following the official GitHub