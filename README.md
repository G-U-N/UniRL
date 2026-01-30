```shell
conda env create -f environment.yml
conda activate unirl
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/huggingface/diffusers.git
pip install flash-attn==2.7.4.post1 --no-build-isolation
```