source $(conda info --base)/etc/profile.d/conda.sh


conda activate edit_reward

/home/ubuntu/miniconda3/envs/edit_reward/bin/gunicorn -c gunicorn.conf.py "app:create_app()"
