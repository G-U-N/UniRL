source $(conda info --base)/etc/profile.d/conda.sh


conda activate aes

/home/ubuntu/miniconda3/envs/aes/bin/gunicorn -c gunicorn.conf.py "app:create_app()"

# pkill gunicorn to stop