from datasets import load_dataset
import random
from tqdm import tqdm

# Parameters
total_files = 571  # Total number of Parquet files
files_to_sample = 10  # Number of files to select
total_samples = 3000  # Desired total samples
dataset_name = "TIGER-Lab/OmniEdit-Filtered-1.2M"
cache_dir = "./parquet_cache"  # Local cache directory
data_dir = "https://huggingface.co/datasets/TIGER-Lab/OmniEdit-Filtered-1.2M/resolve/main/data"  # Subfolder in dataset repo (adjust if needed)

# Step 1: Construct list of Parquet file paths
data_files = [f"{data_dir}/train-{i:05d}-of-{total_files:05d}.parquet" for i in range(total_files)]

# Step 2: Select Parquet files at roughly equal intervals
file_indices = [int(i * (total_files - 1) / (files_to_sample - 1)) for i in range(files_to_sample)]
selected_files = [data_files[i] for i in file_indices]
print(f"Selected files: {selected_files}")

# Step 3: Load selected Parquet files using datasets
i2i_dataset = load_dataset(
    "parquet",
    data_files=selected_files,
    split="train",
    num_proc=16,
    cache_dir=cache_dir
)

# Step 4: Randomly sample 3000 rows directly from the Dataset
# Calculate total number of rows and sample indices
total_rows = len(i2i_dataset)
sample_size = min(total_samples, total_rows)
sample_indices = random.sample(range(total_rows), sample_size)

# Select the sampled rows
sampled_dataset = i2i_dataset.select(sample_indices)

# Step 5: Save directly to Parquet
sampled_dataset.to_parquet('../assets/large_rl_datasets/random_sample_3000.parquet')

print(f"Sampled {len(sampled_dataset)} examples from {files_to_sample} Parquet files and saved to 'random_sample_3000.parquet'.")