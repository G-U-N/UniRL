from datasets import load_dataset
import random
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

print("Loading dataset in streaming mode...")
ds = load_dataset("TIGER-Lab/OmniEdit-Filtered-1.2M", split="train", streaming=True)

print("Collecting data for sampling...")
all_data = []
for i, example in enumerate(ds):
    all_data.append(example)
    if (i + 1) % 10000 == 0:
        print(f"Collected {i + 1} examples...")

print(f"Total examples collected: {len(all_data)}")

sample_size = 3000
sampled_indices = random.sample(range(len(all_data)), min(sample_size, len(all_data)))
sampled_data = [all_data[i] for i in sampled_indices]

df = pd.DataFrame(sampled_data)
print(f"Sampled DataFrame shape: {df.shape}")
print("Sample columns:", df.columns.tolist())
print("First few rows preview:")
print(df.head())

output_file = "sampled_omni_edit_3k.parquet"
table = pa.Table.from_pandas(df)
pq.write_table(table, output_file)
print(f"Saved sampled data to {output_file}")