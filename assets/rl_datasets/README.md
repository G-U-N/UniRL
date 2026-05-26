# Edit Training Dataset Schema

The training script defaults to:

```text
https://huggingface.co/wangfuyun/PrompRL/resolve/main/data/omni_edit_train_50k.parquet
```

Use a `.parquet` or `.jsonl` file with at least:

| Column | Type | Description |
| --- | --- | --- |
| `image` | PIL image, image bytes, or image path | Source image before editing. |
| `prompt` | string | Edit instruction used by FLUX.1-Kontext and EditReward. |

Optional columns:

| Column | Type | Description |
| --- | --- | --- |
| `caption` | string | Source-image caption, kept for logging or downstream reward extensions. |
| `target_caption` | string | Target edited-image caption, kept for logging or downstream reward extensions. |

If your dataset uses different column names, pass `IMAGE_COLUMN=...` and `PROMPT_COLUMN=...` to `scripts/train/edit_grpo.sh`.
