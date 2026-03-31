#!/bin/bash
set -e

# Download eval datasets if not present
EDIT_DATA="data/omni_edit_dev.parquet"
if [ ! -f "$EDIT_DATA" ]; then
    echo "Downloading edit eval dataset..."
    mkdir -p data
    huggingface-cli download wangfuyun/PrompRL data/omni_edit_dev.parquet \
        --repo-type model --local-dir . --local-dir-use-symlinks False
fi

# # Text-to-Image OCR
python unified_inference.py --mode t2i \
    --model_path wangfuyun/PrompRL/promptrl_ocr \
    --model_type flux \
    --prompt_file prompts/ocr_test.txt \
    --output_dir outputs/ocr \
    --use_cot --cot_template ocr_clarity_v2

# # Text-to-Image PS
python unified_inference.py --mode t2i \
    --model_path wangfuyun/PrompRL/promptrl_ps \
    --model_type flux \
    --prompt_file prompts/draw_test.txt \
    --output_dir outputs/pickscore \
    --use_cot --cot_template quality_purev2

# # GenEval
python unified_inference.py --mode geneval \
    --model_path wangfuyun/PrompRL/promptrl_geneval \
    --model_type flux \
    --metadata_file prompts/evaluation_metadata.jsonl \
    --output_dir outputs/geneval \
    --use_cot --cot_template geneval \
    --n_samples 4

# # Image Editing
python unified_inference.py --mode edit \
    --model_path wangfuyun/PrompRL/promptrl_edit \
    --model_type kontext \
    --data_file "$EDIT_DATA" \
    --output_dir outputs/edit \
    --use_cot --cot_template edit_general \
    --guidance_scale 2.5


# python eval.py prompts/config.yaml
