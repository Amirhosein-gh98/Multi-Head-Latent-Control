# Global Head Runbook

## Pipeline

1. generate raw completions with `combined_all_datagen_multimodel.py`
2. label correctness with `combined_all_labeling_multimodel.py`
3. train the head with `train_head_standalone_unsloth_regression_weighted_multimodel.py`

## Minimal Example

```bash
python combined_all_datagen_multimodel.py \
  --model-id Qwen/Qwen3.5-2B \
  --model-family qwen3_5 \
  --thinking-mode on \
  --run-name Qwen3_5_2B_think_on_hard_Mixed_Sources_120k \
  --save-root data/train/Qwen3.5/Qwen3_5_2B_think_on_hard_Mixed_Sources_120k

python combined_all_labeling_multimodel.py \
  --run-root data/train/Qwen3.5/Qwen3_5_2B_think_on_hard_Mixed_Sources_120k \
  --judge-model-id Qwen/Qwen3-VL-30B-A3B-Instruct-FP8

python train_head_standalone_unsloth_regression_weighted_multimodel.py \
  --config recipes/training/qwen3_5_2b_think_on.yaml
```

## Available Recipes

Training configs are stored under:

```text
recipes/training/
recipes/training/qwen3vl/
```

They cover Qwen3.5, Qwen3-VL, and Gemma 4 variants.
