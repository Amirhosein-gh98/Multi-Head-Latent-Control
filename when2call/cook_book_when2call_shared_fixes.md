# When2Call Runbook

This file is a compact runbook for the local `when2call` head. See `README.md` for the full repository overview.

## Pipeline

1. build 4-class labels with `when2call_build_head_labels_4class.py`
2. generate model completions with `when2call_generate_completions_4class.py`
3. train a head with one of the configs in `when2call/receipes/`
4. generate eval completions
5. run head-only and model-only evaluation

## Minimal Example

```bash
python when2call/when2call_build_head_labels_4class.py \
  --dataset_id nvidia/When2Call \
  --splits train_sft train_pref \
  --output_dir data/train/when2call/when2call_processed_4class \
  --model_id Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --tokenizer_id Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --resume \
  --export_parquet

python when2call/when2call_generate_completions_4class.py \
  --input_path data/train/when2call/when2call_processed_4class/when2call_aux_labels.jsonl \
  --output_dir data/train/when2call/qwen3vl/Qwen3-VL-2B-Instruct_4class \
  --model_id Qwen/Qwen3-VL-2B-Instruct \
  --model_family qwen3_vl \
  --thinking_mode off \
  --resume

python when2call/train_when2call_head_4class_3sigmoid.py \
  --config when2call/receipes/train_head_Qwen3-VL-2B-Instruct_4class.yaml
```

## Evaluation Example

```bash
python when2call/eval/generate_when2call_eval_completions_4class.py \
  --model_id Qwen/Qwen3-VL-2B-Instruct \
  --output_path eval_outputs/when2call/Qwen3-VL-2B-Instruct/when2call_test_generated_4class.parquet

python when2call/eval/eval_when2call_head_only_4class_3sigmoid.py \
  --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
  --head_checkpoint_path trained_models/Qwen3-VL-2B-Instruct_When2call_4class/head-final.pt \
  --generated_eval_path eval_outputs/when2call/Qwen3-VL-2B-Instruct/when2call_test_generated_4class.parquet \
  --output_dir eval_outputs/when2call/Qwen3-VL-2B-Instruct/head_only_eval_4class

python when2call/eval/eval_when2call_model_only_judge_4class.py \
  --generated_eval_path eval_outputs/when2call/Qwen3-VL-2B-Instruct/when2call_test_generated_4class.parquet \
  --output_dir eval_outputs/when2call/Qwen3-VL-2B-Instruct/model_only_judge_eval_4class
```

## Notes

- The `receipes/` folder name is kept unchanged to match existing script expectations.
- Swap the recipe file and output paths to run the other Qwen, Gemma 4, or Qwen3.5 variants.
