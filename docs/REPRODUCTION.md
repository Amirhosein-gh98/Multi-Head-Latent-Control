# Research Reproduction

This guide describes the repository's paper pipeline. For pretrained inference, start with the root [README](../README.md). For training details and benchmark interpretation, see [TRAINING.md](TRAINING.md) and [BENCHMARKING.md](BENCHMARKING.md).

## Reproducibility boundary

The Git repository contains source code and training recipes. It does not contain downloaded backbones, raw or processed training data, trained run directories, or benchmark outputs. Released Capability Head weights are available in the [Hugging Face collection](https://huggingface.co/collections/AmirhoseinGH/multi-head-latent-control-capability-heads).

Exact paper reproduction also requires the frozen data snapshots and environment versions used for the paper. Regenerating from current upstream datasets or model revisions may produce different samples and scores. Record Hugging Face revisions, package versions, GPU type, and random seeds for every new run.

## Environment

Use a mutually supported Python, CUDA, PyTorch, Unsloth, and vLLM stack on a CUDA machine. Install the CUDA-compatible PyTorch build first, then:

```bash
git clone https://github.com/Amirhosein-gh98/Multi-Head-Latent-Control.git
cd Multi-Head-Latent-Control
pip install -r requirements.txt
```

Optional math-verification paths also use:

```bash
pip install latex2sympy2-extended math-verify
```

Record the environment before a paper run:

```bash
python --version
nvidia-smi
pip freeze > environment-freeze.txt
```

The PyTorch, CUDA, Transformers, Unsloth, vLLM, and Flash Attention versions must be compatible. Training recipes enable Weights & Biases unless `wandb_enabled: false` is set; use `wandb login` or `WANDB_MODE=offline` as appropriate.

Run all commands below from the repository root. The expected workspace is:

```text
data/
  train/
  benchmarks/
trained_models/
eval_outputs/
```

## Global Capability Head pipeline

The example below reproduces the stages for a Qwen3.5 4B, thinking-off head. Adjust the model, thinking mode, output path, and recipe together; mixing artifacts between modes is invalid.

### 1. Generate candidate completions

```bash
python combined_all_datagen_multimodel.py \
  --model-id Qwen/Qwen3.5-4B \
  --model-family qwen3_5 \
  --thinking-mode off \
  --run-name Qwen3_5_4B_think_off_hard_Mixed_Sources_120k \
  --save-root data/train/Qwen3.5/Qwen3_5_4B_think_off_hard_Mixed_Sources_120k \
  --seed 42
```

Expected outputs:

```text
data/train/Qwen3.5/Qwen3_5_4B_think_off_hard_Mixed_Sources_120k/
  raw/*.parquet
  selection_manifest.json
  generation_stats.json
```

The generator draws from the mixed sources configured in `combined_all_datagen_multimodel.py`. Preserve the selection manifest to identify the exact source mixture and resolved model settings.

### 2. Construct correctness labels

```bash
python combined_all_labeling_multimodel.py \
  --run-root data/train/Qwen3.5/Qwen3_5_4B_think_off_hard_Mixed_Sources_120k \
  --judge-model-id Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
```

Expected outputs:

```text
data/train/Qwen3.5/Qwen3_5_4B_think_off_hard_Mixed_Sources_120k/
  verified/*.parquet
  verification_stats.json
```

The labeler resumes by shard. Do not combine raw and verified shards from different generation configurations.

### 3. Train the Capability Head

The checked-in recipe points at the expected verified dataset directory:

```bash
python train_head_standalone_unsloth_regression_weighted_multimodel.py \
  --config recipes/training/qwen3_5_4b_think_off.yaml
```

Expected outputs under the recipe's `output_dir` include:

```text
aux_head_step1000.pt
aux_head_step2000.pt
...
aux_head_final.pt
```

The final checkpoint is a PyTorch dictionary containing `step`, `head_state`, and the serialized training `cfg`. The processor is also saved to the output directory.

### 4. Run model-routing benchmarks

Download the released head or use the locally trained `aux_head_final.pt`, then run the benchmark driver. For example:

```bash
python multi_agenT_bench/run_multi_agent_generate_then_eval.py \
  --benchmark triviaqa \
  --model1_name_or_path Qwen/Qwen3-VL-2B-Thinking \
  --model2_name_or_path Qwen/Qwen3-VL-32B-Thinking-FP8 \
  --model1_aux_head_ckpt trained_models/Qwen3VL-2B_Thinking_120K_lite/aux_head_final.pt \
  --model1_model_family qwen3_vl \
  --model1_thinking_mode on \
  --model2_model_family qwen3_vl \
  --model2_thinking_mode on \
  --strategy_names single_agent_model1,single_agent_model2,m1_after_finish_handoff_fresh_m2 \
  --model1_aux_thresholds 0.5,0.6,0.7,0.8,0.9 \
  --model2_aux_threshold 0.80 \
  --output_base_root eval_outputs/multi_agent_compact
```

The driver writes a run manifest, generation outputs, per-strategy summaries, threshold summaries, and comparison tables beneath the output root. See [BENCHMARKING.md](BENCHMARKING.md) for dataset requirements and metric definitions.

## Local When2Call pipeline

The local control head learns four behaviors from `nvidia/When2Call`.

```bash
python when2call/when2call_build_head_labels_4class.py \
  --dataset_id nvidia/When2Call \
  --splits train_sft train_pref \
  --output_dir data/train/when2call/when2call_processed_4class \
  --model_id Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --tokenizer_id Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --dtype auto \
  --batch_size 256 \
  --max_tokens 16000 \
  --gpu_memory_utilization 0.50 \
  --tensor_parallel_size 1 \
  --max_model_len 32000 \
  --seed 42 \
  --resume \
  --export_parquet

python when2call/when2call_generate_completions_4class.py \
  --input_path data/train/when2call/when2call_processed_4class/when2call_aux_labels.jsonl \
  --output_dir data/train/when2call/qwen3vl/Qwen3-VL-2B-Instruct_4class \
  --model_id Qwen/Qwen3-VL-2B-Instruct \
  --model_family qwen3_vl \
  --thinking_mode off \
  --batch_size 64 \
  --max_tokens 16000 \
  --gpu_memory_utilization 0.90 \
  --resume

python when2call/train_when2call_head_4class_3sigmoid.py \
  --config when2call/receipes/train_head_Qwen3-VL-2B-Instruct_4class.yaml
```

The `receipes` directory name is intentionally retained because existing commands depend on it.

## Reproduction checklist

- Use the exact backbone repository and revision recorded by the run.
- Match model family and thinking mode across generation, training, head scoring, and benchmark evaluation.
- Use the checkpoint's token-mask mode and hidden-layer selection.
- Keep source manifests, seeds, recipe YAML, environment freeze, and GPU details with the results.
- Compare generated counts and shard counts with `generation_stats.json` and `verification_stats.json` before training.
- Compare the reported task metric and routing/cost fields, not task accuracy alone.

Paper values are configuration-specific and should not be treated as universal gains for other traffic distributions or thresholds.
