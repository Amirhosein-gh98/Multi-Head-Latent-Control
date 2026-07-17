# Training Capability Heads

This document covers the global scalar Capability Head. The separate `when2call` pipeline trains a local four-class behavior head and has its own recipes under `when2call/receipes/`.

## Training data

`train_head_standalone_unsloth_regression_weighted_multimodel.py` accepts a Hugging Face dataset saved to disk, a directory of Parquet or JSON/JSONL shards, or a single Parquet/JSON/JSONL file.

The minimum logical fields are:

| Field | Purpose |
| --- | --- |
| `completion` or `answer` | Assistant output whose latent trajectory is scored. |
| `correctness_score` | Continuous training target, normally in `[0, 1]`. Configurable with `aux_label_column`. |
| `prompt`, `question`, or `text` | User-side input. The released verified schema uses `question`. |
| `system_prompt` | Optional system instruction. |
| `image` or `images` | Optional image input for multimodal examples. |
| `subset_name` | Optional source identifier used for filtering and analysis. |

The released data builder writes a richer verified schema, including source metadata, ground truth, judge outputs, component scores, `correctness_score`, and `correctness_label`. The trainer reconstructs the conversation from the system prompt, user prompt, optional image, and completion.

## Label construction

`combined_all_labeling_multimodel.py` applies task-specific exact, rule-based, or judge-based evaluators and writes a continuous `correctness_score`. It also derives `correctness_label` for diagnostics.

For the scalar head:

- the regression target is `correctness_score`;
- examples below `failure_threshold` are failures for weighting and failure metrics;
- examples at or above the threshold are successes;
- the checked-in recipes use `failure_threshold: 0.5`.

Do not substitute a binary label for the continuous score without deliberately changing the objective and recalibrating the output.

## Hidden-state input

`head_input_mode` controls the token mask passed to the head. Released full-trajectory checkpoints use `completion_text_only`, which excludes prompt, template/control, and vision tokens and retains the generated assistant text. Prefix variants use a completion-prefix mode and must be matched exactly.

Layer configuration follows two conventions:

- `hidden_layer_selection` uses transformer-layer indices: `first`, `middle`, `last`, `index`, `indices`, or `all`;
- legacy `selected_hidden_layer_indices` uses raw `hidden_states` indices, where index 0 is normally the embedding output.

Released lightweight heads select the last transformer layer. Never infer these settings from a repository name; read the checkpoint config or `capability_head_config.json`.

## Architecture

The scalar Capability Head is implemented by `AuxHeadModule` in `aux_head_shared_utils.py`.

1. A frozen backbone produces hidden states for the masked completion tokens.
2. `HiddenFeatureExtractorLite` normalizes and projects token states, applies gated dilated one-dimensional convolutions, Set Attention Blocks, and attention pooling, and returns a 256-dimensional representation.
3. `CorrectnessHeadLite` applies the hidden-state gate and an MLP ending in one logit.
4. A sigmoid maps the logit to a capability score in `[0, 1]`.

The code also supports stronger single-layer and multi-layer trajectory encoders. Multi-layer mode requires the full hidden-state tuple and explicit layer selection. The backbone parameters are frozen; only the auxiliary head is optimized.

## Objective and weighting

The checked-in global recipes use mean-squared error on the sigmoid score. Configuration includes:

- `regression_loss: mse`;
- automatic or fixed failure/success class weighting;
- optional severity weighting based on distance from the failure boundary;
- optional weighted sampling;
- gradient accumulation, clipping, cosine decay, warmup, and minimum learning-rate ratio.

Training logs regression MAE/RMSE, thresholded accuracy and macro metrics, and failure precision/recall/F1. Threshold metrics are monitoring aids; deployment thresholds must be calibrated separately.

## Train a released configuration

After generating and labeling data as described in [REPRODUCTION.md](REPRODUCTION.md), run a checked-in recipe:

```bash
python train_head_standalone_unsloth_regression_weighted_multimodel.py \
  --config recipes/training/qwen3_5_4b_think_off.yaml
```

Important recipe fields include:

```yaml
model_name_or_path: Qwen/Qwen3.5-4B
model_family: qwen3_5
thinking_mode: off
dataset_path: data/train/Qwen3.5/.../verified
output_dir: trained_models/...

dtype: bf16
max_seq_len: 32000
aux_label_column: correctness_score
head_input_mode: completion_text_only
hidden_encoder_type: lite
hidden_layer_selection: last
regression_loss: mse
failure_threshold: 0.5
```

The trainer saves periodic `aux_head_step<N>.pt` files and `aux_head_final.pt`. Each checkpoint contains the head state and serialized config, which should travel together.

## Evaluation during training

The current global trainer logs metrics on the optimization stream; it does not create an independent deployment calibration split automatically. For a new production head:

1. split data by original example or source before generating multiple completions;
2. reserve validation data for checkpoint selection;
3. reserve a separate calibration/test set for threshold selection and final reporting;
4. prevent near-duplicate prompts or trajectories from crossing splits;
5. report failure recall as well as aggregate regression metrics.

## Train for a new backbone

1. Confirm that the model can return hidden states and that the processor/chat template is supported by the shared utilities.
2. Add or select the correct `model_family` and `thinking_mode`; do not rely on automatic inference for a new naming convention.
3. Generate completions with the exact prompt template and thinking setting intended for deployment.
4. Label those completions and inspect the score distribution and failure balance.
5. Copy the closest recipe and change the backbone, dataset path, output path, dtype, sequence limit, token-mask mode, and layer selection deliberately.
6. Train the head while keeping the backbone frozen.
7. Validate checkpoint loading against the backbone hidden size and layer count.
8. Package `capability_head.pt` with sanitized metadata containing the exact base model and revision, architecture settings, label definition, and file checksum.
9. Calibrate routing thresholds on representative held-out traffic.

A head is backbone- and prompting-specific. Even if two models share the same hidden size, their heads are not assumed interchangeable.
