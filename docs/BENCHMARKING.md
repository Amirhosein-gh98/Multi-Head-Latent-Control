# Benchmarking and Routing Evaluation

The benchmark pipeline compares small-only, strong-only, and capability-routed inference under common datasets and accounting rules.

## Supported datasets

`multi_agenT_bench/run_multi_agent_generate_then_eval.py` accepts:

- `mathvista`
- `mathverse`
- `charxiv_reasoning`
- `screenspot_pro`
- `simplevqa`
- `triviaqa`
- `math`
- `mmlu_pro`

Most modes download their dataset from Hugging Face. Two modes require local CSV files:

```text
math       -> data/benchmarks/merged_math.csv
mmlu_pro   -> data/benchmarks/test.csv
```

Those CSVs and their preparation scripts are not included in this repository. Use a Hugging Face-backed mode such as `triviaqa`, `mathvista`, or `simplevqa` unless you have the paper snapshot.

## Routing configurations

The benchmark runtime supports policies including:

- `single_agent_model1`: small model only;
- `single_agent_model2`: strong model only;
- small-model self-repair or retry below threshold;
- fresh handoff from the small model to the strong model;
- handoff with context;
- full-completion and prefix-triggered decisions.

The principal paper-style comparison uses the two single-model baselines and `m1_after_finish_handoff_fresh_m2`. A score below the configured threshold triggers the route. Sweep thresholds rather than reporting only 0.5.

## End-to-end command

First download the released checkpoint:

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download

print(hf_hub_download(
    repo_id="AmirhoseinGH/mhlc-capability-head-qwen3vl-2b-thinking",
    filename="capability_head.pt",
))
PY
```

Pass the printed path to the benchmark driver:

```bash
python multi_agenT_bench/run_multi_agent_generate_then_eval.py \
  --benchmark triviaqa \
  --model1_name_or_path Qwen/Qwen3-VL-2B-Thinking \
  --model2_name_or_path Qwen/Qwen3-VL-32B-Thinking-FP8 \
  --model1_aux_head_ckpt /path/to/capability_head.pt \
  --model1_model_family qwen3_vl \
  --model1_thinking_mode on \
  --model2_model_family qwen3_vl \
  --model2_thinking_mode on \
  --model1_params 2 \
  --model2_params 32 \
  --strategy_names single_agent_model1,single_agent_model2,m1_after_finish_handoff_fresh_m2 \
  --model1_aux_thresholds 0.5,0.6,0.7,0.8,0.9 \
  --model2_aux_threshold 0.80 \
  --judge_batch_size 32 \
  --output_base_root eval_outputs/multi_agent_compact
```

The head must match model 1 exactly. `model1_params` and `model2_params` are used for approximate FLOP summaries; use one consistent parameter-count convention across all comparisons.

The driver generates responses first and then evaluates them. It supports resuming completed work. Use `--overwrite` only when intentionally replacing outputs, `--no_resume` to disable resume behavior, and `--debug` for a reduced diagnostic run.

## Metrics

Depending on the dataset, evaluation uses exact/task-specific correctness or judge-scored correctness. The evaluator records:

- the benchmark's primary metric, commonly accuracy;
- route, handoff, repair, refusal, and failure counts;
- prompt and completion tokens by model;
- tokens scored by the auxiliary head;
- generation and auxiliary call counts;
- judge token use;
- total compute-token touches;
- correct examples per 1,000 generation or compute tokens;
- deltas and token savings relative to a baseline;
- approximate FLOPs when parameter counts are supplied.

Treat token-based compute and parameter-count FLOP estimates as proxies. For production cost reporting, apply provider- and model-specific input/output prices to the per-model token counts. For latency reporting, measure wall-clock latency on the deployment hardware and include queueing, generation, head replay, and handoff overhead.

## Threshold selection

For each candidate threshold, report at least:

1. primary task quality;
2. escalation rate;
3. small- and large-model token totals;
4. auxiliary scoring overhead;
5. average and tail latency when available;
6. monetary cost using the intended serving prices.

Select a threshold on validation traffic under an explicit constraint, such as maximum cost at a target accuracy or maximum escalation rate at a target failure recall. Freeze that threshold before evaluating the test split.

## Outputs

The runner creates a configuration-specific directory beneath `--output_base_root`. Expected artifacts include:

- `run_manifest.json` with models, modes, thresholds, scripts, and runtime settings;
- generated per-example rows for each strategy and threshold;
- per-strategy summaries;
- threshold and baseline-comparison summaries;
- CSV/JSON tables containing task, routing, token, and compute fields.

Keep the entire result directory. Tables and figures should be generated from the saved summaries, not copied from terminal output.

## Reproducing paper tables and figures

1. Use the exact paper data snapshot, backbone revisions, head checkpoint, prompt/thinking modes, sampling profiles, and judge revision.
2. Run both small-only and strong-only baselines in the same environment as the routed policies.
3. Run the full paper threshold grid.
4. Confirm that every strategy has the same example count and stable example identifiers.
5. Use the generated summary CSV/JSON files to assemble quality-versus-cost tables and plots.
6. Preserve run manifests and report any deviation from paper hardware, prices, datasets, or package versions.

The Android World experiment additionally requires the upstream [Mobile-Agent Android World implementation](https://github.com/X-PLUG/MobileAgent/tree/main/Mobile-Agent-v3.5/android_world_v3.5), Android SDK/device setup, and the model services described by that project. This repository supplies verifier proxies under `inference/qwen3vl/android_world/`, but not the full Android environment.
