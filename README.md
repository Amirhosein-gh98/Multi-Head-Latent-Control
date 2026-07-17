<div align="center">

# Multi-Head Latent Control

### Capability-aware routing from a model's own hidden states

[![PyTorch](https://img.shields.io/badge/PyTorch-Capability%20Heads-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Models](https://img.shields.io/badge/Models-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/collections/AmirhoseinGH/multi-head-latent-control-capability-heads)
[![Paper](https://img.shields.io/badge/arXiv-2607.14277-B31B1B?logo=arxiv)](https://arxiv.org/abs/2607.14277)

**Let a small model answer easy requests and escalate only when its latent state says it needs help.**

</div>

![Multi-Head Latent Control overview](assets/main_fig_v3.png)

Multi-Head Latent Control (MHLC) adds lightweight control heads to frozen language models. The **Capability Head** reads the hidden-state trajectory of a generated answer and returns a score from 0 to 1: higher means the current model is more likely to be capable of handling the request. A routing policy can accept the answer or send the request to a stronger model.

This makes it practical to build:

- small-to-large model routing and model cascades;
- selective escalation for difficult requests;
- lower-cost, lower-latency inference on easy traffic;
- collaborative inference systems that retry, repair, use tools, or hand off;
- confidence-aware abstention and agent control.

The backbone stays frozen. A released head is only a few MiB and can be evaluated after generation without fine-tuning or replacing the language model.

## How routing works

1. Generate an answer and get its hidden states.
2. Pass the hidden states to the Capability Head to predict an adequacy score from 0 to 1.
3. Accept the answer when the score is above your threshold; otherwise route the request to a stronger model.

Each head must be used with its matching backbone and thinking mode.

## Pretrained Capability Heads

All weights are in the [Multi-Head Latent Control Capability Heads collection](https://huggingface.co/collections/AmirhoseinGH/multi-head-latent-control-capability-heads). Each repository contains `capability_head.pt` and `capability_head_config.json`.

| Backbone | Mode / variant | Capability Head |
| --- | --- | --- |
| `Qwen/Qwen3-VL-2B-Thinking` | thinking, full trajectory | [`AmirhoseinGH/mhlc-capability-head-qwen3vl-2b-thinking`](https://huggingface.co/AmirhoseinGH/mhlc-capability-head-qwen3vl-2b-thinking) |
| `Qwen/Qwen3-VL-4B-Instruct` | instruct, full trajectory | [`AmirhoseinGH/mhlc-capability-head-qwen3vl-4b-instruct`](https://huggingface.co/AmirhoseinGH/mhlc-capability-head-qwen3vl-4b-instruct) |
| `Qwen/Qwen3-VL-4B-Thinking` | thinking, full trajectory | [`AmirhoseinGH/mhlc-capability-head-qwen3vl-4b-thinking`](https://huggingface.co/AmirhoseinGH/mhlc-capability-head-qwen3vl-4b-thinking) |
| `Qwen/Qwen3.5-4B` | thinking off, full trajectory | [`AmirhoseinGH/mhlc-capability-head-qwen35-4b`](https://huggingface.co/AmirhoseinGH/mhlc-capability-head-qwen35-4b) |
| `Qwen/Qwen3.5-9B` | thinking off, full trajectory | [`AmirhoseinGH/mhlc-capability-head-qwen35-9b`](https://huggingface.co/AmirhoseinGH/mhlc-capability-head-qwen35-9b) |
| `google/gemma-4-E4B-it` | instruct, full trajectory | [`AmirhoseinGH/mhlc-capability-head-gemma4-e4b-instruct`](https://huggingface.co/AmirhoseinGH/mhlc-capability-head-gemma4-e4b-instruct) |
| `google/gemma-4-E4B-it` | thinking, full trajectory | [`AmirhoseinGH/mhlc-capability-head-gemma4-e4b-thinking`](https://huggingface.co/AmirhoseinGH/mhlc-capability-head-gemma4-e4b-thinking) |
| `Qwen/Qwen3-VL-32B-Instruct-FP8` | instruct, full trajectory | [`AmirhoseinGH/mhlc-capability-head-qwen3vl-32b-instruct-step10000`](https://huggingface.co/AmirhoseinGH/mhlc-capability-head-qwen3vl-32b-instruct-step10000) |
| `Qwen/Qwen3-VL-4B-Thinking` | lightweight, full trajectory | [`AmirhoseinGH/mhlc-capability-head-qwen3vl-4b-thinking-lite`](https://huggingface.co/AmirhoseinGH/mhlc-capability-head-qwen3vl-4b-thinking-lite) |
| `Qwen/Qwen3-VL-2B-Thinking` | first 200 completion tokens | [`AmirhoseinGH/mhlc-capability-head-qwen3vl-2b-thinking-prefix200`](https://huggingface.co/AmirhoseinGH/mhlc-capability-head-qwen3vl-2b-thinking-prefix200) |
| `Qwen/Qwen3-VL-4B-Thinking` | first 200 completion tokens | [`AmirhoseinGH/mhlc-capability-head-qwen3vl-4b-thinking-prefix200`](https://huggingface.co/AmirhoseinGH/mhlc-capability-head-qwen3vl-4b-thinking-prefix200) |
| `google/gemma-4-E4B-it` | thinking, first 200 completion tokens | [`AmirhoseinGH/mhlc-capability-head-gemma4-e4b-thinking-prefix200`](https://huggingface.co/AmirhoseinGH/mhlc-capability-head-gemma4-e4b-thinking-prefix200) |

**Prefix heads** are trained to predict model adequacy from a partial answer—the first 200 generated tokens—so routing can happen before the full answer is finished.

Download a checkpoint programmatically:

```python
from huggingface_hub import hf_hub_download

checkpoint_path = hf_hub_download(
    repo_id="AmirhoseinGH/mhlc-capability-head-qwen3vl-2b-thinking",
    filename="capability_head.pt",
)
```

## Installation

Use a Python version supported by your CUDA, PyTorch, Unsloth, and vLLM stack (Python 3.10+). Then install the project dependencies:

```bash
git clone https://github.com/Amirhosein-gh98/Multi-Head-Latent-Control.git
cd Multi-Head-Latent-Control

# Install a CUDA-matched torch build first.
pip install -r requirements.txt
```

The inference path uses PyTorch, Transformers, Hugging Face Hub, Unsloth, and vLLM. Flash Attention, CUDA, PyTorch, Transformers, Unsloth, and vLLM versions must be mutually compatible. Gemma checkpoints may require accepting the model license on Hugging Face.

## Minimal end-to-end routing example

This example generates an answer with Qwen3-VL-2B, extracts its hidden-state signal through the existing runtime, and uses the matching Capability Head to decide whether to accept or escalate.

```python
from huggingface_hub import hf_hub_download

from multi_agenT_bench.compact_multi_agent_shared_optimized_v4_textbench import (
    AuxHeadRuntime,
    AuxHeadRuntimeConfig,
    SamplingConfig,
    VLLMChatRuntime,
    VLLMRuntimeConfig,
)

BACKBONE = "Qwen/Qwen3-VL-2B-Thinking"
HEAD_REPO = "AmirhoseinGH/mhlc-capability-head-qwen3vl-2b-thinking"
THRESHOLD = 0.5
PROMPT = "A shop sells 3 notebooks for $12. What is the price of 7 notebooks?"

def main():
    checkpoint_path = hf_hub_download(HEAD_REPO, "capability_head.pt")

    # Generate an answer with the small model.
    generator = VLLMChatRuntime(VLLMRuntimeConfig(
        model_name_or_path=BACKBONE,
        max_model_len=4096,
        gpu_memory_utilization=0.45,
        max_num_seqs=1,
    ))
    generation = generator.generate(
        messages=[{"role": "user", "content": PROMPT}],
        image=None,
        sampling_cfg=SamplingConfig(greedy=True, max_new_tokens=256),
    )
    generator.unload(drop_processor=True)

    # Score the answer with its matching Capability Head.
    router = AuxHeadRuntime(AuxHeadRuntimeConfig(
        enabled=True,
        model_name_or_path=BACKBONE,
        aux_head_ckpt=checkpoint_path,
        regression_threshold=THRESHOLD,
        head_input_mode="completion_text_only",
    ))
    router.load()
    result = router.score_single(
        prompt_text=PROMPT,
        image=None,
        response_text=generation.text,
    )

    score = result.prob_correct
    should_escalate = score < THRESHOLD

    print(f"Prompt: {PROMPT}")
    print(f"Generated answer: {generation.text}")
    print(f"Capability score: {score:.6f}")
    print(f"Threshold: {THRESHOLD:.2f}")
    print(f"Routing decision: {'ESCALATE' if should_escalate else 'ACCEPT'}")

    if should_escalate:
        print("Route the original prompt to the configured stronger model.")


if __name__ == "__main__":
    main()
```

Save the example as a `.py` file and run it from the repository root. The `main()` guard is required by vLLM multiprocessing. A ready-to-submit one-GPU Slurm version using the `unsloth_vllm` conda environment is in [docs/VERIFICATION.md](docs/VERIFICATION.md).

`AuxHeadRuntime` handles hidden-state extraction and returns the capability score as `prob_correct`.

## Choosing a threshold

`0.5` is a smoke-test default, not a universal operating point. Calibrate the threshold on held-out traffic from the target deployment:

- raise it to escalate more often and favor quality;
- lower it to accept more small-model answers and favor cost or latency;
- report both task quality and routing/cost metrics when comparing policies.

## Research and reproduction

The paper workflow is kept separate from the product quick start:

- [Reproduction guide](docs/REPRODUCTION.md) — environment, data generation, labeling, paper pipeline, and expected artifacts.
- [Training guide](docs/TRAINING.md) — dataset schema, labels, architecture, losses, configs, checkpoints, and new backbones.
- [Benchmarking guide](docs/BENCHMARKING.md) — datasets, routing strategies, thresholds, metrics, cost analysis, and result files.
- [Slurm verification](docs/VERIFICATION.md) — one-GPU smoke test for the example above.

## Repository layout

```text
aux_head_shared_utils.py                 # token masks, head wrapper, shared loading helpers
feature_extractors.py                    # hidden-trajectory encoders and prediction head
multi_agenT_bench/                       # generation, routing, and benchmark evaluation
recipes/training/                        # global Capability Head recipes
when2call/                               # local four-class control-head pipeline
inference/                               # inference-time services
docs/                                    # research, training, benchmarking, and verification guides
```

The existing directory name `multi_agenT_bench` and the `when2call/receipes` spelling are retained for compatibility.

## Citation

```bibtex
@misc{ghasemabadi2026multiheadlatentcontrol,
  title         = {Multi-Head Latent Control: A Unified Interface for LLM Agent Decision Making},
  author        = {Amirhosein Ghasemabadi and Ruichen Chen and Bahador Rashidi and Di Niu},
  year          = {2026},
  eprint        = {2607.14277},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2607.14277}
}
```
