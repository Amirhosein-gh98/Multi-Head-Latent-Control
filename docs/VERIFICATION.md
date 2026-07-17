# Slurm Verification

This smoke test exercises the exact minimal example from the root README with one GPU. It uses:

- backbone: `Qwen/Qwen3-VL-2B-Thinking`;
- head: `AmirhoseinGH/mhlc-capability-head-qwen3vl-2b-thinking`;
- conda environment: `unsloth_vllm`;
- deterministic greedy generation;
- a 0.5 routing threshold;
- metadata, non-empty-generation, and score-range assertions.

Save the following as `verify_capability_routing.slurm` outside the repository or in a local ignored jobs directory. The job writes the example to a temporary `.py` file because vLLM multiprocessing cannot launch from `<stdin>`.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=mhlc-smoke
#SBATCH --output=logs/mhlc-smoke-%j.out
#SBATCH --gres=gpu:1
#SBATCH --partition=LocalQ
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/amirhoseingh/codes/Multi-Head-Latent-Control}"
CONDA_SH="/data00/amirhoseingh/miniconda/etc/profile.d/conda.sh"
VERIFY_SCRIPT="${SLURM_TMPDIR:-/tmp}/verify_mhlc_example_${SLURM_JOB_ID}.py"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"
source "${CONDA_SH}"
conda activate unsloth_vllm
unset LD_LIBRARY_PATH

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/tmp/${USER}/huggingface}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# vLLM uses multiprocessing, so run a real Python file rather than stdin.
cat > "${VERIFY_SCRIPT}" <<'PY'
import json
import random

import torch
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


def main() -> None:
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    assert torch.cuda.is_available(), "This verification requires a CUDA GPU"

    metadata_path = hf_hub_download(HEAD_REPO, "capability_head_config.json")
    checkpoint_path = hf_hub_download(HEAD_REPO, "capability_head.pt")
    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)

    assert metadata["base_model"] == BACKBONE, (
        f"Head expects {metadata['base_model']}, not {BACKBONE}"
    )
    architecture = metadata["architecture"]
    assert architecture["head_input_mode"] == "completion_text_only"
    assert architecture["hidden_layer_selection"] == "last"
    assert metadata["weight_file"] == "capability_head.pt"

    generator = VLLMChatRuntime(VLLMRuntimeConfig(
        model_name_or_path=BACKBONE,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.45,
        max_num_seqs=1,
        enforce_eager=True,
        trust_remote_code=True,
        model_family="qwen3_vl",
        thinking_mode="on",
    ))
    generation = generator.generate(
        messages=[{"role": "user", "content": PROMPT}],
        image=None,
        sampling_cfg=SamplingConfig(greedy=True, max_new_tokens=256),
    )
    generator.unload(drop_processor=True)
    torch.cuda.empty_cache()

    router = AuxHeadRuntime(AuxHeadRuntimeConfig(
        enabled=True,
        model_name_or_path=BACKBONE,
        aux_head_ckpt=checkpoint_path,
        dtype=architecture["dtype"],
        max_seq_len=4096,
        attn_implementation="flash_attention_3",
        regression_threshold=THRESHOLD,
        head_input_mode=architecture["head_input_mode"],
        hidden_layer_selection=architecture["hidden_layer_selection"],
        model_family="qwen3_vl",
        thinking_mode="on",
    ))
    router.load()
    result = router.score_single(
        prompt_text=PROMPT,
        image=None,
        response_text=generation.text,
    )

    score = result.prob_correct
    should_escalate = score < THRESHOLD
    assert generation.text.strip(), "The backbone generated an empty answer"
    assert 0.0 <= score <= 1.0, f"Invalid capability score: {score}"
    assert result.pred == int(not should_escalate), (
        f"Runtime decision mismatch: pred={result.pred}, score={score}"
    )

    print(f"Prompt: {PROMPT}")
    print(f"Generated answer: {generation.text}")
    print(f"Capability score: {score:.6f}")
    print(f"Threshold: {THRESHOLD:.2f}")
    print(f"Routing decision: {'ESCALATE' if should_escalate else 'ACCEPT'}")
    print("VERIFICATION PASSED")


if __name__ == "__main__":
    main()
PY

python "${VERIFY_SCRIPT}"
```

Submit it from the repository root so the relative log directory is predictable:

```bash
cd /home/amirhoseingh/codes/Multi-Head-Latent-Control
mkdir -p logs
sbatch /path/to/verify_capability_routing.slurm
```

Monitor the result:

```bash
squeue -j JOB_ID
tail -f logs/mhlc-smoke-JOB_ID.out
```

Success ends with `VERIFICATION PASSED`. The first run downloads the backbone and head, so it requires Hugging Face network access and enough cache space. Remove or change `--partition=LocalQ` if the target cluster uses a different GPU partition. A GPU with BF16 and Flash Attention 3 support is recommended; if Flash Attention 3 is unavailable, use a compatible attention implementation supported by the installed Unsloth/Transformers stack and record the deviation.
