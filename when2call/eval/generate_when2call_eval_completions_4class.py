#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate model completions on the official When2Call test split for the 4-class setup.

This is separate from head evaluation on purpose:
1) this script only generates and saves completions
2) a second script loads the saved completions and evaluates the head + model action selection

Important:
- This version is aligned with the 4-class train/eval setup:
  tool_call, request_for_info, cannot_answer, direct_answer
- Direct-answer rows are kept and evaluated like the other classes.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

CONFIG = {
    "model_id": "Qwen/Qwen3-VL-2B-Instruct",
    "tokenizer_id": None,
    "model_family": "auto",
    "thinking_mode": "auto",
    "trust_remote_code": True,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.50,
    "dtype": "auto",
    "max_model_len": 32000,
    "quantization": None,
    "dataset_name": "nvidia/When2Call",
    "dataset_config": "test",
    "eval_split": "mcq",
    "max_eval_rows": None,
    "batch_size": 64,
    "max_tokens": 16000,
    "temperature": 1.0,
    "top_p": 0.95,
    "seed": 42,
    "output_path": "eval_outputs/when2call/Qwen3-VL-2B-Instruct/when2call_test_generated_4class.parquet",
    "also_write_jsonl": True,
}

CANONICAL_CLASS_ORDER = ["tool_call", "request_for_info", "cannot_answer", "direct_answer"]


def _infer_model_family(model_id: str, requested_family: str = "auto") -> str:
    requested_family = str(requested_family or "auto").strip().lower()
    if requested_family != "auto":
        return requested_family
    mid = (model_id or "").strip().lower()
    if "qwen3.5" in mid:
        return "qwen3_5"
    if "qwen3-vl" in mid:
        return "qwen3_vl"
    if "gemma-4" in mid:
        return "gemma4"
    if "qwen3" in mid:
        return "qwen3"
    return "other"


def _resolve_thinking_enabled(model_id: str, model_family: str, thinking_mode: Any) -> bool:
    if isinstance(thinking_mode, bool):
        return bool(thinking_mode)
    mode = str(thinking_mode or "auto").strip().lower()
    if mode == "on":
        return True
    if mode == "off":
        return False

    mid = (model_id or "").strip().lower()
    if model_family == "qwen3_5":
        if any(x in mid for x in ["qwen3.5-0.8b", "qwen3.5-2b"]):
            return False
        return True
    if model_family == "qwen3_vl":
        return "thinking" in mid
    if model_family == "qwen3":
        if "instruct" in mid and "thinking" not in mid:
            return False
        return True
    if model_family == "gemma4":
        return False
    return False


def _remove_gemma_think_prefix(text: str) -> str:
    text = text or ""
    return re.sub(r"^\s*<\|think\|>\s*\n?", "", text, count=1)


def _apply_runtime_prompting(messages: List[Dict[str, Any]], model_family: str, thinking_enabled: bool) -> List[Dict[str, Any]]:
    patched: List[Dict[str, Any]] = []
    for msg in messages:
        cloned = dict(msg)
        if isinstance(cloned.get("content"), list):
            cloned["content"] = list(cloned["content"])
        patched.append(cloned)

    if model_family != "gemma4":
        return patched

    if not patched or patched[0].get("role") != "system":
        if thinking_enabled:
            patched.insert(0, {"role": "system", "content": "<|think|>"})
        return patched

    system_content = patched[0].get("content", "")
    if not isinstance(system_content, str):
        return patched

    system_content = _remove_gemma_think_prefix(system_content)
    if thinking_enabled:
        system_content = "<|think|>\n" + system_content if system_content else "<|think|>"
    patched[0]["content"] = system_content
    return patched


def _chat_template_kwargs_for_runtime(model_family: str, thinking_enabled: bool) -> Optional[Dict[str, Any]]:
    if model_family in {"qwen3_5", "qwen3"}:
        return {"enable_thinking": bool(thinking_enabled)}
    return None


def _patch_chat_template_callable(bound_callable, model_family: str, thinking_enabled: bool):
    def patched(messages, *args, **kwargs):
        if model_family == "gemma4":
            messages = _apply_runtime_prompting(messages, model_family, thinking_enabled)
        elif model_family in {"qwen3_5", "qwen3"}:
            kwargs = dict(kwargs)
            kwargs.setdefault("enable_thinking", thinking_enabled)
        return bound_callable(messages, *args, **kwargs)
    return patched


def _patch_processor_for_runtime_prompting(processor, model_family: str, thinking_enabled: bool):
    patched_targets = []

    if hasattr(processor, "apply_chat_template"):
        processor.apply_chat_template = _patch_chat_template_callable(
            processor.apply_chat_template, model_family, thinking_enabled
        )
        patched_targets.append("processor.apply_chat_template")

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        tokenizer.apply_chat_template = _patch_chat_template_callable(
            tokenizer.apply_chat_template, model_family, thinking_enabled
        )
        patched_targets.append("tokenizer.apply_chat_template")

    return processor, {
        "resolved_model_family": model_family,
        "resolved_thinking_enabled": bool(thinking_enabled),
        "patched_targets": patched_targets,
    }


def _set_image_processor_max_pixels_safe(processor: Any, max_pixels: int) -> Dict[str, Any]:
    info = {"requested_max_pixels": int(max_pixels), "applied_on": [], "errors": []}
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return info
    if hasattr(image_processor, "max_pixels"):
        try:
            image_processor.max_pixels = int(max_pixels)
            info["applied_on"].append("image_processor.max_pixels")
        except Exception as e:
            info["errors"].append(f"image_processor.max_pixels: {type(e).__name__}: {e}")
    return info


def _resolve_attn_implementation_for_model(attn_implementation: str, model_family: str) -> str:
    attn = str(attn_implementation or "").strip()
    if model_family == "gemma4" and attn == "flash_attention_3":
        return "sdpa"
    return attn


def _sanitize_forward_inputs(forward_inputs: Dict[str, Any], model_family: str) -> Dict[str, Any]:
    return dict(forward_inputs)


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_args() -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default=CONFIG["model_id"])
    ap.add_argument("--tokenizer_id", default=CONFIG["tokenizer_id"])
    ap.add_argument("--model_family", default=CONFIG["model_family"], choices=["auto","qwen3_5","qwen3","qwen3_vl","gemma4"])
    ap.add_argument("--thinking_mode", default=CONFIG["thinking_mode"])
    ap.add_argument("--trust_remote_code", type=str2bool, default=CONFIG["trust_remote_code"])
    ap.add_argument("--tensor_parallel_size", type=int, default=CONFIG["tensor_parallel_size"])
    ap.add_argument("--gpu_memory_utilization", type=float, default=CONFIG["gpu_memory_utilization"])
    ap.add_argument("--dtype", default=CONFIG["dtype"])
    ap.add_argument("--max_model_len", type=int, default=CONFIG["max_model_len"])
    ap.add_argument("--quantization", default=CONFIG["quantization"])
    ap.add_argument("--dataset_name", default=CONFIG["dataset_name"])
    ap.add_argument("--dataset_config", default=CONFIG["dataset_config"])
    ap.add_argument("--eval_split", default=CONFIG["eval_split"])
    ap.add_argument("--max_eval_rows", type=int, default=CONFIG["max_eval_rows"])
    ap.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
    ap.add_argument("--max_tokens", type=int, default=CONFIG["max_tokens"])
    ap.add_argument("--temperature", type=float, default=CONFIG["temperature"])
    ap.add_argument("--top_p", type=float, default=CONFIG["top_p"])
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    ap.add_argument("--output_path", default=CONFIG["output_path"])
    ap.add_argument("--also_write_jsonl", type=str2bool, default=CONFIG["also_write_jsonl"])
    return vars(ap.parse_args())


def normalize_class_name(x: Any) -> str:
    s = str(x or "").strip().lower()
    aliases = {
        "direct": "direct_answer",
        "direct_answer": "direct_answer",
        "answer_directly": "direct_answer",
        "tool": "tool_call",
        "tool_call": "tool_call",
        "toolcall": "tool_call",
        "call_tool": "tool_call",
        "request_for_info": "request_for_info",
        "request_info": "request_for_info",
        "follow_up": "request_for_info",
        "followup": "request_for_info",
        "clarification": "request_for_info",
        "clarification_question": "request_for_info",
        "cannot_answer": "cannot_answer",
        "cannot_answer_with_provided_tools": "cannot_answer",
        "cannot_answer_with_tools": "cannot_answer",
        "unable_to_answer": "cannot_answer",
    }
    return aliases.get(s, s)


def tools_to_text(tools: Any) -> str:
    if tools is None:
        return "[]"
    if isinstance(tools, str):
        return tools
    try:
        return json.dumps(tools, ensure_ascii=False, indent=2)
    except Exception:
        return str(tools)


def build_system_prompt(tools_json: str) -> str:
    return f"""You are a careful assistant. Choose exactly one response mode:

1) Tool call — use a tool only if it can answer the request and all required information is available.
2) Clarification question — ask for missing required information before using a tool.
3) Cannot answer with provided tools — use this if the available tools are insufficient.
4) Direct answer — answer directly when no tool is needed and the request can be answered now.

Rules:
- Do not guess missing information.
- Do not hallucinate tools or unavailable facts.
- If clarifying, do not call a tool yet.
- In all cases, put your final response inside \\boxed{{...}}.

Available tools:
<tools>
{tools_json}
</tools>""".strip()


def load_eval_rows(cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    ds = load_dataset(cfg["dataset_name"], cfg["dataset_config"])
    if cfg["eval_split"] not in ds:
        raise ValueError(f"eval_split={cfg['eval_split']!r} not found. Available: {list(ds.keys())}")
    split = ds[cfg["eval_split"]]
    if cfg["max_eval_rows"] is not None:
        split = split.select(range(min(int(cfg["max_eval_rows"]), len(split))))
    rows: List[Dict[str, Any]] = []
    stats = {"loaded_rows_before_filtering": len(split), "kept_rows": 0, "dropped_unsupported_gold_rows": 0, "gold_counts": {name: 0 for name in CANONICAL_CLASS_ORDER}}
    for i in range(len(split)):
        row = split[i]
        gold_label = normalize_class_name(row.get("correct_answer"))
        if gold_label not in CANONICAL_CLASS_ORDER:
            stats["dropped_unsupported_gold_rows"] += 1
            continue
        stats["gold_counts"][gold_label] += 1
        rows.append(row)
    stats["kept_rows"] = len(rows)
    return rows, stats


def build_conversations(rows: List[Dict[str, Any]], model_family: str, thinking_enabled: bool) -> List[List[Dict[str, Any]]]:
    conversations: List[List[Dict[str, Any]]] = []
    for ex in rows:
        tools_json = tools_to_text(ex.get("tools", []))
        system_prompt = build_system_prompt(tools_json)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": str(ex["question"])}]
        conversations.append(_apply_runtime_prompting(messages, model_family, thinking_enabled))
    return conversations


def main() -> None:
    cfg = parse_args()
    out_path = Path(cfg["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_id = cfg["tokenizer_id"] or cfg["model_id"]
    resolved_model_family = _infer_model_family(cfg["model_id"], cfg["model_family"])
    resolved_thinking_enabled = _resolve_thinking_enabled(cfg["model_id"], resolved_model_family, cfg["thinking_mode"])
    chat_template_kwargs = _chat_template_kwargs_for_runtime(resolved_model_family, resolved_thinking_enabled)

    llm_kwargs = {
        "model": cfg["model_id"],
        "tokenizer": tokenizer_id,
        "trust_remote_code": bool(cfg["trust_remote_code"]),
        "tensor_parallel_size": int(cfg["tensor_parallel_size"]),
        "gpu_memory_utilization": float(cfg["gpu_memory_utilization"]),
        "max_model_len": int(cfg["max_model_len"]),
        "dtype": cfg["dtype"],
        "seed": int(cfg["seed"]),
    }
    if cfg["quantization"] not in (None, "", "none", "None"):
        llm_kwargs["quantization"] = cfg["quantization"]
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=float(cfg["temperature"]),
        top_p=float(cfg["top_p"]),
        max_tokens=int(cfg["max_tokens"]),
    )

    rows, row_stats = load_eval_rows(cfg)
    if not rows:
        raise ValueError("No supported 4-class eval rows remain after filtering.")
    conversations = build_conversations(rows, resolved_model_family, resolved_thinking_enabled)

    saved_rows: List[Dict[str, Any]] = []
    batch_size = int(cfg["batch_size"])
    for start in tqdm(range(0, len(rows), batch_size), desc="generate", dynamic_ncols=True):
        end = min(start + batch_size, len(rows))
        batch_rows = rows[start:end]
        batch_conversations = conversations[start:end]
        chat_kwargs = {"sampling_params": sampling, "use_tqdm": False}
        if chat_template_kwargs:
            chat_kwargs["chat_template_kwargs"] = chat_template_kwargs
        outputs = llm.chat(batch_conversations, **chat_kwargs)
        for ex, out in zip(batch_rows, outputs):
            completion = out.outputs[0].text if out.outputs else ""
            tools_json = tools_to_text(ex.get("tools", []))
            system_prompt = build_system_prompt(tools_json)
            gold_label = normalize_class_name(ex["correct_answer"])
            saved_rows.append({
                "uuid": str(ex.get("uuid", len(saved_rows))),
                "eval_split": cfg["eval_split"],
                "question": str(ex["question"]),
                "tools": ex.get("tools", []),
                "tools_json": tools_json,
                "correct_answer": str(ex["correct_answer"]),
                "gold_label": gold_label,
                "answers": ex.get("answers", {}),
                "system_prompt": system_prompt,
                "prompt_text": str(ex["question"]),
                "completion": completion,
                "generation_model_id": cfg["model_id"],
                "completion_length": len(completion),
                "generation_temperature": float(cfg["temperature"]),
                "generation_top_p": float(cfg["top_p"]),
                "requested_model_family": cfg["model_family"],
                "resolved_model_family": resolved_model_family,
                "requested_thinking_mode": cfg["thinking_mode"],
                "resolved_thinking_enabled": bool(resolved_thinking_enabled),
            })

    ds = Dataset.from_list(saved_rows)
    ds.to_parquet(str(out_path))
    print(f"saved parquet to {out_path}")

    stats_path = out_path.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({**cfg, "row_filter_stats": row_stats, "num_generated_rows": len(saved_rows), "tokenizer_id": tokenizer_id, "resolved_model_family": resolved_model_family, "resolved_thinking_enabled": bool(resolved_thinking_enabled)}, f, ensure_ascii=False, indent=2)
    print(f"saved stats to {stats_path}")

    if cfg.get("also_write_jsonl", False):
        jsonl_path = out_path.with_suffix(".jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in saved_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"saved jsonl to {jsonl_path}")


if __name__ == "__main__":
    main()
