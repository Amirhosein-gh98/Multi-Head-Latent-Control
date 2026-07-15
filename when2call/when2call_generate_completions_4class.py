#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate model completions on top of the labeled When2Call auxiliary-head dataset.

This 4-class version:
- Reads the labeled When2Call file produced by when2call_build_head_labels_4class.py
- Keeps the 4 behavior targets:
  tool_call, request_for_info, cannot_answer, direct_answer
- Drops only ambiguous / unusable rows
- Generates completions with the target model
- Saves output as parquet shards of fixed size (default 2000 rows per shard)
- Supports resume by scanning existing parquet shards and skipping already-generated sample_ids
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import pyarrow.parquet as pq
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_TOKENIZER_ID = None
DEFAULT_INPUT_DIR = "data/train/when2call/when2call_processed/when2call_aux_labels.jsonl"
DEFAULT_OUTPUT_DIR = "data/train/when2call/qwen3vl/Qwen3-VL-2B-Instruct_4class/"
DEFAULT_PARQUET_CHUNK_SIZE = 2000

SUPPORTED_BEHAVIORS = {"tool_call", "request_for_info", "cannot_answer", "direct_answer"}


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1","true","t","yes","y","on"}:
        return True
    if s in {"0","false","f","no","n","off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def load_rows(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    p = Path(path)
    if p.is_dir():
        parquet_files = sorted(p.rglob("*.parquet"))
        jsonl_files = sorted(p.rglob("*.jsonl"))
        if parquet_files:
            ds = load_dataset("parquet", data_files=[str(x) for x in parquet_files], split="train")
        elif jsonl_files:
            ds = load_dataset("json", data_files=[str(x) for x in jsonl_files], split="train")
        else:
            raise FileNotFoundError(f"No parquet/jsonl files found under directory: {path}")
        n = len(ds) if limit is None else min(len(ds), int(limit))
        return [ds[i] for i in range(n)]

    if p.suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= int(limit):
                    break
        return rows

    ds = load_dataset("parquet" if p.suffix == ".parquet" else "json", data_files=str(p), split="train")
    n = len(ds) if limit is None else min(len(ds), int(limit))
    return [ds[i] for i in range(n)]


def messages_to_text(messages: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for msg in messages:
        role = str(msg.get("role", "unknown")).capitalize()
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "\n".join(str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in content)
        else:
            content = str(content)
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


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


def keep_only_supported_rows(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    kept: List[Dict[str, Any]] = []
    counts = {
        "total_input_rows": 0,
        "kept_rows": 0,
        "dropped_missing_sample_id": 0,
        "dropped_invalid_prompt_messages_json": 0,
        "dropped_unusable_behavior": 0,
        "dropped_unsupported_behavior": 0,
    }

    for row in rows:
        counts["total_input_rows"] += 1

        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            counts["dropped_missing_sample_id"] += 1
            continue

        try:
            prompt_messages = json.loads(row["prompt_messages_json"])
            if not isinstance(prompt_messages, list):
                raise ValueError("prompt_messages_json is not a list")
        except Exception:
            counts["dropped_invalid_prompt_messages_json"] += 1
            continue

        try:
            usable_behavior = int(row.get("usable_behavior", 0) or 0)
        except Exception:
            usable_behavior = 0
        if usable_behavior <= 0:
            counts["dropped_unusable_behavior"] += 1
            continue

        behavior = str(row.get("behavior", "")).strip()
        if behavior not in SUPPORTED_BEHAVIORS:
            counts["dropped_unsupported_behavior"] += 1
            continue

        kept.append(row)

    counts["kept_rows"] = len(kept)
    return kept, counts


def _safe_read_sample_ids_from_parquet(path: Path) -> List[str]:
    try:
        tbl = pq.read_table(path, columns=["sample_id"])
        col = tbl.column("sample_id").to_pylist()
        return [str(x) for x in col if x is not None and str(x)]
    except Exception:
        try:
            ds = load_dataset("parquet", data_files=str(path), split="train")
            return [str(x) for x in ds["sample_id"] if x is not None and str(x)]
        except Exception:
            return []


def load_existing_ids_from_shards(shard_dir: Path) -> Tuple[set[str], int]:
    done: set[str] = set()
    next_shard_idx = 0
    shard_files = sorted(shard_dir.glob("shard-*.parquet"))
    for path in tqdm(shard_files, desc="scanning existing parquet shards", dynamic_ncols=True):
        done.update(_safe_read_sample_ids_from_parquet(path))
        stem = path.stem
        try:
            idx = int(stem.split("-")[-1])
            next_shard_idx = max(next_shard_idx, idx + 1)
        except Exception:
            continue
    return done, next_shard_idx


def write_parquet_shard(rows: Sequence[Dict[str, Any]], shard_path: Path) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(list(rows)).to_parquet(str(shard_path))


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", default=DEFAULT_INPUT_DIR)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--tokenizer_id", default=DEFAULT_TOKENIZER_ID)
    ap.add_argument("--trust_remote_code", type=str2bool, default=True)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--model_family", default="auto", choices=["auto","qwen3_5","qwen3","qwen3_vl","gemma4"])
    ap.add_argument("--thinking_mode", default="auto")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=32000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--parquet_chunk_size", type=int, default=DEFAULT_PARQUET_CHUNK_SIZE)
    args = ap.parse_args()

    set_seed(int(args.seed))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / "parquet_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "generation_stats.json"

    loaded_rows = load_rows(args.input_path, args.limit)
    rows, filter_stats = keep_only_supported_rows(loaded_rows)

    rng = random.Random(int(args.seed))
    rng.shuffle(rows)

    print(
        f"[filter] kept {filter_stats['kept_rows']} / {filter_stats['total_input_rows']} rows "
        f"(dropped_unusable_behavior={filter_stats['dropped_unusable_behavior']}, "
        f"dropped_unsupported_behavior={filter_stats['dropped_unsupported_behavior']}, "
        f"dropped_invalid_prompt_messages_json={filter_stats['dropped_invalid_prompt_messages_json']}, "
        f"dropped_missing_sample_id={filter_stats['dropped_missing_sample_id']})",
        flush=True,
    )

    done_ids: set[str] = set()
    next_shard_idx = 0
    if args.resume:
        done_ids, next_shard_idx = load_existing_ids_from_shards(shard_dir)
        if done_ids:
            print(f"[resume] found {len(done_ids)} already-generated rows across parquet shards", flush=True)

    resolved_model_family = _infer_model_family(args.model_id, args.model_family)
    resolved_thinking_enabled = _resolve_thinking_enabled(args.model_id, resolved_model_family, args.thinking_mode)
    chat_template_kwargs = _chat_template_kwargs_for_runtime(resolved_model_family, resolved_thinking_enabled)

    llm_kwargs = dict(
        model=args.model_id,
        trust_remote_code=bool(args.trust_remote_code),
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        dtype=args.dtype,
    )
    tokenizer_id = args.tokenizer_id or args.model_id
    if tokenizer_id:
        llm_kwargs["tokenizer"] = tokenizer_id
    if args.quantization not in (None, "", "none", "None"):
        llm_kwargs["quantization"] = args.quantization
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
    )

    batch_rows: List[Dict[str, Any]] = []
    batch_conversations: List[List[Dict[str, Any]]] = []
    shard_buffer: List[Dict[str, Any]] = []

    total_candidates = len(rows)
    total_skipped_resume = 0
    total_generated = 0
    total_written = 0

    progress = tqdm(total=total_candidates, desc="generating completions", dynamic_ncols=True)

    def flush_shard_buffer(force: bool = False) -> None:
        nonlocal shard_buffer, next_shard_idx, total_written
        chunk_size = int(args.parquet_chunk_size)
        while len(shard_buffer) >= chunk_size or (force and shard_buffer):
            take = chunk_size if len(shard_buffer) >= chunk_size else len(shard_buffer)
            rows_to_write = shard_buffer[:take]
            shard_buffer = shard_buffer[take:]
            shard_path = shard_dir / f"shard-{next_shard_idx:06d}.parquet"
            write_parquet_shard(rows_to_write, shard_path)
            total_written += len(rows_to_write)
            next_shard_idx += 1

    def flush_generation_batch() -> int:
        nonlocal batch_rows, batch_conversations, shard_buffer, total_generated
        if not batch_rows:
            return 0

        current_n = len(batch_rows)
        chat_kwargs = {"sampling_params": sampling, "use_tqdm": False}
        if chat_template_kwargs:
            chat_kwargs["chat_template_kwargs"] = chat_template_kwargs
        outs = llm.chat(batch_conversations, **chat_kwargs)
        write_rows: List[Dict[str, Any]] = []

        for row, out in zip(batch_rows, outs):
            completion = out.outputs[0].text if out.outputs else ""
            prompt_messages = json.loads(row["prompt_messages_json"])
            system_prompt = build_system_prompt(row.get("tools_json", "[]"))
            full_messages = (
                [{"role": "system", "content": system_prompt}]
                + prompt_messages
                + [{"role": "assistant", "content": completion}]
            )
            out_row = dict(row)
            out_row.update(
                {
                    "system_prompt": system_prompt,
                    "prompt_text": messages_to_text(prompt_messages),
                    "completion": completion,
                    "messages_json": json.dumps(full_messages, ensure_ascii=False),
                    "generation_model_id": args.model_id,
                    "completion_length": len(completion),
                    "generation_temperature": float(args.temperature),
                    "generation_top_p": float(args.top_p),
                    "requested_model_family": args.model_family,
                    "resolved_model_family": resolved_model_family,
                    "requested_thinking_mode": args.thinking_mode,
                    "resolved_thinking_enabled": bool(resolved_thinking_enabled),
                }
            )
            write_rows.append(out_row)

        shard_buffer.extend(write_rows)
        total_generated += len(write_rows)
        flush_shard_buffer(force=False)
        batch_rows = []
        batch_conversations = []
        return current_n

    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id and sample_id in done_ids:
            total_skipped_resume += 1
            progress.update(1)
            continue

        prompt_messages = json.loads(row["prompt_messages_json"])
        system_prompt = build_system_prompt(row.get("tools_json", "[]"))
        messages = [{"role": "system", "content": system_prompt}] + prompt_messages
        messages = _apply_runtime_prompting(messages, resolved_model_family, resolved_thinking_enabled)
        batch_rows.append(row)
        batch_conversations.append(messages)

        if len(batch_rows) >= int(args.batch_size):
            finished_n = flush_generation_batch()
            progress.update(finished_n)

    if batch_rows:
        finished_n = flush_generation_batch()
        progress.update(finished_n)

    progress.close()
    flush_shard_buffer(force=True)

    stats = {
        "model_id": args.model_id,
        "input_path": args.input_path,
        "output_dir": str(out_dir),
        "parquet_shard_dir": str(shard_dir),
        "parquet_chunk_size": int(args.parquet_chunk_size),
        "num_input_rows_loaded": len(loaded_rows),
        "num_rows_after_filtering": len(rows),
        "filter_stats": filter_stats,
        "num_total_candidates": total_candidates,
        "num_skipped_resume": total_skipped_resume,
        "num_generated_new": total_generated,
        "num_rows_written_new": total_written,
        "next_shard_idx": next_shard_idx,
        "generation_temperature": float(args.temperature),
        "generation_top_p": float(args.top_p),
        "max_tokens": int(args.max_tokens),
        "tokenizer_id": tokenizer_id,
        "trust_remote_code": bool(args.trust_remote_code),
        "dtype": args.dtype,
        "quantization": args.quantization,
        "requested_model_family": args.model_family,
        "resolved_model_family": resolved_model_family,
        "requested_thinking_mode": args.thinking_mode,
        "resolved_thinking_enabled": bool(resolved_thinking_enabled),
    }
    save_json(stats_path, stats)
    print(f"[done] wrote {total_written} new rows into parquet shards under {shard_dir}", flush=True)
    print(f"[done] stats saved to {stats_path}", flush=True)


if __name__ == "__main__":
    main()
