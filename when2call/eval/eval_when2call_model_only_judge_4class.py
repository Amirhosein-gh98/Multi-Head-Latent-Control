#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate the MODEL ITSELF on saved When2Call completions using a separate judge model.

This script DOES NOT regenerate the base-model completion. It reads the saved
completion from generated_eval_path, asks a judge model to infer the 4-class action
represented by that completion, and reports how often the saved completion matches
the gold action label.

Important:
- This version is aligned with the 4-class setup:
  tool_call, request_for_info, cannot_answer, direct_answer
- Direct-answer rows are kept and judged like the other classes.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import datasets
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(os.path.dirname(current_dir))
for p in [current_dir, base_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from aux_head_shared_utils import save_json, set_seed

CONFIG = {
    "generated_eval_path": "eval_outputs/when2call/Qwen3-VL-2B-Instruct/when2call_test_generated_4class.parquet",
    "output_dir": "eval_outputs/when2call/Qwen3-VL-2B-Instruct/model_only_judge_eval_4class",
    "seed": 42,
    "max_eval_rows": None,
    "judge_model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    "judge_tokenizer_id": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    "judge_dtype": "auto",
    "judge_quantization": None,
    "judge_tensor_parallel_size": 1,
    "judge_gpu_memory_utilization": 0.50,
    "judge_max_model_len": 32000,
    "judge_batch_size": 128,
    "judge_max_tokens": 2000,
    "judge_temperature": 1.0,
    "judge_top_p": 0.95,
}

CANONICAL_CLASS_ORDER = ["tool_call", "request_for_info", "cannot_answer", "direct_answer"]

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for tool-use behavior.
Read the tools, the user question, and the assistant response.
Decide which 4-class behavior the assistant response represents.
Return JSON only."""


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
    ap.add_argument("--generated_eval_path", default=CONFIG["generated_eval_path"])
    ap.add_argument("--output_dir", default=CONFIG["output_dir"])
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    ap.add_argument("--max_eval_rows", type=int, default=CONFIG["max_eval_rows"])
    ap.add_argument("--judge_model_id", default=CONFIG["judge_model_id"])
    ap.add_argument("--judge_tokenizer_id", default=CONFIG["judge_tokenizer_id"])
    ap.add_argument("--judge_dtype", default=CONFIG["judge_dtype"])
    ap.add_argument("--judge_quantization", default=CONFIG["judge_quantization"])
    ap.add_argument("--judge_tensor_parallel_size", type=int, default=CONFIG["judge_tensor_parallel_size"])
    ap.add_argument("--judge_gpu_memory_utilization", type=float, default=CONFIG["judge_gpu_memory_utilization"])
    ap.add_argument("--judge_max_model_len", type=int, default=CONFIG["judge_max_model_len"])
    ap.add_argument("--judge_batch_size", type=int, default=CONFIG["judge_batch_size"])
    ap.add_argument("--judge_max_tokens", type=int, default=CONFIG["judge_max_tokens"])
    ap.add_argument("--judge_temperature", type=float, default=CONFIG["judge_temperature"])
    ap.add_argument("--judge_top_p", type=float, default=CONFIG["judge_top_p"])
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


def build_judge_user_prompt(row: Dict[str, Any]) -> str:
    tools_json = row.get("tools_json") or tools_to_text(row.get("tools", []))
    question = str(row.get("question", ""))
    completion = str(row.get("completion", ""))
    return f"""
Choose exactly one behavior label for the assistant response:
- tool_call
- request_for_info
- cannot_answer
- direct_answer

Available tools:
<tools>
{tools_json}
</tools>

User question:
{question}

Assistant response:
{completion}

Return exactly:
{{
  "behavior": "tool_call|request_for_info|cannot_answer|direct_answer",
  "confidence": 0.0 to 1.0,
  "quality": 0.0 to 1.0,
  "reason": "short reason"
}}
""".strip()


def extract_first_json_object(text: str) -> Dict[str, Any]:
    s = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    start = s.find("{")
    if start < 0:
        raise ValueError(f"Could not find JSON object in: {text[:400]}")
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(s)):
        ch = s[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(s[start:idx + 1])
    raise ValueError(f"Could not parse complete JSON object from: {text[:400]}")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def confusion_and_metrics(gold_ids: List[int], pred_ids: List[int], class_names: List[str]) -> Dict[str, Any]:
    n = len(class_names)
    conf = torch.zeros((n, n), dtype=torch.long)
    for g, p in zip(gold_ids, pred_ids):
        if 0 <= g < n and 0 <= p < n:
            conf[g, p] += 1
    tp = conf.diag().float()
    row = conf.sum(dim=1).float()
    col = conf.sum(dim=0).float()
    prec = tp / col.clamp_min(1.0)
    rec = tp / row.clamp_min(1.0)
    f1 = (2.0 * prec * rec) / (prec + rec).clamp_min(1e-12)
    total = conf.sum().float().clamp_min(1.0)
    metrics = {
        "accuracy": float((tp.sum() / total).item()),
        "macro_precision": float(prec.mean().item()),
        "macro_recall": float(rec.mean().item()),
        "macro_f1": float(f1.mean().item()),
        "class_names": list(class_names),
        "confusion": conf.tolist(),
        "per_class": {},
    }
    for i, name in enumerate(class_names):
        metrics["per_class"][name] = {"precision": float(prec[i].item()), "recall": float(rec[i].item()), "f1": float(f1[i].item()), "support": int(row[i].item())}
    return metrics


def load_generated_rows(path: str, max_rows: int | None, supported_class_names: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    p = Path(path)
    if p.suffix == ".parquet":
        ds = datasets.load_dataset("parquet", data_files=str(p), split="train")
    elif p.suffix == ".jsonl":
        ds = datasets.load_dataset("json", data_files=str(p), split="train")
    else:
        raise ValueError(f"Unsupported file type: {path}")
    if max_rows is not None:
        ds = ds.select(range(min(int(max_rows), len(ds))))
    rows: List[Dict[str, Any]] = []
    stats = {"loaded_rows_before_filtering": len(ds), "kept_rows": 0, "dropped_unsupported_gold_rows": 0}
    supported = set(supported_class_names)
    for i in range(len(ds)):
        row = ds[i]
        gold_name = normalize_class_name(row.get("gold_label") or row.get("correct_answer"))
        if gold_name not in supported:
            stats["dropped_unsupported_gold_rows"] += 1
            continue
        if not row.get("uuid"):
            row["uuid"] = row.get("sample_id") or f"row_{i}"
        row["gold_label"] = gold_name
        rows.append(row)
    stats["kept_rows"] = len(rows)
    return rows, stats


def _plot_hist(values_a: List[float], values_b: List[float], label_a: str, label_b: str, title: str, xlab: str, out_path: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    if values_a:
        plt.hist(values_a, bins=30, alpha=0.65, label=label_a)
    if values_b:
        plt.hist(values_b, bins=30, alpha=0.65, label=label_b)
    plt.xlabel(xlab)
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_confusion(confusion: List[List[int]], class_names: List[str], title: str, out_path: Path) -> None:
    cm = torch.tensor(confusion).numpy()
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Gold")
    plt.title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    cfg = parse_args()
    set_seed(int(cfg["seed"]))
    outdir = Path(cfg["output_dir"])
    plots_dir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows, row_stats = load_generated_rows(cfg["generated_eval_path"], cfg["max_eval_rows"], CANONICAL_CLASS_ORDER)
    if not rows:
        raise ValueError("No supported eval rows remain after filtering. Regenerate eval completions with the 4-class generator.")

    save_json(outdir / "config_used.json", cfg)
    save_json(outdir / "row_filter_stats.json", row_stats)

    tokenizer = AutoTokenizer.from_pretrained(cfg["judge_tokenizer_id"], trust_remote_code=True)
    llm_kwargs = {
        "model": cfg["judge_model_id"],
        "tokenizer": cfg["judge_tokenizer_id"],
        "trust_remote_code": True,
        "tensor_parallel_size": int(cfg["judge_tensor_parallel_size"]),
        "gpu_memory_utilization": float(cfg["judge_gpu_memory_utilization"]),
        "max_model_len": int(cfg["judge_max_model_len"]),
        "dtype": cfg["judge_dtype"],
    }
    if cfg["judge_quantization"] not in (None, "", "none", "None"):
        llm_kwargs["quantization"] = cfg["judge_quantization"]
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=float(cfg["judge_temperature"]),
        top_p=float(cfg["judge_top_p"]),
        max_tokens=int(cfg["judge_max_tokens"]),
    )

    prompts: List[str] = []
    for row in rows:
        msgs = [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": build_judge_user_prompt(row)}]
        prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    gold_ids: List[int] = []
    pred_ids: List[int] = []
    conf_correct: List[float] = []
    conf_wrong: List[float] = []
    qual_correct: List[float] = []
    qual_wrong: List[float] = []
    class_to_id = {name: i for i, name in enumerate(CANONICAL_CLASS_ORDER)}
    invalid_judge_output_count = 0
    records: List[Dict[str, Any]] = []

    bs = int(cfg["judge_batch_size"])
    for start in tqdm(range(0, len(rows), bs), desc="judge-eval", dynamic_ncols=True):
        chunk_rows = rows[start:start + bs]
        chunk_prompts = prompts[start:start + bs]
        outs = llm.generate(chunk_prompts, sampling, use_tqdm=False)
        for row, out in zip(chunk_rows, outs):
            raw_text = out.outputs[0].text if out.outputs else ""
            try:
                obj = extract_first_json_object(raw_text)
            except Exception:
                obj = {"behavior": "unknown", "confidence": 0.0, "quality": 0.0, "reason": raw_text[:200]}
            pred_name = normalize_class_name(obj.get("behavior", "unknown"))
            gold_name = normalize_class_name(row.get("gold_label"))
            gold_id = class_to_id[gold_name]
            pred_id = class_to_id.get(pred_name, -1)
            conf = max(0.0, min(1.0, _safe_float(obj.get("confidence", 0.0), 0.0)))
            qual = max(0.0, min(1.0, _safe_float(obj.get("quality", 0.0), 0.0)))
            is_correct = int(pred_name == gold_name)
            if pred_id < 0:
                invalid_judge_output_count += 1
            gold_ids.append(gold_id)
            pred_ids.append(pred_id)
            if is_correct:
                conf_correct.append(conf)
                qual_correct.append(qual)
            else:
                conf_wrong.append(conf)
                qual_wrong.append(qual)
            records.append({
                "uuid": row.get("uuid"),
                "question": row.get("question"),
                "tools": row.get("tools") or row.get("tools_json"),
                "gold_label": gold_name,
                "generated_completion": row.get("completion"),
                "judge_raw": raw_text,
                "judge_pred_label": pred_name,
                "judge_confidence": conf,
                "judge_quality": qual,
                "judge_is_correct": is_correct,
                "judge_reason": str(obj.get("reason", ""))[:400],
            })

    metrics = {
        "num_rows_scored": len(rows),
        "class_names": CANONICAL_CLASS_ORDER,
        "row_filter_stats": row_stats,
        "invalid_judge_output_count": int(invalid_judge_output_count),
        "judge_on_saved_completion": confusion_and_metrics(gold_ids, pred_ids, CANONICAL_CLASS_ORDER),
    }
    save_json(outdir / "metrics.json", metrics)

    with open(outdir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    prob_summary = {
        "judge_confidence_correct_mean": float(sum(conf_correct) / max(len(conf_correct), 1)),
        "judge_confidence_wrong_mean": float(sum(conf_wrong) / max(len(conf_wrong), 1)),
        "judge_quality_correct_mean": float(sum(qual_correct) / max(len(qual_correct), 1)),
        "judge_quality_wrong_mean": float(sum(qual_wrong) / max(len(qual_wrong), 1)),
    }
    save_json(outdir / "judge_summary.json", prob_summary)

    _plot_hist(conf_correct, conf_wrong, "correct", "wrong", "Judge confidence by correctness", "Confidence", plots_dir / "judge_confidence_by_correctness.png")
    _plot_hist(qual_correct, qual_wrong, "correct", "wrong", "Judge quality by correctness", "Quality", plots_dir / "judge_quality_by_correctness.png")
    _plot_confusion(metrics["judge_on_saved_completion"]["confusion"], CANONICAL_CLASS_ORDER, "Judge confusion matrix", plots_dir / "judge_confusion.png")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved metrics to {outdir / 'metrics.json'}")
    print(f"saved predictions to {outdir / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
