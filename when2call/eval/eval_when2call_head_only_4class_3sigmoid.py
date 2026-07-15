#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import datasets
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from unsloth import FastVisionModel

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(os.path.dirname(current_dir))
for p in [current_dir, base_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from aux_head_shared_utils import (
    AuxHeadModule,
    ChatBatchBuilder,
    dtype_from_str,
    get_device,
    infer_hidden_size_and_num_hidden_layers,
    move_batch_to_device,
    resolve_transformer_layer_indices,
    save_json,
    set_seed,
)

HEAD_CLASS_NAMES = ["tool_call", "request_for_info", "cannot_answer"]
BEHAVIOR_CLASS_NAMES = ["tool_call", "request_for_info", "cannot_answer", "direct_answer"]
BEHAVIOR_TO_ID = {name: i for i, name in enumerate(BEHAVIOR_CLASS_NAMES)}


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


CONFIG = {
    "model_name_or_path": "Qwen/Qwen3-VL-2B-Instruct",
    "trust_remote_code": True,
    "attn_implementation": "flash_attention_2",
    "prefer_unsloth_mirror": True,
    "load_in_4bit": False,
    "load_in_8bit": False,
    "use_gradient_checkpointing": "unsloth",
    "dtype": "bf16",
    "model_family": "auto",
    "thinking_mode": "auto",
    "head_checkpoint_path": "trained_models/Qwen3-VL-2B-Instruct_When2call_4class_3sigmoid/head-final.pt",
    "generated_eval_path": "eval_outputs/when2call/Qwen3-VL-2B-Instruct/when2call_test_generated_4class.parquet",
    "output_dir": "eval_outputs/when2call/Qwen3-VL-2B-Instruct/head_only_eval_4class_3sigmoid",
    "seed": 42,
    "max_eval_rows": None,
    "max_seq_len": 32000,
    "max_pixels": 200_000,
    "max_head_input_tokens": None,
    "head_input_mode": None,
    "hidden_encoder_type": None,
    "selected_hidden_layer_indices": None,
    "hidden_layer_selection": "middle",
    "hidden_layer_index": None,
    "class_names": list(HEAD_CLASS_NAMES),
    "behavior_class_names": list(BEHAVIOR_CLASS_NAMES),
    "decision_threshold": 0.1,
    "per_device_batch_size": 1,
    "num_workers": 2,
    "pin_memory": True,
    "persistent_workers": False,
}


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_optional_int_list(text: str | None) -> List[int] | None:
    if text is None:
        return None
    s = str(text).strip()
    if not s or s.lower() == "none":
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [int(x) for x in obj]
    except Exception:
        pass
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args() -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", default=CONFIG["model_name_or_path"])
    ap.add_argument("--trust_remote_code", type=str2bool, default=CONFIG["trust_remote_code"])
    ap.add_argument("--attn_implementation", default=CONFIG["attn_implementation"])
    ap.add_argument("--prefer_unsloth_mirror", type=str2bool, default=CONFIG["prefer_unsloth_mirror"])
    ap.add_argument("--load_in_4bit", type=str2bool, default=CONFIG["load_in_4bit"])
    ap.add_argument("--load_in_8bit", type=str2bool, default=CONFIG["load_in_8bit"])
    ap.add_argument("--use_gradient_checkpointing", default=CONFIG["use_gradient_checkpointing"])
    ap.add_argument("--dtype", default=CONFIG["dtype"])
    ap.add_argument("--model_family", default=CONFIG["model_family"], choices=["auto","qwen3_5","qwen3","qwen3_vl","gemma4"])
    ap.add_argument("--thinking_mode", default=CONFIG["thinking_mode"])
    ap.add_argument("--head_checkpoint_path", default=CONFIG["head_checkpoint_path"])
    ap.add_argument("--generated_eval_path", default=CONFIG["generated_eval_path"])
    ap.add_argument("--output_dir", default=CONFIG["output_dir"])
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    ap.add_argument("--max_eval_rows", type=int, default=CONFIG["max_eval_rows"])
    ap.add_argument("--max_seq_len", type=int, default=CONFIG["max_seq_len"])
    ap.add_argument("--max_pixels", type=int, default=CONFIG["max_pixels"])
    ap.add_argument("--max_head_input_tokens", "--max_input_tokens", dest="max_head_input_tokens", type=int, default=CONFIG["max_head_input_tokens"])
    ap.add_argument("--head_input_mode", default=CONFIG["head_input_mode"])
    ap.add_argument("--hidden_encoder_type", default=CONFIG["hidden_encoder_type"])
    ap.add_argument("--selected_hidden_layer_indices", default=None)
    ap.add_argument("--hidden_layer_selection", default=CONFIG["hidden_layer_selection"])
    ap.add_argument("--hidden_layer_index", type=int, default=CONFIG["hidden_layer_index"])
    ap.add_argument("--class_names", nargs="+", default=list(CONFIG["class_names"]))
    ap.add_argument("--behavior_class_names", nargs="+", default=list(CONFIG["behavior_class_names"]))
    ap.add_argument("--decision_threshold", type=float, default=CONFIG["decision_threshold"])
    ap.add_argument("--per_device_batch_size", type=int, default=CONFIG["per_device_batch_size"])
    ap.add_argument("--num_workers", type=int, default=CONFIG["num_workers"])
    ap.add_argument("--pin_memory", type=str2bool, default=CONFIG["pin_memory"])
    ap.add_argument("--persistent_workers", type=str2bool, default=CONFIG["persistent_workers"])
    args = vars(ap.parse_args())
    args["selected_hidden_layer_indices"] = parse_optional_int_list(args["selected_hidden_layer_indices"])
    if args["head_input_mode"] is not None and str(args["head_input_mode"]).lower() == "none":
        args["head_input_mode"] = None
    if args["hidden_encoder_type"] is not None and str(args["hidden_encoder_type"]).lower() == "none":
        args["hidden_encoder_type"] = None
    return args


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


def _resolve_selected_hidden_layers_from_cfg(cfg: Dict[str, Any], *, num_hidden_layers: int, ckpt_selected_hidden_layer_indices: Optional[Sequence[int]]) -> Optional[List[int]]:
    if cfg.get("selected_hidden_layer_indices") is not None:
        raw = list(cfg["selected_hidden_layer_indices"])
    else:
        mode = str(cfg.get("hidden_layer_selection") or "middle").strip().lower()
        if mode in {"", "none", "default", "last"}:
            raw = [-1]
        elif mode == "all":
            raw = None
        elif mode == "auto":
            raw = list(ckpt_selected_hidden_layer_indices) if ckpt_selected_hidden_layer_indices is not None else [-1]
        elif mode == "middle":
            raw = [num_hidden_layers // 2]
        elif mode in {"index", "fixed", "manual"}:
            if cfg.get("hidden_layer_index") is None:
                raise ValueError("hidden_layer_selection='index' requires --hidden_layer_index.")
            raw = [int(cfg["hidden_layer_index"])]
        else:
            raise ValueError(f"Unsupported hidden_layer_selection={cfg.get('hidden_layer_selection')!r}. Use one of: last, middle, index, all, auto.")
    resolved = resolve_transformer_layer_indices(raw, num_hidden_layers)
    if raw is not None and resolved is None:
        raise ValueError(f"No valid hidden layers remained after resolving {raw} for a model with num_hidden_layers={num_hidden_layers}.")
    return resolved


def _select_hidden_for_head(*, all_hidden_states: Sequence[torch.Tensor], head: AuxHeadModule) -> Tuple[torch.Tensor, Optional[Sequence[torch.Tensor]]]:
    hidden_states = all_hidden_states if head.requires_all_hidden_states else None
    if head.requires_all_hidden_states:
        return all_hidden_states[-1], hidden_states
    selected = head.selected_hidden_layer_indices
    if selected is None:
        return all_hidden_states[-1], hidden_states
    if len(selected) != 1:
        raise ValueError(
            "For hidden_encoder_type in {'lite','strong_single'}, exactly one hidden layer must be selected. "
            f"Got selected_hidden_layer_indices={selected}."
        )
    layer_idx = int(selected[0])
    return all_hidden_states[layer_idx + 1], hidden_states


def _limit_head_token_mask(batch: Dict[str, Any], max_head_input_tokens: Optional[int]) -> Dict[str, Any]:
    if max_head_input_tokens is None:
        return batch
    max_head_input_tokens = int(max_head_input_tokens)
    if max_head_input_tokens <= 0:
        raise ValueError(f"max_head_input_tokens must be > 0 when set, got {max_head_input_tokens}.")
    token_mask = batch.get("head_token_mask")
    if token_mask is None:
        return batch
    active = token_mask > 0
    active_rank = active.to(torch.long).cumsum(dim=-1)
    limited = active & (active_rank <= max_head_input_tokens)
    batch["head_token_mask"] = limited.to(dtype=token_mask.dtype)
    return batch


class EvalCollator(ChatBatchBuilder):
    def __init__(self, processor: Any, max_seq_len: int, head_input_mode: str, max_head_input_tokens: Optional[int]):
        super().__init__(processor, max_seq_len, head_input_mode)
        self.max_head_input_tokens = max_head_input_tokens

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        messages_batch: List[List[Dict[str, Any]]] = []
        meta: Dict[str, List[Any]] = {"uuid": [], "gold_label": [], "completion": [], "question": [], "tools": []}
        for row_idx, ex in enumerate(examples):
            tools_json = ex.get("tools_json") or tools_to_text(ex.get("tools", []))
            messages_batch.append([
                {"role": "system", "content": ex.get("system_prompt") or build_system_prompt(tools_json)},
                {"role": "user", "content": str(ex["question"])},
                {"role": "assistant", "content": str(ex["completion"])},
            ])
            meta["uuid"].append(ex.get("uuid") or ex.get("sample_id") or f"row_{row_idx}")
            meta["gold_label"].append(ex.get("gold_label") or ex.get("correct_answer"))
            meta["completion"].append(ex.get("completion"))
            meta["question"].append(ex.get("question"))
            meta["tools"].append(ex.get("tools") or ex.get("tools_json"))
        batch = self.build_from_messages(messages_batch, images=[None] * len(messages_batch), labels=None)
        batch = _limit_head_token_mask(batch, self.max_head_input_tokens)
        batch["meta"] = meta
        return batch


@torch.no_grad()
def forward_head(model: Any, head: AuxHeadModule, batch: Dict[str, Any], device: torch.device, fp_dtype: torch.dtype | None) -> torch.Tensor:
    batch = move_batch_to_device(batch, device, fp_dtype)
    token_mask = batch.pop("head_token_mask")
    batch.pop("meta", None)
    keep = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "position_ids", "cache_position", "mm_token_type_ids")
    forward_inputs = {k: batch[k] for k in keep if k in batch}
    resolved_model_family = getattr(model, "_when2call_resolved_model_family", "other")
    forward_inputs = _sanitize_forward_inputs(forward_inputs, resolved_model_family)
    autocast_enabled = device.type == "cuda" and fp_dtype in {torch.float16, torch.bfloat16}
    autocast_dtype = fp_dtype if fp_dtype is not None else torch.bfloat16
    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"
    backbone = getattr(model, "model", None)
    need_hidden_states_for_head = head.requires_all_hidden_states or (head.selected_hidden_layer_indices is not None)
    with torch.autocast(device_type=autocast_device_type, dtype=autocast_dtype, enabled=autocast_enabled):
        out = (
            backbone(**forward_inputs, use_cache=False, return_dict=True, output_hidden_states=need_hidden_states_for_head)
            if backbone is not None else
            model(**forward_inputs, use_cache=False, return_dict=True, output_hidden_states=need_hidden_states_for_head)
        )
        if need_hidden_states_for_head:
            all_hidden_states = out.hidden_states
            last_hidden, hidden_states = _select_hidden_for_head(all_hidden_states=all_hidden_states, head=head)
        else:
            last_hidden = getattr(out, "last_hidden_state", None)
            if last_hidden is None:
                last_hidden = out.hidden_states[-1]
            hidden_states = None
        logits = head(last_hidden=last_hidden, hidden_states=hidden_states, token_mask=token_mask)
        logits = logits if logits.ndim == 2 else logits.view(logits.shape[0], -1)
    return logits.detach().cpu()


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
        metrics["per_class"][name] = {
            "precision": float(prec[i].item()),
            "recall": float(rec[i].item()),
            "f1": float(f1[i].item()),
            "support": int(row[i].item()),
        }
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


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _plot_hist(values_a: List[float], values_b: List[float], label_a: str, label_b: str, title: str, out_path: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    if values_a:
        plt.hist(values_a, bins=30, alpha=0.65, label=label_a)
    if values_b:
        plt.hist(values_b, bins=30, alpha=0.65, label=label_b)
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_confusion(confusion: List[List[int]], class_names: List[str], title: str, out_path: Path) -> None:
    cm = torch.tensor(confusion).numpy()
    plt.figure(figsize=(6.5, 5.5))
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


def head_probs_to_behavior_scores(head_probs: List[float]) -> Dict[str, float]:
    max_head = max(head_probs) if head_probs else 0.0
    return {
        "tool_call": float(head_probs[0]),
        "request_for_info": float(head_probs[1]),
        "cannot_answer": float(head_probs[2]),
        "direct_answer": float(max(0.0, 1.0 - max_head)),
    }


def predict_behavior_from_probs(head_probs: List[float], threshold: float) -> str:
    if max(head_probs) < float(threshold):
        return "direct_answer"
    idx = max(range(len(head_probs)), key=lambda i: head_probs[i])
    return HEAD_CLASS_NAMES[idx]


def main() -> None:
    cfg = parse_args()
    if not cfg["head_checkpoint_path"]:
        raise ValueError("Set --head_checkpoint_path.")

    set_seed(int(cfg["seed"]))
    device = get_device()
    fp_dtype = dtype_from_str(cfg["dtype"])
    outdir = Path(cfg["output_dir"])
    plots_dir = outdir / "plots"
    _ensure_dir(outdir)
    _ensure_dir(plots_dir)

    ckpt = torch.load(cfg["head_checkpoint_path"], map_location="cpu")
    ckpt_cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    if cfg["head_input_mode"] is None:
        cfg["head_input_mode"] = ckpt_cfg.get("head_input_mode", "completion_text_only")
    if cfg["hidden_encoder_type"] is None:
        cfg["hidden_encoder_type"] = ckpt_cfg.get("hidden_encoder_type", "lite")
    if cfg["max_head_input_tokens"] is None and ckpt_cfg.get("max_head_input_tokens") is not None:
        cfg["max_head_input_tokens"] = ckpt_cfg.get("max_head_input_tokens")
    if cfg["selected_hidden_layer_indices"] is None:
        cfg["selected_hidden_layer_indices"] = ckpt_cfg.get("selected_hidden_layer_indices")
    if ckpt_cfg.get("class_names"):
        cfg["class_names"] = list(ckpt_cfg["class_names"])
    if ckpt_cfg.get("behavior_class_names"):
        cfg["behavior_class_names"] = list(ckpt_cfg["behavior_class_names"])
    if ckpt_cfg.get("max_seq_len"):
        cfg["max_seq_len"] = int(ckpt_cfg["max_seq_len"])
    if ckpt_cfg.get("decision_threshold") is not None:
        cfg["decision_threshold"] = float(ckpt_cfg["decision_threshold"])

    class_names = [normalize_class_name(x) for x in cfg["class_names"]]
    behavior_class_names = [normalize_class_name(x) for x in cfg["behavior_class_names"]]
    if class_names != HEAD_CLASS_NAMES:
        raise ValueError(f"Expected 3 head classes {HEAD_CLASS_NAMES}, got {class_names}")
    if behavior_class_names != BEHAVIOR_CLASS_NAMES:
        raise ValueError(f"Expected 4 behavior classes {BEHAVIOR_CLASS_NAMES}, got {behavior_class_names}")
    behavior_to_id = {name: i for i, name in enumerate(behavior_class_names)}

    requested_model_id = cfg["model_name_or_path"]
    resolved_model_family = _infer_model_family(requested_model_id, cfg["model_family"])
    resolved_thinking_enabled = _resolve_thinking_enabled(requested_model_id, resolved_model_family, cfg["thinking_mode"])
    resolved_attn_implementation = _resolve_attn_implementation_for_model(cfg["attn_implementation"], resolved_model_family)
    candidate_model_ids = [requested_model_id]
    if cfg["prefer_unsloth_mirror"] and requested_model_id.startswith("Qwen/"):
        candidate_model_ids = ["unsloth/" + requested_model_id.split("/", 1)[1], requested_model_id]

    model = None
    processor = None
    last_err = None
    loaded_model_id = candidate_model_ids[0]
    for candidate in candidate_model_ids:
        try:
            model, processor = FastVisionModel.from_pretrained(
                candidate,
                max_seq_length=cfg["max_seq_len"],
                load_in_4bit=cfg["load_in_4bit"],
                load_in_8bit=cfg["load_in_8bit"],
                use_gradient_checkpointing=cfg["use_gradient_checkpointing"],
                trust_remote_code=cfg["trust_remote_code"],
                attn_implementation=resolved_attn_implementation,
            )
            loaded_model_id = candidate
            break
        except Exception as e:
            last_err = e
            print(f"[warn] failed model candidate={candidate}: {e}", flush=True)
    if model is None or processor is None:
        raise RuntimeError(f"Could not load any model candidate from {candidate_model_ids}: {last_err}")

    model = model.to(device)
    model.eval()
    FastVisionModel.for_inference(model)
    for p in model.parameters():
        p.requires_grad_(False)

    max_pixels_info = _set_image_processor_max_pixels_safe(processor, cfg["max_pixels"])
    processor, runtime_prompting = _patch_processor_for_runtime_prompting(
        processor=processor,
        model_family=resolved_model_family,
        thinking_enabled=resolved_thinking_enabled,
    )
    setattr(model, "_when2call_resolved_model_family", resolved_model_family)
    save_json(outdir / "runtime_model_prompting.json", {
        "requested_model_id": requested_model_id,
        "resolved_model_id_for_load": loaded_model_id,
        "resolved_model_family": resolved_model_family,
        "requested_thinking_mode": cfg["thinking_mode"],
        "resolved_thinking_enabled": bool(resolved_thinking_enabled),
        "requested_attn_implementation": cfg["attn_implementation"],
        "resolved_attn_implementation": resolved_attn_implementation,
        "patched_targets": runtime_prompting["patched_targets"],
        "max_pixels_info": max_pixels_info,
    })

    hidden_size, num_hidden_layers = infer_hidden_size_and_num_hidden_layers(model)
    resolved_selected_hidden_layers = _resolve_selected_hidden_layers_from_cfg(
        cfg,
        num_hidden_layers=num_hidden_layers,
        ckpt_selected_hidden_layer_indices=ckpt_cfg.get("selected_hidden_layer_indices"),
    )
    cfg["selected_hidden_layer_indices"] = resolved_selected_hidden_layers

    head = AuxHeadModule(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        hidden_encoder_type=cfg["hidden_encoder_type"],
        num_labels=len(class_names),
        selected_hidden_layer_indices=resolved_selected_hidden_layers,
    ).to(device)
    if fp_dtype is not None:
        head = head.to(dtype=fp_dtype)
    state = ckpt.get("head_state", ckpt)
    missing, unexpected = head.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing head keys: {missing}")
    if unexpected:
        print(f"[warn] unexpected head keys: {unexpected}")
    head.eval()

    rows, row_stats = load_generated_rows(cfg["generated_eval_path"], cfg["max_eval_rows"], behavior_class_names)
    if not rows:
        raise ValueError("No supported eval rows remain after filtering.")

    save_json(outdir / "config_used.json", {**cfg, "loaded_model_id": loaded_model_id, "resolved_model_family": resolved_model_family, "resolved_thinking_enabled": bool(resolved_thinking_enabled), "resolved_attn_implementation": resolved_attn_implementation})
    save_json(outdir / "row_filter_stats.json", row_stats)

    loader = DataLoader(
        rows,
        batch_size=cfg["per_device_batch_size"],
        shuffle=False,
        collate_fn=EvalCollator(processor, cfg["max_seq_len"], cfg["head_input_mode"], cfg["max_head_input_tokens"]),
        num_workers=cfg["num_workers"],
        pin_memory=bool(cfg["pin_memory"] and torch.cuda.is_available()),
        persistent_workers=bool(cfg["persistent_workers"] and cfg["num_workers"] > 0),
        drop_last=False,
    )

    gold_ids: List[int] = []
    pred_ids: List[int] = []
    pred_score_correct: List[float] = []
    pred_score_wrong: List[float] = []
    gold_score_correct: List[float] = []
    gold_score_wrong: List[float] = []
    prob_one_vs_rest: Dict[str, List[float]] = {name: [] for name in behavior_class_names}
    per_pred_class_scores: Dict[str, Dict[str, List[float]]] = {
        name: {"correct": [], "wrong": []} for name in behavior_class_names
    }
    records: List[Dict[str, Any]] = []

    for batch in tqdm(loader, desc="head-eval", dynamic_ncols=True):
        meta = batch["meta"]
        logits = forward_head(model, head, batch, device, fp_dtype)
        probs = torch.sigmoid(logits)
        for i in range(probs.shape[0]):
            gold_name = normalize_class_name(meta["gold_label"][i])
            head_prob_vec = [float(x) for x in probs[i].tolist()]
            behavior_scores = head_probs_to_behavior_scores(head_prob_vec)

            pred_name = predict_behavior_from_probs(head_prob_vec, float(cfg["decision_threshold"]))
            pred_id = behavior_to_id[pred_name]
            gold_id = behavior_to_id[gold_name]
            pred_score = float(behavior_scores[pred_name])
            gold_score = float(behavior_scores[gold_name])
            is_correct = int(pred_id == gold_id)

            gold_ids.append(gold_id)
            pred_ids.append(pred_id)
            if is_correct:
                pred_score_correct.append(pred_score)
                gold_score_correct.append(gold_score)
                per_pred_class_scores[pred_name]["correct"].append(pred_score)
            else:
                pred_score_wrong.append(pred_score)
                gold_score_wrong.append(gold_score)
                per_pred_class_scores[pred_name]["wrong"].append(pred_score)
            for cls_name in behavior_class_names:
                prob_one_vs_rest[cls_name].append(float(behavior_scores[cls_name]))

            sorted_scores = sorted(behavior_scores.values(), reverse=True)
            margin = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else float(sorted_scores[0])
            records.append({
                "uuid": meta["uuid"][i],
                "question": meta["question"][i],
                "tools": meta["tools"][i],
                "gold_label": gold_name,
                "generated_completion": meta["completion"][i],
                "head_pred_label": pred_name,
                "head_logits": [float(x) for x in logits[i].tolist()],
                "head_probs": head_prob_vec,
                "behavior_scores": behavior_scores,
                "head_pred_score": pred_score,
                "head_gold_behavior_score": gold_score,
                "head_margin": margin,
                "head_is_correct": is_correct,
            })

    metrics = {
        "num_rows_scored": len(rows),
        "head_class_names": class_names,
        "behavior_class_names": behavior_class_names,
        "row_filter_stats": row_stats,
        "decision_threshold": float(cfg["decision_threshold"]),
        "head_on_generated_completion": confusion_and_metrics(gold_ids, pred_ids, behavior_class_names),
    }
    save_json(outdir / "metrics.json", metrics)

    with open(outdir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    prob_summary = {
        "predicted_behavior_score_correct_mean": float(sum(pred_score_correct) / max(len(pred_score_correct), 1)),
        "predicted_behavior_score_wrong_mean": float(sum(pred_score_wrong) / max(len(pred_score_wrong), 1)),
        "gold_behavior_score_correct_mean": float(sum(gold_score_correct) / max(len(gold_score_correct), 1)),
        "gold_behavior_score_wrong_mean": float(sum(gold_score_wrong) / max(len(gold_score_wrong), 1)),
        "per_predicted_class": {},
    }
    for cls_name in behavior_class_names:
        c_vals = per_pred_class_scores[cls_name]["correct"]
        w_vals = per_pred_class_scores[cls_name]["wrong"]
        prob_summary["per_predicted_class"][cls_name] = {
            "num_correct_predictions": len(c_vals),
            "num_wrong_predictions": len(w_vals),
            "mean_prob_when_correct": float(sum(c_vals) / max(len(c_vals), 1)),
            "mean_prob_when_wrong": float(sum(w_vals) / max(len(w_vals), 1)),
        }
    save_json(outdir / "probability_summary.json", prob_summary)

    _plot_hist(pred_score_correct, pred_score_wrong, "correct", "wrong", "Predicted behavior score by correctness", plots_dir / "predicted_behavior_score_by_correctness.png")
    _plot_hist(gold_score_correct, gold_score_wrong, "correct", "wrong", "Gold behavior score by correctness", plots_dir / "gold_behavior_score_by_correctness.png")
    for cls_name in behavior_class_names:
        plt.figure(figsize=(7, 4.5))
        plt.hist(prob_one_vs_rest[cls_name], bins=30)
        plt.xlabel(f"score({cls_name})")
        plt.ylabel("Count")
        plt.title(f"Behavior score distribution for {cls_name}")
        plt.tight_layout()
        plt.savefig(plots_dir / f"prob_{cls_name}_one_vs_rest.png", dpi=160)
        plt.close()
        _plot_hist(
            per_pred_class_scores[cls_name]["correct"],
            per_pred_class_scores[cls_name]["wrong"],
            "predicted correctly",
            "predicted wrongly",
            f"{cls_name}: predicted-class score, correct vs wrong",
            plots_dir / f"prob_{cls_name}_predicted_correct_vs_wrong.png",
        )
    _plot_confusion(metrics["head_on_generated_completion"]["confusion"], behavior_class_names, "Head confusion matrix", plots_dir / "head_confusion.png")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved metrics to {outdir / 'metrics.json'}")
    print(f"saved predictions to {outdir / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
