#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
for p in [current_dir, base_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import math
import json
import argparse
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import datasets
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from unsloth import FastVisionModel

from aux_head_shared_utils import (
    AuxHeadModule,
    ChatBatchBuilder,
    count_params,
    dtype_from_str,
    filter_dataset_by_subset_name,
    get_device,
    infer_hidden_size_and_num_hidden_layers,
    load_config,
    load_dataset_auto,
    move_batch_to_device,
    resolve_transformer_layer_indices,
    save_json,
    set_seed,
)

HEAD_CLASS_NAMES = ["tool_call", "request_for_info", "cannot_answer"]
BEHAVIOR_CLASS_NAMES = ["tool_call", "request_for_info", "cannot_answer", "direct_answer"]
BEHAVIOR_TO_ID = {name: i for i, name in enumerate(BEHAVIOR_CLASS_NAMES)}
HEAD_TARGETS = {
    "tool_call": [1.0, 0.0, 0.0],
    "request_for_info": [0.0, 1.0, 0.0],
    "cannot_answer": [0.0, 0.0, 1.0],
    "direct_answer": [0.0, 0.0, 0.0],
}


@dataclass
class CFG:
    model_name_or_path: str = "Qwen/Qwen3-VL-4B-Instruct"
    trust_remote_code: bool = True
    attn_implementation: str = "flash_attention_3"
    prefer_unsloth_mirror: bool = True
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    use_gradient_checkpointing: str = "unsloth"
    dtype: str = "bf16"
    model_family: str = "auto"
    thinking_mode: Any = "auto"

    dataset_path: str = ""
    subset_name: Optional[str] = None
    output_dir: str = "./out_when2call_head_4class_3sigmoid"
    seed: int = 42

    per_device_batch_size: int = 1
    grad_accum_steps: int = 16
    num_epochs: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.10

    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = False

    max_seq_len: int = 8192
    max_pixels: int = 200_000
    max_head_input_tokens: Optional[int] = None
    head_input_mode: str = "completion_text_only"
    hidden_encoder_type: str = "lite"

    selected_hidden_layer_indices: Optional[list[int]] = None
    hidden_layer_selection: Optional[str] = "middle"
    hidden_layer_index: Optional[int] = None

    label_column: str = "behavior_class"
    label_name_column: str = "behavior"
    usable_column: Optional[str] = "usable_behavior"
    drop_unusable_rows: bool = True

    class_names: list[str] = field(default_factory=lambda: list(HEAD_CLASS_NAMES))
    behavior_class_names: list[str] = field(default_factory=lambda: list(BEHAVIOR_CLASS_NAMES))
    decision_threshold: float = 0.5

    # Interpreted as positive-class weights for BCE.
    class_weight_strategy: str = "auto"  # auto | none
    class_weights: Optional[list[float]] = None
    max_class_weight: float = 20.0
    min_class_weight: float = 1.0

    use_weighted_sampler: bool = False
    sampler_power: float = 1.0

    log_every: int = 10
    save_every: int = 1000

    wandb_enabled: bool = True
    wandb_project: str = "qwen3vl-aux-head"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tags: Optional[list[str]] = None


def load_cfg(path: str) -> CFG:
    cfg = CFG()
    raw_cfg = load_config(path)
    for k, v in raw_cfg.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    if cfg.max_head_input_tokens is None and raw_cfg.get("max_input_tokens") is not None:
        cfg.max_head_input_tokens = int(raw_cfg["max_input_tokens"])
    if cfg.wandb_tags is not None and not isinstance(cfg.wandb_tags, list):
        cfg.wandb_tags = [str(cfg.wandb_tags)]
    return cfg



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
        "ambiguous": "ambiguous",
        "uncertain": "ambiguous",
    }
    return aliases.get(s, s)


def _resolve_selected_hidden_layer_indices_from_cfg(cfg: CFG, num_hidden_layers: int) -> Optional[list[int]]:
    if cfg.selected_hidden_layer_indices is not None:
        raw = list(cfg.selected_hidden_layer_indices)
    else:
        mode = str(cfg.hidden_layer_selection or "middle").strip().lower()
        if mode in {"", "none", "default", "last"}:
            raw = [-1]
        elif mode == "middle":
            raw = [num_hidden_layers // 2]
        elif mode in {"index", "fixed", "manual"}:
            if cfg.hidden_layer_index is None:
                raise ValueError("hidden_layer_selection='index' requires hidden_layer_index to be set.")
            raw = [int(cfg.hidden_layer_index)]
        elif mode == "all":
            raw = None
        else:
            raise ValueError(
                f"Unsupported hidden_layer_selection={cfg.hidden_layer_selection!r}. Use one of: last, middle, index, all."
            )
    resolved = resolve_transformer_layer_indices(raw, num_hidden_layers)
    if raw is not None and resolved is None:
        raise ValueError(
            f"No valid hidden layers remained after resolving {raw} for a model with num_hidden_layers={num_hidden_layers}."
        )
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


def load_dataset_auto_robust(path: str):
    p = Path(path)
    try:
        return load_dataset_auto(path)
    except Exception as e:
        load_err = e
    if not p.is_dir():
        raise load_err
    from glob import glob
    import pyarrow as pa
    import pyarrow.parquet as pq
    parquet_files = sorted(glob(str(p / "**" / "*.parquet"), recursive=True))
    if not parquet_files:
        raise load_err
    print(f"[warn] default parquet loading failed; falling back to robust shard merge for {len(parquet_files)} files", flush=True)
    tables = []
    col_order = []
    col_types = {}
    for fp in parquet_files:
        table = pq.read_table(fp)
        tables.append(table)
        for field in table.schema:
            name = field.name
            if name not in col_types:
                col_types[name] = field.type
                col_order.append(name)
    norm_tables = []
    for table in tables:
        n = table.num_rows
        arrays = []
        for name in col_order:
            if name in table.column_names:
                arrays.append(table[name])
            else:
                arrays.append(pa.nulls(n, type=col_types[name]))
        norm_tables.append(pa.table(arrays, names=col_order))
    merged = pa.concat_tables(norm_tables)
    ds = datasets.Dataset.from_dict(merged.to_pydict())
    return datasets.DatasetDict({"train": ds})


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(f"<{item.get('type', 'content')}>")
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p).strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text", ""))
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _parse_messages_from_example(ex: Dict[str, Any]) -> List[Dict[str, Any]]:
    if ex.get("messages_json"):
        raw = json.loads(ex["messages_json"])
        return [{"role": str(msg.get("role", "user")), "content": _content_to_text(msg.get("content", ""))} for msg in raw]
    if ex.get("messages") is not None:
        raw = ex["messages"]
        return [{"role": str(msg.get("role", "user")), "content": _content_to_text(msg.get("content", ""))} for msg in raw]
    system_prompt = str(ex.get("system_prompt") or "")
    prompt = str(ex.get("prompt") or ex.get("question") or ex.get("prompt_text") or ex.get("context_text") or "")
    completion = str(ex.get("completion") or "")
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": completion})
    return messages


def derive_supervision(ex: Dict[str, Any], cfg: CFG) -> Tuple[int, List[float], str, int]:
    usable = 1
    if cfg.usable_column and cfg.usable_column in ex:
        try:
            usable = int(ex[cfg.usable_column])
        except Exception:
            usable = 0

    name_candidates = [cfg.label_name_column, "behavior", "latent_category", "gold_label", "correct_answer"]
    for field in name_candidates:
        if not field or field not in ex:
            continue
        name = normalize_class_name(ex.get(field))
        if not name:
            continue
        if name == "ambiguous":
            return 0, [0.0, 0.0, 0.0], "ambiguous", -100
        if name in HEAD_TARGETS:
            if cfg.usable_column is None:
                usable = 1
            return int(usable), list(HEAD_TARGETS[name]), name, (BEHAVIOR_TO_ID[name] if usable > 0 else -100)

    if all(k in ex for k in ("label_tool_call", "label_request_for_info", "label_cannot_answer")):
        targets = [float(ex.get("label_tool_call", 0) or 0), float(ex.get("label_request_for_info", 0) or 0), float(ex.get("label_cannot_answer", 0) or 0)]
        if sum(int(x > 0.5) for x in targets) == 0:
            behavior_name = "direct_answer"
        else:
            behavior_name = cfg.class_names[int(max(range(len(targets)), key=lambda i: targets[i]))]
        return int(usable), targets, behavior_name, (BEHAVIOR_TO_ID[behavior_name] if usable > 0 else -100)

    if cfg.label_column in ex and ex[cfg.label_column] is not None and str(ex[cfg.label_column]) != "":
        try:
            y = int(ex[cfg.label_column])
            if 0 <= y < len(cfg.behavior_class_names):
                behavior_name = normalize_class_name(cfg.behavior_class_names[y])
                if behavior_name in HEAD_TARGETS:
                    return int(usable), list(HEAD_TARGETS[behavior_name]), behavior_name, (y if usable > 0 else -100)
        except Exception:
            pass

    return 0, [0.0, 0.0, 0.0], "ambiguous", -100


class MultiLabelBehaviorCollator(ChatBatchBuilder):
    def __init__(self, processor: Any, cfg: CFG):
        super().__init__(processor, cfg.max_seq_len, cfg.head_input_mode)
        self.cfg = cfg

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        messages_batch: List[List[Dict[str, Any]]] = []
        images: List[Optional[Any]] = []
        targets: List[List[float]] = []
        usable: List[float] = []
        behavior_ids: List[int] = []
        for ex in examples:
            messages_batch.append(_parse_messages_from_example(ex))
            images.append(None)
            u, t, _, behavior_id = derive_supervision(ex, self.cfg)
            usable.append(float(u))
            targets.append([float(x) for x in t])
            behavior_ids.append(int(behavior_id))
        batch = self.build_from_messages(messages_batch, images, labels=None)
        batch = _limit_head_token_mask(batch, self.cfg.max_head_input_tokens)
        batch["aux_targets"] = torch.tensor(targets, dtype=torch.float32)
        batch["aux_usable_mask"] = torch.tensor(usable, dtype=torch.float32)
        batch["aux_behavior_ids"] = torch.tensor(behavior_ids, dtype=torch.long)
        return batch


def lr_scale(step: int, total: int, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return min_ratio + (1.0 - min_ratio) * cosine


def save_checkpoint(head, cfg: CFG, outdir: Path, step: int, name: str) -> Path:
    path = outdir / name
    torch.save({"step": step, "head_state": head.state_dict(), "cfg": asdict(cfg)}, path)
    return path


def compute_pos_weights(ds, cfg: CFG) -> torch.Tensor:
    num_classes = len(cfg.class_names)
    if cfg.class_weights is not None:
        if len(cfg.class_weights) != num_classes:
            raise ValueError("class_weights length must equal number of head classes")
        return torch.tensor([float(x) for x in cfg.class_weights], dtype=torch.float32)
    if str(cfg.class_weight_strategy).lower() == "none":
        return torch.ones(num_classes, dtype=torch.float32)
    pos_counts = torch.zeros(num_classes, dtype=torch.float64)
    usable_n = 0.0
    for ex in ds:
        u, targets, _, _ = derive_supervision(ex, cfg)
        if u <= 0:
            continue
        usable_n += 1.0
        pos_counts += torch.tensor(targets, dtype=torch.float64)
    if usable_n <= 0:
        return torch.ones(num_classes, dtype=torch.float32)
    weights = []
    for c in pos_counts.tolist():
        neg = max(usable_n - c, 0.0)
        if c <= 0:
            w = float(cfg.max_class_weight)
        else:
            w = neg / c
            w = min(float(cfg.max_class_weight), max(float(cfg.min_class_weight), float(w)))
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


def compute_behavior_sample_weights(ds, cfg: CFG) -> Dict[int, float]:
    counts = {i: 0.0 for i in range(len(cfg.behavior_class_names))}
    usable_n = 0.0
    for ex in ds:
        u, _, _, behavior_id = derive_supervision(ex, cfg)
        if u <= 0 or behavior_id < 0:
            continue
        usable_n += 1.0
        counts[int(behavior_id)] += 1.0
    if usable_n <= 0:
        return {i: 1.0 for i in range(len(cfg.behavior_class_names))}
    weights = {}
    for i in range(len(cfg.behavior_class_names)):
        c = counts.get(i, 0.0)
        if c <= 0:
            w = float(cfg.max_class_weight)
        else:
            w = usable_n / (len(cfg.behavior_class_names) * c)
            w = min(float(cfg.max_class_weight), max(float(cfg.min_class_weight), float(w)))
        weights[i] = w
    return weights


def build_sample_weights(ds, cfg: CFG) -> torch.Tensor:
    behavior_weights = compute_behavior_sample_weights(ds, cfg)
    weights: List[float] = []
    power = float(cfg.sampler_power)
    for ex in ds:
        u, _, _, behavior_id = derive_supervision(ex, cfg)
        if u <= 0 or behavior_id < 0:
            weights.append(1e-6)
            continue
        weights.append(max(float(behavior_weights[int(behavior_id)]), 1e-6) ** power)
    return torch.tensor(weights, dtype=torch.float64)


def masked_multilabel_bce_loss(logits: torch.Tensor, targets: torch.Tensor, usable_mask: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    per_elem = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    per_example = per_elem.mean(dim=-1)
    weight = usable_mask.to(per_example.dtype)
    return (per_example * weight).sum() / weight.sum().clamp_min(1e-8)


def project_behavior_from_probs(probs: torch.Tensor, threshold: float) -> torch.Tensor:
    if probs.ndim != 2 or probs.shape[1] != len(HEAD_CLASS_NAMES):
        raise ValueError(f"Expected probs of shape [B, {len(HEAD_CLASS_NAMES)}], got {tuple(probs.shape)}")
    pred = torch.argmax(probs, dim=-1)
    any_active = (probs >= threshold).any(dim=-1)
    direct_id = BEHAVIOR_TO_ID["direct_answer"]
    pred = pred.where(any_active, torch.full_like(pred, direct_id))
    return pred


def confusion_and_metrics(conf: torch.Tensor, class_names: Sequence[str]) -> Dict[str, float]:
    conf = conf.float()
    tp = torch.diag(conf)
    row = conf.sum(dim=1)
    col = conf.sum(dim=0)
    prec = tp / torch.clamp(col, min=1.0)
    rec = tp / torch.clamp(row, min=1.0)
    f1 = 2 * prec * rec / torch.clamp(prec + rec, min=1e-8)
    total = conf.sum().clamp(min=1.0)
    out: Dict[str, Any] = {
        "acc": (tp.sum() / total).item(),
        "macro_precision": prec.mean().item(),
        "macro_recall": rec.mean().item(),
        "macro_f1": f1.mean().item(),
        "confusion": conf.long().tolist(),
        "class_names": list(class_names),
    }
    for i, name in enumerate(class_names):
        out[f"{name}_precision"] = prec[i].item()
        out[f"{name}_recall"] = rec[i].item()
        out[f"{name}_f1"] = f1[i].item()
        out[f"{name}_support"] = row[i].item()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_cfg(ap.parse_args().config)

    device = get_device()
    fp_dtype = dtype_from_str(cfg.dtype)
    set_seed(int(cfg.seed))

    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    wb = None
    if cfg.wandb_enabled:
        import wandb
        wb = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            group=cfg.wandb_group,
            tags=cfg.wandb_tags,
            config=asdict(cfg),
            settings=wandb.Settings(start_method="thread"),
        )

    requested_model_id = cfg.model_name_or_path
    resolved_model_family = _infer_model_family(requested_model_id, cfg.model_family)
    resolved_thinking_enabled = _resolve_thinking_enabled(requested_model_id, resolved_model_family, cfg.thinking_mode)
    resolved_attn_implementation = _resolve_attn_implementation_for_model(cfg.attn_implementation, resolved_model_family)
    candidate_model_ids = [requested_model_id]
    if cfg.prefer_unsloth_mirror and requested_model_id.startswith("Qwen/"):
        candidate_model_ids = ["unsloth/" + requested_model_id.split("/", 1)[1], requested_model_id]

    last_err = None
    model = None
    processor = None
    model_id = candidate_model_ids[0]
    for candidate in candidate_model_ids:
        try:
            model, processor = FastVisionModel.from_pretrained(
                candidate,
                max_seq_length=cfg.max_seq_len,
                load_in_4bit=cfg.load_in_4bit,
                load_in_8bit=cfg.load_in_8bit,
                use_gradient_checkpointing=cfg.use_gradient_checkpointing,
                trust_remote_code=cfg.trust_remote_code,
                attn_implementation=resolved_attn_implementation,
            )
            model_id = candidate
            break
        except Exception as e:
            last_err = e
            print(f"[warn] failed to load model candidate={candidate}: {e}", flush=True)
    if model is None or processor is None:
        raise RuntimeError(f"Could not load any model candidate from {candidate_model_ids}: {last_err}")

    model = model.to(device)
    FastVisionModel.for_inference(model)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    max_pixels_info = _set_image_processor_max_pixels_safe(processor, cfg.max_pixels)
    processor, runtime_prompting = _patch_processor_for_runtime_prompting(
        processor=processor,
        model_family=resolved_model_family,
        thinking_enabled=resolved_thinking_enabled,
    )
    save_json(outdir / "runtime_model_prompting.json", {
        "requested_model_id": requested_model_id,
        "resolved_model_id_for_load": model_id,
        "resolved_model_family": resolved_model_family,
        "requested_thinking_mode": cfg.thinking_mode,
        "resolved_thinking_enabled": bool(resolved_thinking_enabled),
        "requested_attn_implementation": cfg.attn_implementation,
        "resolved_attn_implementation": resolved_attn_implementation,
        "patched_targets": runtime_prompting["patched_targets"],
        "max_pixels_info": max_pixels_info,
    })

    ds = load_dataset_auto_robust(cfg.dataset_path)
    train_ds = ds["train"] if isinstance(ds, datasets.DatasetDict) else ds
    train_ds = filter_dataset_by_subset_name(train_ds, cfg.subset_name).shuffle(seed=int(cfg.seed))
    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty after filtering.")

    if cfg.drop_unusable_rows:
        keep_idx: List[int] = []
        usable_counts = 0
        for i, ex in enumerate(train_ds):
            u, _, _, behavior_id = derive_supervision(ex, cfg)
            if u > 0 and behavior_id >= 0:
                keep_idx.append(i)
                usable_counts += 1
        if usable_counts == 0:
            raise ValueError("No usable labeled rows remain after filtering")
        train_ds = train_ds.select(keep_idx)

    pos_weights = compute_pos_weights(train_ds, cfg)

    hidden_size, num_hidden_layers = infer_hidden_size_and_num_hidden_layers(model)
    resolved_selected_hidden_layer_indices = _resolve_selected_hidden_layer_indices_from_cfg(cfg, num_hidden_layers)
    cfg.selected_hidden_layer_indices = resolved_selected_hidden_layer_indices
    save_json(outdir / "config_used.json", {**asdict(cfg), "resolved_model_family": resolved_model_family, "resolved_thinking_enabled": bool(resolved_thinking_enabled), "resolved_attn_implementation": resolved_attn_implementation})

    head = AuxHeadModule(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        hidden_encoder_type=cfg.hidden_encoder_type,
        num_labels=len(cfg.class_names),
        selected_hidden_layer_indices=resolved_selected_hidden_layer_indices,
    ).to(device)
    if fp_dtype is not None:
        head = head.to(dtype=fp_dtype)

    total_backbone, train_backbone = count_params(model)
    total_head, train_head = count_params(head)

    behavior_counts = {name: 0 for name in cfg.behavior_class_names}
    usable_rows = 0
    for ex in train_ds:
        u, _, behavior_name, behavior_id = derive_supervision(ex, cfg)
        if u > 0 and behavior_id >= 0:
            usable_rows += 1
            behavior_counts[behavior_name] += 1

    print(f"model={model_id}")
    print(f"dataset rows={len(train_ds)} usable_rows={usable_rows}")
    print(f"head_class_names={cfg.class_names}")
    print(f"behavior_class_names={cfg.behavior_class_names}")
    print(f"behavior_counts={behavior_counts}")
    print(f"positive_class_weights={pos_weights.tolist()}")
    print(f"decision_threshold={cfg.decision_threshold}")
    print(f"max_head_input_tokens={cfg.max_head_input_tokens if cfg.max_head_input_tokens is not None else 'all'}")
    print(f"backbone params: total={total_backbone:,} trainable={train_backbone:,}")
    print(f"head params: total={total_head:,} trainable={train_head:,}")
    if head.selected_hidden_layer_indices is not None:
        print(f"selected_hidden_layer_indices={head.selected_hidden_layer_indices}")

    sampler = None
    shuffle = True
    if cfg.use_weighted_sampler:
        sample_weights = build_sample_weights(train_ds, cfg)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False

    loader = DataLoader(
        train_ds,
        batch_size=cfg.per_device_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=MultiLabelBehaviorCollator(processor, cfg),
        num_workers=cfg.num_workers,
        pin_memory=bool(cfg.pin_memory and torch.cuda.is_available()),
        persistent_workers=bool(cfg.persistent_workers and cfg.num_workers > 0),
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    optimizer.zero_grad(set_to_none=True)

    steps_per_epoch = math.ceil(len(loader) / cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    pbar = tqdm(total=total_steps, desc="train", dynamic_ncols=True)

    autocast_enabled = device.type == "cuda" and fp_dtype in {torch.float16, torch.bfloat16}
    autocast_dtype = fp_dtype if fp_dtype is not None else torch.bfloat16
    backbone = getattr(model, "model", None)
    need_hidden_states_for_head = head.requires_all_hidden_states or (head.selected_hidden_layer_indices is not None)
    pos_weights_device = pos_weights.to(device)

    loss_sum = 0.0
    loss_n = 0
    opt_step = 0
    confusion = torch.zeros(len(cfg.behavior_class_names), len(cfg.behavior_class_names), dtype=torch.long)

    def maybe_log(opt_step_now: int) -> None:
        nonlocal loss_sum, loss_n
        if cfg.log_every <= 0 or opt_step_now % cfg.log_every != 0:
            return
        md = confusion_and_metrics(confusion, cfg.behavior_class_names)
        logs: Dict[str, float] = {
            "step": float(opt_step_now),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "loss": loss_sum / max(1, loss_n),
            "acc": float(md["acc"]),
            "macro_precision": float(md["macro_precision"]),
            "macro_recall": float(md["macro_recall"]),
            "macro_f1": float(md["macro_f1"]),
        }
        for name in cfg.behavior_class_names:
            logs[f"{name}_precision"] = float(md[f"{name}_precision"])
            logs[f"{name}_recall"] = float(md[f"{name}_recall"])
            logs[f"{name}_f1"] = float(md[f"{name}_f1"])
            logs[f"{name}_support"] = float(md[f"{name}_support"])
        pbar.set_postfix({"loss": f"{logs['loss']:.4f}", "macro_f1": f"{logs['macro_f1']:.4f}"})
        if wb is not None:
            wb.log(logs, step=opt_step_now)
        else:
            print(json.dumps(logs, ensure_ascii=False), flush=True)
        loss_sum = 0.0
        loss_n = 0

    def maybe_save(opt_step_now: int) -> None:
        if cfg.save_every > 0 and opt_step_now % cfg.save_every == 0:
            ckpt = save_checkpoint(head, cfg, outdir, opt_step_now, f"head-step-{opt_step_now}.pt")
            print(f"[save] {ckpt}", flush=True)

    def optimizer_step_from_accum(accum_steps: int) -> int:
        nonlocal opt_step, accum
        if accum_steps <= 0:
            return 0
        if cfg.max_grad_norm and cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.max_grad_norm)
        opt_step += 1
        scale = lr_scale(opt_step, total_steps, warmup_steps, cfg.min_lr_ratio)
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.lr * scale
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        pbar.update(1)
        maybe_log(opt_step)
        maybe_save(opt_step)
        return 1

    for _ in range(cfg.num_epochs):
        accum = 0
        for batch in loader:
            accum += 1
            batch = move_batch_to_device(batch, device, fp_dtype)

            targets = batch.pop("aux_targets").to(dtype=torch.float32)
            usable_mask = batch.pop("aux_usable_mask").to(dtype=torch.float32)
            behavior_ids = batch.pop("aux_behavior_ids").to(dtype=torch.long)
            token_mask = batch.pop("head_token_mask")

            forward_inputs = {k: batch[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "position_ids", "cache_position", "mm_token_type_ids") if k in batch}
            forward_inputs = _sanitize_forward_inputs(forward_inputs, resolved_model_family)

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
                    out = (
                        backbone(**forward_inputs, use_cache=False, return_dict=True, output_hidden_states=need_hidden_states_for_head)
                        if backbone is not None
                        else model(**forward_inputs, use_cache=False, return_dict=True, output_hidden_states=need_hidden_states_for_head)
                    )
                if need_hidden_states_for_head:
                    all_hidden_states = out.hidden_states
                    last_hidden, hidden_states = _select_hidden_for_head(all_hidden_states=all_hidden_states, head=head)
                else:
                    last_hidden = getattr(out, "last_hidden_state", None)
                    if last_hidden is None:
                        last_hidden = out.hidden_states[-1]
                    hidden_states = None

            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
                logits = head(last_hidden=last_hidden, hidden_states=hidden_states, token_mask=token_mask)
                logits = logits if logits.ndim == 2 else logits.view(logits.shape[0], -1)
                loss = masked_multilabel_bce_loss(logits=logits, targets=targets, usable_mask=usable_mask, pos_weight=pos_weights_device)

            (loss / cfg.grad_accum_steps).backward()
            loss_sum += float(loss.detach().cpu().item())
            loss_n += 1

            probs = torch.sigmoid(logits.detach()).cpu()
            preds = project_behavior_from_probs(probs, float(cfg.decision_threshold)).to(torch.long)
            behavior_ids_cpu = behavior_ids.detach().cpu().to(torch.long)
            usable_cpu = usable_mask.detach().cpu().to(torch.long)
            for y, yhat, u in zip(behavior_ids_cpu.tolist(), preds.tolist(), usable_cpu.tolist()):
                if int(u) <= 0 or int(y) < 0:
                    continue
                confusion[int(y), int(yhat)] += 1

            if accum >= cfg.grad_accum_steps:
                optimizer_step_from_accum(accum)

        if accum > 0:
            optimizer_step_from_accum(accum)

    pbar.close()

    final_metrics: Dict[str, Any] = confusion_and_metrics(confusion, cfg.behavior_class_names)
    final_metrics["steps"] = opt_step
    final_metrics["head_class_names"] = cfg.class_names
    final_metrics["behavior_class_names"] = cfg.behavior_class_names
    final_metrics["positive_class_weights"] = pos_weights.tolist()
    final_metrics["behavior_counts"] = behavior_counts
    final_metrics["decision_threshold"] = float(cfg.decision_threshold)
    save_json(outdir / "final_metrics.json", final_metrics)
    save_checkpoint(head, cfg, outdir, opt_step, "head-final.pt")
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2), flush=True)

    if wb is not None:
        wb.summary.update(final_metrics)
        wb.finish()


if __name__ == "__main__":
    main()
