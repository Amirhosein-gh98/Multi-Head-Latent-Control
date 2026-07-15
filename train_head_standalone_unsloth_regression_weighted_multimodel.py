#layer selection + failure weight on top of autoweight
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import math
import re
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Any, Dict, List

import datasets
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from unsloth import FastVisionModel


import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
for p in [current_dir, base_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from aux_head_shared_utils import (
    AuxHeadModule,
    VlmCollator,
    compute_metrics_from_confusion,
    count_params,
    dtype_from_str,
    filter_dataset_by_subset_name,
    get_device,
    infer_hidden_size_and_num_hidden_layers,
    load_config,
    load_dataset_auto,
    move_batch_to_device,
    save_json,
    set_seed,
)


@dataclass
class CFG:
    model_name_or_path: str = ""
    trust_remote_code: bool = True
    attn_implementation: str = "flash_attention_3"
    prefer_unsloth_mirror: bool = True
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    use_gradient_checkpointing: str = "unsloth"
    dtype: str = "bf16"

    model_family: str = "auto"   # auto | qwen3_5 | qwen3 | qwen3_vl | gemma4
    thinking_mode: str = "auto"  # auto | on | off

    dataset_path: str = ""
    aux_label_column: str = "correctness_score"
    subset_name: Optional[str] = None
    output_dir: str = "./out"
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

    max_seq_len: int = 32000
    max_pixels: int = 200_000
    num_labels: int = 1
    head_input_mode: str = "completion_text_only"
    hidden_encoder_type: str = "lite"

    # Backward-compatible raw hidden_states indices.
    # If this is set, it overrides the user-friendly selection below.
    selected_hidden_layer_indices: Optional[list[int]] = None

    # User-friendly layer selection:
    # None | "first" | "middle" | "last" | "index" | "indices" | "all"
    hidden_layer_selection: Optional[str] = "last"

    # These are transformer-layer indices, not raw hidden_states indices.
    # 0 = first transformer block, -1 = last transformer block.
    hidden_layer_index: Optional[int] = None
    hidden_layer_indices: Optional[list[int]] = None

    regression_loss: str = "mse"  # mse | smooth_l1 | bce
    threshold_for_logging: float = 0.5

    # Failure-aware training
    failure_threshold: float = 0.5
    class_weight_strategy: str = "auto"   # "manual" | "auto"
    failure_weight: float = 3.0           # used only when class_weight_strategy == "manual"
    success_weight: float = 1.0           # used only when class_weight_strategy == "manual"

    # Multipliers applied on top of the resolved auto/manual weights.
    failure_weight_multiplier: float = 1.0
    success_weight_multiplier: float = 1.0

    min_class_weight: float = 1.0
    max_class_weight: float = 10.0
    severity_power: float = 1.0
    use_weighted_sampler: bool = True

    log_every: int = 2
    save_every: int = 1000

    wandb_enabled: bool = True
    wandb_project: str = "qwen3vl-aux-head"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tags: Optional[list[str]] = None


def _maybe_add_forward_key(dst: Dict[str, Any], batch: Dict[str, Any], key: str) -> None:
    if key not in batch:
        return
    v = batch[key]
    if v is None:
        return
    if isinstance(v, bool):
        return
    if torch.is_tensor(v):
        if v.ndim == 0 and v.dtype == torch.bool:
            return
        if v.numel() == 0:
            return
    dst[key] = v


def load_cfg(path: str) -> CFG:
    cfg = CFG()
    for k, v in load_config(path).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    if isinstance(cfg.thinking_mode, bool):
        cfg.thinking_mode = "on" if cfg.thinking_mode else "off"
    elif cfg.thinking_mode is None:
        cfg.thinking_mode = "auto"
    else:
        cfg.thinking_mode = str(cfg.thinking_mode).strip().lower()

    if cfg.model_family is None:
        cfg.model_family = "auto"
    else:
        cfg.model_family = str(cfg.model_family).strip().lower()

    if cfg.wandb_tags is not None and not isinstance(cfg.wandb_tags, list):
        cfg.wandb_tags = [str(cfg.wandb_tags)]
    return cfg



def _infer_model_family(model_id: str, requested_family: str = "auto") -> str:
    requested_family = (requested_family or "auto").strip().lower()
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
        return thinking_mode

    mode = str(thinking_mode or "auto").strip().lower()
    if mode in {"on", "true", "1", "yes"}:
        return True
    if mode in {"off", "false", "0", "no"}:
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


def _patch_gemma_messages(messages: List[Dict[str, Any]], thinking_enabled: bool):
    patched: List[Dict[str, Any]] = []
    for msg in messages:
        cloned = dict(msg)
        if isinstance(cloned.get("content"), list):
            cloned["content"] = list(cloned["content"])
        patched.append(cloned)

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


def _patch_chat_template_callable(bound_callable, model_family: str, thinking_enabled: bool):
    def patched(messages, *args, **kwargs):
        if model_family == "gemma4":
            messages = _patch_gemma_messages(messages, thinking_enabled)
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

    runtime_info = {
        "resolved_model_family": model_family,
        "resolved_thinking_enabled": bool(thinking_enabled),
        "patched_targets": patched_targets,
    }
    return processor, runtime_info


def _resolve_model_id_for_load(requested_model_id: str, resolved_model_family: str, prefer_unsloth_mirror: bool) -> str:
    model_id = str(requested_model_id or "")
    if prefer_unsloth_mirror and model_id.startswith("Qwen/") and resolved_model_family == "qwen3_vl":
        return "unsloth/" + model_id.split("/", 1)[1]
    return model_id


def _resolve_attn_implementation(attn_implementation: str, resolved_model_family: str) -> str:
    attn = str(attn_implementation or "").strip()
    if resolved_model_family == "gemma4" and attn == "flash_attention_3":
        return "sdpa"
    return attn


def _safe_set_max_pixels(processor_like, max_pixels: int) -> None:
    targets = []
    if processor_like is not None:
        targets.append(processor_like)
        ip = getattr(processor_like, "image_processor", None)
        if ip is not None:
            targets.append(ip)
    for target in targets:
        try:
            if hasattr(target, "max_pixels"):
                setattr(target, "max_pixels", int(max_pixels))
                return
        except Exception:
            pass
        for attr in ("size", "image_size"):
            obj = getattr(target, attr, None)
            if isinstance(obj, dict):
                obj.setdefault("max_pixels", int(max_pixels))


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


def _to_regression_scores(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2 and logits.shape[-1] == 1:
        logits = logits[:, 0]
    elif logits.ndim != 1:
        raise ValueError(f"Expected logits shape [B] or [B,1] for regression, got {tuple(logits.shape)}")
    return torch.sigmoid(logits)


def _normalize_hidden_state_index(idx: int, num_hidden_layers: int) -> int:
    """
    Normalize a raw hidden_states index.
    hidden_states usually has length = num_hidden_layers + 1
    because index 0 is the embedding output.
    """
    num_states = int(num_hidden_layers) + 1
    idx = int(idx)
    if idx < 0:
        idx = num_states + idx
    if idx < 0 or idx >= num_states:
        raise ValueError(
            f"hidden state index {idx} is out of range for {num_states} hidden states "
            f"(valid raw hidden_states indices: 0..{num_states - 1}, negatives allowed)"
        )
    return idx


def _normalize_transformer_layer_index(idx: int, num_hidden_layers: int) -> int:
    """
    Convert a transformer-layer index into a raw hidden_states index.
    User-facing convention:
      0  -> first transformer block
      -1 -> last transformer block
    Returned value is the corresponding hidden_states index, so +1 offset
    is applied because hidden_states[0] is usually the embedding output.
    """
    n = int(num_hidden_layers)
    idx = int(idx)
    if idx < 0:
        idx = n + idx
    if idx < 0 or idx >= n:
        raise ValueError(
            f"transformer layer index {idx} is out of range for {n} layers "
            f"(valid transformer-layer indices: 0..{n - 1}, negatives allowed)"
        )
    return idx + 1


def _resolve_selected_hidden_layer_indices_from_cfg(cfg: CFG, num_hidden_layers: int) -> Optional[list[int]]:
    """
    Resolve the final raw hidden_states indices passed into AuxHeadModule.

    Priority:
      1) cfg.selected_hidden_layer_indices  (backward-compatible raw hidden_states indices)
      2) cfg.hidden_layer_selection / cfg.hidden_layer_index / cfg.hidden_layer_indices
      3) None -> keep previous behavior
    """
    if cfg.selected_hidden_layer_indices is not None:
        return [
            _normalize_hidden_state_index(i, num_hidden_layers)
            for i in cfg.selected_hidden_layer_indices
        ]

    sel = cfg.hidden_layer_selection
    if sel is None or str(sel).strip().lower() in {"", "none", "default"}:
        return None

    sel = str(sel).strip().lower()

    if sel == "first":
        return [_normalize_transformer_layer_index(0, num_hidden_layers)]

    if sel == "middle":
        middle_idx = max(0, (int(num_hidden_layers) - 1) // 2)
        return [_normalize_transformer_layer_index(middle_idx, num_hidden_layers)]

    if sel == "last":
        return [_normalize_transformer_layer_index(-1, num_hidden_layers)]

    if sel == "index":
        if cfg.hidden_layer_index is None:
            raise ValueError("hidden_layer_selection='index' requires hidden_layer_index to be set.")
        return [_normalize_transformer_layer_index(cfg.hidden_layer_index, num_hidden_layers)]

    if sel == "indices":
        if not cfg.hidden_layer_indices:
            raise ValueError("hidden_layer_selection='indices' requires hidden_layer_indices to be set.")
        return [
            _normalize_transformer_layer_index(i, num_hidden_layers)
            for i in cfg.hidden_layer_indices
        ]

    if sel == "all":
        return list(range(1, int(num_hidden_layers) + 1))

    raise ValueError(
        f"Unsupported hidden_layer_selection={cfg.hidden_layer_selection!r}. "
        f"Use one of: first, middle, last, index, indices, all, or leave it null."
    )


def _resolve_class_weights(
    labels: torch.Tensor,
    failure_threshold: float,
    class_weight_strategy: str,
    failure_weight: float,
    success_weight: float,
    failure_weight_multiplier: float,
    success_weight_multiplier: float,
    min_class_weight: float,
    max_class_weight: float,
) -> tuple[float, float, dict]:
    labels = labels.detach().to(torch.float32).clamp(0.0, 1.0)

    num_failures = int((labels < float(failure_threshold)).sum().item())
    num_successes = int((labels >= float(failure_threshold)).sum().item())

    if class_weight_strategy == "manual":
        base_fw = float(failure_weight)
        base_sw = float(success_weight)
    elif class_weight_strategy == "auto":
        # Keep the majority class at 1.0 and upweight the minority class by ratio.
        if num_failures == 0 or num_successes == 0:
            base_fw = 1.0
            base_sw = 1.0
        elif num_failures < num_successes:
            base_fw = float(num_successes) / float(num_failures)
            base_sw = 1.0
        elif num_successes < num_failures:
            base_fw = 1.0
            base_sw = float(num_failures) / float(num_successes)
        else:
            base_fw = 1.0
            base_sw = 1.0
    else:
        raise ValueError(f"Unsupported class_weight_strategy: {class_weight_strategy}")

    fw = base_fw * float(failure_weight_multiplier)
    sw = base_sw * float(success_weight_multiplier)

    fw = min(max(float(fw), float(min_class_weight)), float(max_class_weight))
    sw = min(max(float(sw), float(min_class_weight)), float(max_class_weight))

    info = {
        "num_failures": num_failures,
        "num_successes": num_successes,
        "base_failure_weight": base_fw,
        "base_success_weight": base_sw,
        "failure_weight_multiplier": float(failure_weight_multiplier),
        "success_weight_multiplier": float(success_weight_multiplier),
        "resolved_failure_weight": fw,
        "resolved_success_weight": sw,
    }
    return fw, sw, info


def _build_sample_weights(
    labels: torch.Tensor,
    failure_threshold: float,
    failure_weight: float,
    success_weight: float,
    severity_power: float,
) -> torch.Tensor:
    labels = labels.detach().to(torch.float32).clamp(0.0, 1.0)
    is_failure = labels < float(failure_threshold)

    class_w = torch.where(
        is_failure,
        torch.full_like(labels, float(failure_weight)),
        torch.full_like(labels, float(success_weight)),
    )

    severity_w = 1.0 + torch.pow(1.0 - labels, float(severity_power))
    return (class_w * severity_w).to(torch.float64)


def _weighted_regression_loss(
    pred_scores: torch.Tensor,
    labels: torch.Tensor,
    loss_name: str,
    failure_threshold: float,
    failure_weight: float,
    success_weight: float,
    severity_power: float,
) -> torch.Tensor:
    labels = labels.to(torch.float32).clamp(0.0, 1.0)
    is_failure = labels < float(failure_threshold)

    class_w = torch.where(
        is_failure,
        torch.full_like(labels, float(failure_weight)),
        torch.full_like(labels, float(success_weight)),
    )
    severity_w = 1.0 + torch.pow(1.0 - labels, float(severity_power))
    weights = class_w * severity_w

    if loss_name == "mse":
        per_item = torch.square(pred_scores - labels)
    elif loss_name == "smooth_l1":
        per_item = F.smooth_l1_loss(pred_scores, labels, reduction="none")
    elif loss_name == "bce":
        pred_scores = pred_scores.clamp(1e-6, 1.0 - 1e-6)
        per_item = F.binary_cross_entropy(pred_scores, labels, reduction="none")
    else:
        raise ValueError(f"Unsupported regression_loss: {loss_name}")

    return (weights * per_item).sum() / weights.sum().clamp_min(1e-8)


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b > 0 else 0.0


def _failure_metrics_from_confusion(conf: torch.Tensor) -> dict:
    tn = int(conf[0, 0].item())
    fp = int(conf[0, 1].item())
    fn = int(conf[1, 0].item())
    tp = int(conf[1, 1].item())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    specificity = _safe_div(tn, tn + fp)
    balanced_acc = 0.5 * (recall + specificity)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "failure_precision": precision,
        "failure_recall": recall,
        "failure_f1": f1,
        "specificity": specificity,
        "balanced_acc": balanced_acc,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_cfg(ap.parse_args().config)

    if int(cfg.num_labels) != 1:
        raise ValueError("For regression training, set num_labels=1.")

    device = get_device()
    fp_dtype = dtype_from_str(cfg.dtype)
    set_seed(int(cfg.seed))

    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "config_used.json", asdict(cfg))

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
    resolved_thinking_enabled = _resolve_thinking_enabled(
        requested_model_id, resolved_model_family, cfg.thinking_mode
    )

    model_id = _resolve_model_id_for_load(
        requested_model_id=requested_model_id,
        resolved_model_family=resolved_model_family,
        prefer_unsloth_mirror=cfg.prefer_unsloth_mirror,
    )
    actual_attn_implementation = _resolve_attn_implementation(
        cfg.attn_implementation,
        resolved_model_family,
    )

    model, processor = FastVisionModel.from_pretrained(
        model_id,
        max_seq_length=cfg.max_seq_len,
        load_in_4bit=cfg.load_in_4bit,
        load_in_8bit=cfg.load_in_8bit,
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        trust_remote_code=cfg.trust_remote_code,
        attn_implementation=actual_attn_implementation,
    )
    model = model.to(device)
    FastVisionModel.for_inference(model)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    _safe_set_max_pixels(processor, cfg.max_pixels)

    processor, runtime_prompting = _patch_processor_for_runtime_prompting(
        processor=processor,
        model_family=resolved_model_family,
        thinking_enabled=resolved_thinking_enabled,
    )
    save_json(
        outdir / "runtime_model_prompting.json",
        {
            "requested_model_id": requested_model_id,
            "resolved_model_id_for_load": model_id,
            "resolved_model_family": resolved_model_family,
            "requested_thinking_mode": cfg.thinking_mode,
            "resolved_thinking_enabled": resolved_thinking_enabled,
            "requested_attn_implementation": cfg.attn_implementation,
            "actual_attn_implementation": actual_attn_implementation,
            "patched_targets": runtime_prompting["patched_targets"],
        },
    )

    ds = load_dataset_auto(cfg.dataset_path)
    train_ds = ds["train"] if isinstance(ds, datasets.DatasetDict) else ds
    train_ds = filter_dataset_by_subset_name(train_ds, cfg.subset_name).shuffle(seed=int(cfg.seed))

    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty after filtering.")

    label_values = torch.tensor(
        [float(x) for x in train_ds[cfg.aux_label_column]],
        dtype=torch.float32,
    ).clamp(0.0, 1.0)

    resolved_failure_weight, resolved_success_weight, class_weight_info = _resolve_class_weights(
        labels=label_values,
        failure_threshold=cfg.failure_threshold,
        class_weight_strategy=cfg.class_weight_strategy,
        failure_weight=cfg.failure_weight,
        success_weight=cfg.success_weight,
        failure_weight_multiplier=cfg.failure_weight_multiplier,
        success_weight_multiplier=cfg.success_weight_multiplier,
        min_class_weight=cfg.min_class_weight,
        max_class_weight=cfg.max_class_weight,
    )

    num_failures = class_weight_info["num_failures"]
    num_successes = class_weight_info["num_successes"]

    print(
        f"dataset rows={len(train_ds)} "
        f"failures(<{cfg.failure_threshold})={num_failures} "
        f"successes(>={cfg.failure_threshold})={num_successes}"
    )
    print(
        f"class_weight_strategy={cfg.class_weight_strategy} "
        f"base_failure_weight={class_weight_info['base_failure_weight']:.4f} "
        f"base_success_weight={class_weight_info['base_success_weight']:.4f} "
        f"failure_weight_multiplier={cfg.failure_weight_multiplier:.4f} "
        f"success_weight_multiplier={cfg.success_weight_multiplier:.4f} "
        f"resolved_failure_weight={resolved_failure_weight:.4f} "
        f"resolved_success_weight={resolved_success_weight:.4f} "
        f"min_class_weight={cfg.min_class_weight} "
        f"max_class_weight={cfg.max_class_weight}"
    )

    hidden_size, num_hidden_layers = infer_hidden_size_and_num_hidden_layers(model)
    resolved_selected_hidden_layer_indices = _resolve_selected_hidden_layer_indices_from_cfg(
        cfg=cfg,
        num_hidden_layers=num_hidden_layers,
    )
    head = AuxHeadModule(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        hidden_encoder_type=cfg.hidden_encoder_type,
        num_labels=cfg.num_labels,
        selected_hidden_layer_indices=resolved_selected_hidden_layer_indices,
    ).to(device)
    if fp_dtype is not None:
        head = head.to(dtype=fp_dtype)

    total_backbone, train_backbone = count_params(model)
    total_head, train_head = count_params(head)
    print(f"model={model_id}")
    print(
        f"requested_model_id={requested_model_id} "
        f"resolved_model_family={resolved_model_family} "
        f"thinking_mode={cfg.thinking_mode} "
        f"resolved_thinking_enabled={resolved_thinking_enabled} "
        f"attn_implementation={actual_attn_implementation}"
    )
    print(f"head_input_mode={cfg.head_input_mode} hidden_encoder_type={cfg.hidden_encoder_type}")
    print("training prompt reconstruction uses system_prompt + user prompt from the dataset; head_input_mode only controls which assistant-side tokens feed the aux head mask.")
    print(f"target_column={cfg.aux_label_column} regression_loss={cfg.regression_loss}")
    print(
        f"failure_threshold={cfg.failure_threshold} "
        f"class_weight_strategy={cfg.class_weight_strategy} "
        f"base_failure_weight={class_weight_info['base_failure_weight']:.4f} "
        f"base_success_weight={class_weight_info['base_success_weight']:.4f} "
        f"failure_weight_multiplier={cfg.failure_weight_multiplier:.4f} "
        f"success_weight_multiplier={cfg.success_weight_multiplier:.4f} "
        f"failure_weight={resolved_failure_weight:.4f} "
        f"success_weight={resolved_success_weight:.4f} "
        f"severity_power={cfg.severity_power} "
        f"use_weighted_sampler={cfg.use_weighted_sampler}"
    )
    print(f"backbone params: total={total_backbone:,} trainable={train_backbone:,}")
    print(f"head params: total={total_head:,} trainable={train_head:,}")
    if head.selected_hidden_layer_indices is not None:
        print(
            "selected_hidden_layer_indices="
            f"{head.selected_hidden_layer_indices} "
            "(raw hidden_states indices; index 0 is typically embeddings)"
        )
        resolved_transformer_layers = [i - 1 for i in head.selected_hidden_layer_indices if i > 0]
        if len(resolved_transformer_layers) > 0:
            print(f"selected_transformer_layer_indices={resolved_transformer_layers}")
    else:
        print("selected_hidden_layer_indices=None (using AuxHeadModule default behavior)")

    sampler = None
    shuffle = True
    if cfg.use_weighted_sampler:
        sample_weights = _build_sample_weights(
            labels=label_values,
            failure_threshold=cfg.failure_threshold,
            failure_weight=resolved_failure_weight,
            success_weight=resolved_success_weight,
            severity_power=cfg.severity_power,
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    loader = DataLoader(
        train_ds,
        batch_size=cfg.per_device_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=VlmCollator(processor, cfg.aux_label_column, cfg.max_seq_len, cfg.head_input_mode),
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
    need_all_hidden_states = head.requires_all_hidden_states

    loss_sum = 0.0
    loss_n = 0
    abs_err_sum = 0.0
    sq_err_sum = 0.0
    reg_n = 0
    conf_sum = torch.zeros((2, 2), dtype=torch.long)
    fail_conf_sum = torch.zeros((2, 2), dtype=torch.long)
    opt_step = 0

    for _ in range(cfg.num_epochs):
        accum = 0
        for batch in loader:
            accum += 1
            batch = move_batch_to_device(batch, device, fp_dtype)

            labels = batch.pop("aux_labels").to(dtype=torch.float32)
            labels = labels.clamp(0.0, 1.0)

            token_mask = batch.pop("head_token_mask")
            forward_inputs: Dict[str, Any] = {}

            # always keep text keys
            for k in ("input_ids", "attention_mask", "position_ids", "cache_position"):
                _maybe_add_forward_key(forward_inputs, batch, k)

            if resolved_model_family == "gemma4":
                # Gemma 4 vision path
                for k in (
                    "pixel_values",
                    "image_position_ids",
                    "pixel_attention_mask",
                    "image_attention_mask",
                    "image_sizes",
                ):
                    _maybe_add_forward_key(forward_inputs, batch, k)
            else:
                # Qwen / Qwen-VL / Qwen3.5 multimodal path
                for k in (
                    "pixel_values",
                    "image_grid_thw",
                    "pixel_values_videos",
                    "video_grid_thw",
                    "mm_token_type_ids",
                ):
                    _maybe_add_forward_key(forward_inputs, batch, k)

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
                    out = (
                        backbone(
                            **forward_inputs,
                            use_cache=False,
                            return_dict=True,
                            output_hidden_states=need_all_hidden_states,
                        )
                        if backbone is not None else
                        model(
                            **forward_inputs,
                            use_cache=False,
                            return_dict=True,
                            output_hidden_states=need_all_hidden_states,
                        )
                    )
                last_hidden = getattr(out, "last_hidden_state", None)
                if last_hidden is None:
                    last_hidden = out.hidden_states[-1]
                hidden_states = out.hidden_states if need_all_hidden_states else None

            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
                logits = head(last_hidden=last_hidden, hidden_states=hidden_states, token_mask=token_mask)
                pred_scores = _to_regression_scores(logits)
                loss = _weighted_regression_loss(
                    pred_scores=pred_scores,
                    labels=labels,
                    loss_name=cfg.regression_loss,
                    failure_threshold=cfg.failure_threshold,
                    failure_weight=resolved_failure_weight,
                    success_weight=resolved_success_weight,
                    severity_power=cfg.severity_power,
                )

            (loss / cfg.grad_accum_steps).backward()
            loss_sum += float(loss.detach().cpu().item())
            loss_n += 1

            pred_cpu = pred_scores.detach().cpu().to(torch.float32)
            labels_cpu = labels.detach().cpu().to(torch.float32)
            abs_err_sum += torch.abs(pred_cpu - labels_cpu).sum().item()
            sq_err_sum += torch.square(pred_cpu - labels_cpu).sum().item()
            reg_n += int(pred_cpu.numel())

            pred_bin = (pred_cpu >= float(cfg.threshold_for_logging)).to(torch.long)
            label_bin = (labels_cpu >= float(cfg.threshold_for_logging)).to(torch.long)
            for y, yhat in zip(label_bin.tolist(), pred_bin.tolist()):
                conf_sum[int(y), int(yhat)] += 1

            label_fail = (labels_cpu < float(cfg.failure_threshold)).to(torch.long)
            pred_fail = (pred_cpu < float(cfg.failure_threshold)).to(torch.long)
            for y, yhat in zip(label_fail.tolist(), pred_fail.tolist()):
                fail_conf_sum[int(y), int(yhat)] += 1

            if accum < cfg.grad_accum_steps:
                continue

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
            pbar.set_postfix(lr=f"{optimizer.param_groups[0]['lr']:.2e}")

            if opt_step % cfg.log_every == 0:
                avg_loss = loss_sum / max(1, loss_n)
                mae = abs_err_sum / max(1, reg_n)
                rmse = math.sqrt(sq_err_sum / max(1, reg_n))

                metrics = compute_metrics_from_confusion(conf_sum)
                fail_metrics = _failure_metrics_from_confusion(fail_conf_sum)

                print(
                    f"[step {opt_step:6d}/{total_steps}] "
                    f"loss={avg_loss:.4f} "
                    f"mae={mae:.4f} rmse={rmse:.4f} "
                    f"fail_recall={fail_metrics['failure_recall']:.3f} "
                    f"fail_precision={fail_metrics['failure_precision']:.3f} "
                    f"fail_f1={fail_metrics['failure_f1']:.3f} "
                    f"bal_acc={fail_metrics['balanced_acc']:.3f} "
                    f"thr_acc={metrics['acc']:.3f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

                if wb is not None:
                    import wandb
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/mae": mae,
                        "train/rmse": rmse,
                        "train/failure_recall": fail_metrics["failure_recall"],
                        "train/failure_precision": fail_metrics["failure_precision"],
                        "train/failure_f1": fail_metrics["failure_f1"],
                        "train/balanced_acc": fail_metrics["balanced_acc"],
                        "train/thr_acc": metrics["acc"],
                        "train/thr_macro_precision": metrics["macro_precision"],
                        "train/thr_macro_recall": metrics["macro_recall"],
                        "train/thr_macro_f1": metrics["macro_f1"],
                        "train/lr": optimizer.param_groups[0]["lr"],
                    }, step=opt_step)

                loss_sum = 0.0
                loss_n = 0
                abs_err_sum = 0.0
                sq_err_sum = 0.0
                reg_n = 0
                conf_sum.zero_()
                fail_conf_sum.zero_()

            if opt_step % cfg.save_every == 0:
                ckpt = save_checkpoint(head, cfg, outdir, opt_step, f"aux_head_step{opt_step}.pt")
                print(f"saved: {ckpt}")

    pbar.close()
    final_ckpt = save_checkpoint(head, cfg, outdir, opt_step, "aux_head_final.pt")
    processor.save_pretrained(str(outdir))
    print(f"saved: {final_ckpt}")

    if wb is not None:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()