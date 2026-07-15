#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""OpenAI-compatible proxy verifier server for Qwen3.5-9B instruct + auxiliary head.

This server preserves the original generation request shape by forwarding the
incoming OpenAI /v1/chat/completions JSON to a plain small vLLM OpenAI server,
then runs the aux head on the original messages plus the generated answer.

This updated variant preserves the original multimodal aux-scoring path,
including image collection and forwarding of vision tensors to the aux backbone.

Flow:
1) receive request
2) forward request JSON to GENERATOR_BASE_URL/chat/completions
3) read generated answer + usage
4) run aux head on (messages, answer)
5) return normal OpenAI-style output + top-level gnosis_score

Recommended architecture:
- small plain vLLM server      -> port 8000
- verifier proxy server        -> port 8001
- large plain vLLM server      -> port 8002
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import requests
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

_THIS_DIR = Path(__file__).resolve().parent
for _p in (_THIS_DIR, _THIS_DIR.parent, Path.cwd()):
    ps = str(_p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir))) if len(Path(current_dir).parts) >= 3 else current_dir
for p in [current_dir, base_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from aux_head_shared_utils import (  # type: ignore
    AuxHeadModule,
    ChatBatchBuilder,
    dtype_from_str,
    get_device,
    infer_hidden_size_and_num_hidden_layers,
    move_batch_to_device,
)
from unsloth import FastVisionModel


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
GENERATOR_BASE_URL = os.getenv("GENERATOR_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
GENERATOR_API_KEY = os.getenv("GENERATOR_API_KEY", "EMPTY")
# Optional: override the upstream model field before proxying.
GENERATOR_MODEL_NAME = os.getenv("GENERATOR_MODEL_NAME", "Qwen/Qwen3.5-9B").strip()
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "Qwen3.5-9B-Instruct-AuxHeadServer")

AUX_MODEL_NAME_OR_PATH = os.getenv("AUX_MODEL_NAME_OR_PATH", "Qwen/Qwen3.5-9B")
AUX_MODEL_FAMILY = os.getenv("AUX_MODEL_FAMILY", "qwen3_5").strip().lower()
AUX_THINKING_MODE = os.getenv("AUX_THINKING_MODE", "off").strip().lower()
AUX_HEAD_CKPT = os.getenv(
    "AUX_HEAD_CKPT",
    "trained_models/"
    "Qwen3_5_9B_think_off_hard_Mixed_Sources_120k_auxhead/aux_head_final.pt",
)
AUX_DTYPE = os.getenv("AUX_DTYPE", "bf16")
AUX_MAX_SEQ_LEN = int(os.getenv("AUX_MAX_SEQ_LEN", "32000"))
AUX_MAX_PIXELS = int(os.getenv("AUX_MAX_PIXELS", "200000"))
AUX_ATTN_IMPLEMENTATION = os.getenv("AUX_ATTN_IMPLEMENTATION", "flash_attention_3")
AUX_LOAD_IN_4BIT = os.getenv("AUX_LOAD_IN_4BIT", "false").lower() == "true"
AUX_LOAD_IN_8BIT = os.getenv("AUX_LOAD_IN_8BIT", "false").lower() == "true"
AUX_USE_GRADIENT_CHECKPOINTING = os.getenv("AUX_USE_GRADIENT_CHECKPOINTING", "unsloth")
AUX_PREFER_UNSLOTH_MIRROR = os.getenv("AUX_PREFER_UNSLOTH_MIRROR", "true").lower() == "true"

DEFAULT_AUX_HEAD_CFG = {
    "num_labels": 1,
    "head_input_mode": "completion_text_only",
    "hidden_encoder_type": "lite",
    "selected_hidden_layer_indices": None,
}

EMBED_GNOSIS_SCORE_IN_CONTENT = os.getenv("EMBED_GNOSIS_SCORE_IN_CONTENT", "false").lower() == "true"
GNOSIS_TAG_TEMPLATE = "gnosis_score={:.6f}"

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))

app = FastAPI()


# -----------------------------------------------------------------------------
# Runtime prompt helpers
# -----------------------------------------------------------------------------
def infer_model_family_for_runtime(model_id: str, requested_family: str = "auto") -> str:
    requested_family = str(requested_family or "auto").strip().lower()
    mid = str(model_id or "").strip().lower()

    # Let obvious checkpoint identity win. This prevents mismatched family tags
    # from disabling model-specific runtime behavior.
    if "qwen3.5" in mid:
        return "qwen3_5"
    if "qwen3-vl" in mid:
        return "qwen3_vl"
    if "gemma-4" in mid:
        return "gemma4"
    if "qwen3" in mid:
        return "qwen3"

    if requested_family != "auto":
        return requested_family
    return "other"


def resolve_thinking_enabled_for_runtime(model_id: str, model_family: str, thinking_mode: Any) -> bool:
    if isinstance(thinking_mode, bool):
        return thinking_mode

    mode = str(thinking_mode or "auto").strip().lower()
    if mode in {"on", "true", "1", "yes"}:
        return True
    if mode in {"off", "false", "0", "no"}:
        return False

    mid = str(model_id or "").strip().lower()
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


def _patch_gemma_messages(messages: List[Dict[str, Any]], thinking_enabled: bool) -> List[Dict[str, Any]]:
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


def patch_processor_for_runtime_prompting(processor, model_family: str, thinking_enabled: bool):
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


def try_set_max_pixels(processor_like, max_pixels: int) -> None:
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


def resolve_attn_implementation_for_runtime(attn_implementation: str, model_family: str) -> str:
    attn = str(attn_implementation or "").strip()
    if model_family == "gemma4" and attn == "flash_attention_3":
        return "sdpa"
    return attn


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
class TokenEstimator:
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._failed: set[str] = set()

    def _get_tokenizer(self, model_name: str):
        if model_name in self._cache:
            return self._cache[model_name]
        if model_name in self._failed:
            return None
        tok = None
        try:
            proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            tok = getattr(proc, "tokenizer", None)
        except Exception:
            tok = None
        if tok is None:
            try:
                tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            except Exception:
                tok = None
        if tok is None:
            self._failed.add(model_name)
            return None
        self._cache[model_name] = tok
        return tok

    def count(self, model_name: str, text: str) -> Tuple[int, str]:
        tok = self._get_tokenizer(model_name)
        if tok is not None:
            try:
                ids = tok(text or "", add_special_tokens=False)["input_ids"]
                return int(len(ids)), "estimated_text_tokenizer"
            except Exception:
                pass
        approx = 0 if not text else max(1, int(round(len(text) / 4.0)))
        return approx, "estimated_chars_div4"


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t in ("text", "input_text"):
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


def _load_image_from_data_url(url: str) -> Image.Image:
    _, b64 = url.split(",", 1)
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _load_image_from_path_like(s: str) -> Image.Image:
    if s.startswith("file://"):
        path = urlparse(s).path
        return Image.open(path).convert("RGB")
    if s.startswith("/") or Path(s).exists():
        return Image.open(s).convert("RGB")
    raise ValueError(f"Unsupported local image path: {s}")


def _load_image_from_url_field(url: str) -> Image.Image:
    if not isinstance(url, str) or not url:
        raise ValueError("Empty image url field.")
    if url.startswith("data:image"):
        return _load_image_from_data_url(url)
    if url.startswith("file://") or url.startswith("/"):
        return _load_image_from_path_like(url)
    if re.match(r"^[A-Za-z]:\\", url):
        return Image.open(url).convert("RGB")
    if Path(url).exists():
        return Image.open(url).convert("RGB")
    raise ValueError(f"Unsupported non-local image url: {url}")


def _normalize_messages_and_collect_images(messages: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
    normalized: List[Dict[str, Any]] = []
    all_images: List[Image.Image] = []

    for msg in messages or []:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if isinstance(content, str):
            normalized.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            normalized.append({"role": role, "content": _extract_text_from_content(content)})
            continue

        norm_content: List[Dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t in ("text", "input_text"):
                norm_content.append({"type": "text", "text": str(item.get("text", ""))})
            elif t in ("image", "input_image"):
                img_val = item.get("image") or item.get("path") or item.get("url")
                if isinstance(img_val, str):
                    all_images.append(_load_image_from_url_field(img_val))
                    norm_content.append({"type": "image"})
            elif t == "image_url":
                image_url = item.get("image_url", {})
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                all_images.append(_load_image_from_url_field(str(url)))
                norm_content.append({"type": "image"})
        normalized.append({"role": role, "content": norm_content})
    return normalized, all_images


def _extract_answer_text_from_chat_response(resp_json: Dict[str, Any]) -> str:
    try:
        choices = resp_json.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return _extract_text_from_content(content)
    except Exception:
        pass
    return ""


def _stream_one_chunk(model_name: str, content: str, score: float, usage: Dict[str, Any]) -> StreamingResponse:
    created = int(time.time())
    resp_id = f"chatcmpl-{int(time.time() * 1000)}"
    system_fp = GNOSIS_TAG_TEMPLATE.format(score)

    def gen():
        first = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "system_fingerprint": system_fp,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(first, ensure_ascii=False)}\\n\\n"

        last = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "system_fingerprint": system_fp,
            "usage": usage,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }
        yield f"data: {json.dumps(last, ensure_ascii=False)}\\n\\n"
        yield "data: [DONE]\\n\\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def _usage_from_upstream_or_estimate(upstream_json: Dict[str, Any], prompt_text: str, answer: str, model_name: str, estimator: TokenEstimator) -> Dict[str, Any]:
    usage = upstream_json.get("usage") if isinstance(upstream_json.get("usage"), dict) else None
    if usage is not None:
        return usage
    prompt_tokens, s1 = estimator.count(model_name, prompt_text)
    completion_tokens, s2 = estimator.count(model_name, answer)
    source = s1 if s1 == s2 else f"{s1}+{s2}"
    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens + completion_tokens),
        "usage_source": source,
    }


# -----------------------------------------------------------------------------
# Aux-head scorer
# -----------------------------------------------------------------------------
def _resolve_aux_head_cfg(ckpt_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "num_labels": int(ckpt_cfg.get("num_labels", DEFAULT_AUX_HEAD_CFG["num_labels"])),
        "head_input_mode": ckpt_cfg.get("head_input_mode", DEFAULT_AUX_HEAD_CFG["head_input_mode"]),
        "hidden_encoder_type": ckpt_cfg.get("hidden_encoder_type", DEFAULT_AUX_HEAD_CFG["hidden_encoder_type"]),
        "selected_hidden_layer_indices": ckpt_cfg.get(
            "selected_hidden_layer_indices",
            DEFAULT_AUX_HEAD_CFG["selected_hidden_layer_indices"],
        ),
    }


@dataclass
class ScorerRuntime:
    model: Any
    processor: Any
    head: Any
    device: Any
    fp_dtype: Any
    resolved_model_name: str
    resolved_model_family: str
    resolved_thinking_enabled: bool
    runtime_prompting_info: Dict[str, Any]
    ckpt_cfg: Dict[str, Any]
    aux_head_cfg: Dict[str, Any]


class MessageAuxHeadScorer:
    def __init__(self):
        if not AUX_HEAD_CKPT.strip():
            raise ValueError("AUX_HEAD_CKPT must be set for the verifier server.")

        self.device = get_device()
        self.fp_dtype = dtype_from_str(AUX_DTYPE)
        model_name = AUX_MODEL_NAME_OR_PATH.strip()
        if AUX_PREFER_UNSLOTH_MIRROR and model_name.startswith("Qwen/"):
            model_name = "unsloth/" + model_name.split("/", 1)[1]

        resolved_model_family = infer_model_family_for_runtime(model_name, AUX_MODEL_FAMILY)
        resolved_thinking_enabled = resolve_thinking_enabled_for_runtime(
            model_name, resolved_model_family, AUX_THINKING_MODE
        )
        actual_attn_implementation = resolve_attn_implementation_for_runtime(
            AUX_ATTN_IMPLEMENTATION, resolved_model_family
        )

        model, processor = FastVisionModel.from_pretrained(
            model_name,
            max_seq_length=AUX_MAX_SEQ_LEN,
            load_in_4bit=AUX_LOAD_IN_4BIT,
            load_in_8bit=AUX_LOAD_IN_8BIT,
            use_gradient_checkpointing=AUX_USE_GRADIENT_CHECKPOINTING,
            trust_remote_code=True,
            attn_implementation=actual_attn_implementation,
        )
        model = model.to(self.device)
        FastVisionModel.for_inference(model)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        processor, runtime_prompting_info = patch_processor_for_runtime_prompting(
            processor, resolved_model_family, resolved_thinking_enabled
        )
        try_set_max_pixels(processor, AUX_MAX_PIXELS)

        ckpt = torch.load(AUX_HEAD_CKPT, map_location="cpu")
        ckpt_cfg = ckpt.get("cfg", {}) or {}
        aux_head_cfg = _resolve_aux_head_cfg(ckpt_cfg)
        hidden_size, num_hidden_layers = infer_hidden_size_and_num_hidden_layers(model)
        head = AuxHeadModule(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            hidden_encoder_type=aux_head_cfg["hidden_encoder_type"],
            num_labels=aux_head_cfg["num_labels"],
            selected_hidden_layer_indices=aux_head_cfg["selected_hidden_layer_indices"],
        ).to(self.device)
        head.load_state_dict(ckpt["head_state"])
        head.eval()

        self.runtime = ScorerRuntime(
            model=model,
            processor=processor,
            head=head,
            device=self.device,
            fp_dtype=self.fp_dtype,
            resolved_model_name=model_name,
            resolved_model_family=resolved_model_family,
            resolved_thinking_enabled=resolved_thinking_enabled,
            runtime_prompting_info=runtime_prompting_info,
            ckpt_cfg=ckpt_cfg,
            aux_head_cfg=aux_head_cfg,
        )

    def score(self, messages: Sequence[Dict[str, Any]], response_text: str) -> Dict[str, Any]:
        normalized_messages, pil_images = _normalize_messages_and_collect_images(messages)
        normalized_messages = list(normalized_messages) + [{"role": "assistant", "content": str(response_text)}]

        if not pil_images:
            sample_images: Any = None
        elif len(pil_images) == 1:
            sample_images = pil_images[0]
        else:
            sample_images = pil_images

        batch_builder = ChatBatchBuilder(
            processor=self.runtime.processor,
            max_seq_len=AUX_MAX_SEQ_LEN,
            head_input_mode=self.runtime.aux_head_cfg["head_input_mode"],
        )
        batch = batch_builder.build_from_messages([normalized_messages], [sample_images])
        batch = move_batch_to_device(batch, self.runtime.device, self.runtime.fp_dtype)

        backbone = getattr(self.runtime.model, "model", None)
        need_all_hidden_states = bool(getattr(self.runtime.head, "requires_all_hidden_states", False))
        forward_inputs = {
            k: batch[k]
            for k in (
                "input_ids",
                "attention_mask",
                "pixel_values",
                "image_grid_thw",
                "pixel_values_videos",
                "video_grid_thw",
                "mm_token_type_ids",
                "position_ids",
                "cache_position",
            )
            if k in batch
        }

        with torch.inference_mode():
            out = (
                backbone(
                    **forward_inputs,
                    use_cache=False,
                    return_dict=True,
                    output_hidden_states=need_all_hidden_states,
                )
                if backbone is not None
                else self.runtime.model(
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
            logits = self.runtime.head(
                last_hidden=last_hidden,
                hidden_states=hidden_states,
                token_mask=batch["head_token_mask"],
            ).float()

            if int(self.runtime.aux_head_cfg["num_labels"]) == 1:
                score_t = torch.sigmoid(logits.view(-1))
                pred_t = (score_t >= 0.5).long()
                score = float(score_t[0].item())
                pred = int(pred_t[0].item())
                probs = [1.0 - score, score]
            else:
                probs_t = torch.softmax(logits, dim=-1)
                pred = int(probs_t.argmax(dim=-1)[0].item())
                score = float(probs_t[0, 1].item())
                probs = [float(x) for x in probs_t[0].detach().cpu().tolist()]

        return {
            "head_pred": pred,
            "head_prob_correct": score,
            "head_probs": probs,
            "num_labels": int(self.runtime.aux_head_cfg["num_labels"]),
            "resolved_aux_head_cfg": self.runtime.aux_head_cfg,
            "resolved_model_family": self.runtime.resolved_model_family,
            "resolved_thinking_enabled": self.runtime.resolved_thinking_enabled,
            "runtime_prompting_info": self.runtime.runtime_prompting_info,
        }


# -----------------------------------------------------------------------------
# Proxy backend
# -----------------------------------------------------------------------------
class ProxyOpenAIBackend:
    def __init__(self):
        self.base_url = GENERATOR_BASE_URL
        self.api_key = GENERATOR_API_KEY
        self.token_estimator = TokenEstimator()

    def generate(self, req: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        payload = dict(req)
        payload["stream"] = False
        if GENERATOR_MODEL_NAME:
            payload["model"] = GENERATOR_MODEL_NAME

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url}/chat/completions"
        resp = requests.post(url, headers=headers, json=payload, timeout=600)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Upstream generator failed: status={resp.status_code} url={url} body={resp.text[:2000]}"
            )
        j = resp.json()
        answer = _extract_answer_text_from_chat_response(j)
        prompt_text = json.dumps(payload.get("messages", []), ensure_ascii=False)
        usage = _usage_from_upstream_or_estimate(
            upstream_json=j,
            prompt_text=prompt_text,
            answer=answer,
            model_name=GENERATOR_MODEL_NAME or payload.get("model") or AUX_MODEL_NAME_OR_PATH,
            estimator=self.token_estimator,
        )
        return answer, j, usage


# -----------------------------------------------------------------------------
# App lifecycle
# -----------------------------------------------------------------------------
GEN_BACKEND: Any = None
AUX_SCORER: Optional[MessageAuxHeadScorer] = None


@app.on_event("startup")
def startup() -> None:
    global GEN_BACKEND, AUX_SCORER
    print("=" * 80)
    print("Proxy verifier server startup")
    print("GENERATOR_BASE_URL:", GENERATOR_BASE_URL)
    print("GENERATOR_MODEL_NAME:", GENERATOR_MODEL_NAME)
    print("SERVED_MODEL_NAME:", SERVED_MODEL_NAME)
    print("AUX_MODEL_NAME_OR_PATH:", AUX_MODEL_NAME_OR_PATH)
    print("AUX_MODEL_FAMILY:", AUX_MODEL_FAMILY)
    print("AUX_THINKING_MODE:", AUX_THINKING_MODE)
    print("AUX_HEAD_CKPT:", AUX_HEAD_CKPT)
    print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
    print("=" * 80)

    AUX_SCORER = MessageAuxHeadScorer()
    print("Resolved aux head cfg:", AUX_SCORER.runtime.aux_head_cfg)
    print("Resolved model family:", AUX_SCORER.runtime.resolved_model_family)
    print("Resolved thinking enabled:", AUX_SCORER.runtime.resolved_thinking_enabled)
    print("Runtime prompting info:", AUX_SCORER.runtime.runtime_prompting_info)
    GEN_BACKEND = ProxyOpenAIBackend()


@app.get("/")
def root() -> Dict[str, Any]:
    aux_head_cfg = AUX_SCORER.runtime.aux_head_cfg if AUX_SCORER is not None else None
    return {
        "status": "ok",
        "served_model_name": SERVED_MODEL_NAME,
        "backend_mode": "proxy_openai_exact",
        "generator_base_url": GENERATOR_BASE_URL,
        "generator_model_name": GENERATOR_MODEL_NAME or None,
        "aux_model_name_or_path": AUX_MODEL_NAME_OR_PATH,
        "aux_model_family": AUX_MODEL_FAMILY,
        "aux_thinking_mode": AUX_THINKING_MODE,
        "resolved_model_family": AUX_SCORER.runtime.resolved_model_family if AUX_SCORER is not None else None,
        "resolved_thinking_enabled": AUX_SCORER.runtime.resolved_thinking_enabled if AUX_SCORER is not None else None,
        "runtime_prompting_info": AUX_SCORER.runtime.runtime_prompting_info if AUX_SCORER is not None else None,
        "resolved_aux_head_cfg": aux_head_cfg,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        req = await request.json()
        model_name = req.get("model") or SERVED_MODEL_NAME
        stream = bool(req.get("stream", False))
        messages = req.get("messages", [])

        answer, raw_gen_response, usage = GEN_BACKEND.generate(req)
        aux = AUX_SCORER.score(messages=messages, response_text=answer)
        score = float(aux["head_prob_correct"])

        content = answer
        if EMBED_GNOSIS_SCORE_IN_CONTENT:
            content = f"<think>{GNOSIS_TAG_TEMPLATE.format(score)}</think>\n{answer}"

        payload = {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "system_fingerprint": GNOSIS_TAG_TEMPLATE.format(score),
            "gnosis_score": score,
            "gnosis_meta": {
                "head_pred": int(aux["head_pred"]),
                "head_probs": [float(x) for x in aux["head_probs"]],
                "num_labels": int(aux["num_labels"]),
                "resolved_aux_head_cfg": aux["resolved_aux_head_cfg"],
                "resolved_model_family": aux["resolved_model_family"],
                "resolved_thinking_enabled": aux["resolved_thinking_enabled"],
                "runtime_prompting_info": aux["runtime_prompting_info"],
                "score_source": "server_aux_head",
                "backend_mode": "proxy_openai_exact",
            },
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": usage,
            "generator_response": raw_gen_response,
        }

        if stream:
            return _stream_one_chunk(model_name, content, score, usage)
        return JSONResponse(payload)
    except Exception:
        err = traceback.format_exc()
        return JSONResponse(status_code=500, content={"detail": f"SERVER ERROR:\n{err}"})


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
