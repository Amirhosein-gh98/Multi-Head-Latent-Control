#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import json
import random
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import datasets
import torch
import torch.nn as nn
from PIL import Image

from feature_extractors import (
    HiddenFeatureExtractorLite,
    CorrectnessHeadLite,
    StrongContentOnlyHiddenEncoder,
    StrongMultiLayerHiddenTrajectoryEncoder,
)

ALLOWED_SUBSET_NAMES = {
    "vqav2",
    "scienceqa",
    "chartqa",
    "docvqa",
    "screenqa",
    "aokvqa",
    "ai2d_merged",
    "infographic_vqa",
    "groundui",
    "aguvis-stage-1",
    "aguvis-stage-2",
    "mm-openr1",
    "dapo",
    "triviaqa",
    "apigen-mt-5k",
}

AGUVIS2_FORMAT_INSTRUCTION = (
    "\n\nReturn exactly one <think>...</think> block followed by one <code>...</code> block and nothing else.\n"
    "- In <think>, write one short sentence describing the next move.\n"
    "- In <code>, write exactly one API call in the dataset style.\n"
    "- Do not use markdown fences. Do not add any extra prose outside the tags."
)

APIGEN_NEXT_TURN_INSTRUCTION = (
    "Continue the conversation by writing the next assistant turn only.\n"
    "The next turn may be either:\n"
    "- a normal assistant reply, or\n"
    "- a single function-call JSON object\n"
    "Choose whichever is appropriate from the conversation state.\n"
    "Do not add role labels. Do not explain what you are doing outside the next turn itself."
)

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(__import__("os").environ.get("LOCAL_RANK", "0"))
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def dtype_from_str(s: str) -> Optional[torch.dtype]:
    s = str(s or "bf16").lower()
    if s == "bf16":
        return torch.bfloat16
    if s == "fp16":
        return torch.float16
    if s == "fp32":
        return torch.float32
    if s == "auto":
        return None
    return torch.bfloat16


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if p.suffix in {".yaml", ".yml"}:
        import yaml
        return yaml.safe_load(p.read_text()) or {}
    if p.suffix == ".json":
        return json.loads(p.read_text())
    return {}


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_dataset_auto(path: str):
    p = Path(path)
    if p.is_dir() and (p / "dataset_info.json").exists():
        return datasets.load_from_disk(str(p))
    if p.is_dir():
        pq = sorted(glob(str(p / "**" / "*.parquet"), recursive=True))
        if pq:
            return datasets.load_dataset("parquet", data_files={"train": pq})
        js = sorted(glob(str(p / "**" / "*.json"), recursive=True) + glob(str(p / "**" / "*.jsonl"), recursive=True))
        if js:
            return datasets.load_dataset("json", data_files={"train": js})
    if p.is_file() and p.suffix == ".parquet":
        return datasets.load_dataset("parquet", data_files={"train": str(p)})
    if p.is_file() and p.suffix in {".json", ".jsonl"}:
        return datasets.load_dataset("json", data_files={"train": str(p)})
    raise FileNotFoundError(f"Could not load dataset from {path}")


def filter_dataset_by_subset_name(ds, subset_name: Optional[str]):
    if not subset_name:
        return ds
    subset_name = str(subset_name).strip().lower()
    if "subset_name" not in ds.column_names:
        return ds
    idx = [i for i, v in enumerate(ds["subset_name"]) if str(v).strip().lower() == subset_name]
    return ds.select(idx) if idx else ds


def infer_hidden_size_and_num_hidden_layers(model: nn.Module) -> Tuple[int, int]:
    text_cfg = getattr(getattr(model, "config", None), "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, "hidden_size") and hasattr(text_cfg, "num_hidden_layers"):
        return int(text_cfg.hidden_size), int(text_cfg.num_hidden_layers)
    cfg = getattr(model, "config", None)
    return int(cfg.hidden_size), int(cfg.num_hidden_layers)


def resolve_transformer_layer_indices(indices: Optional[Sequence[int]], num_hidden_layers: int) -> Optional[List[int]]:
    if not indices:
        return None
    out = []
    for idx in indices:
        idx = int(idx)
        idx = num_hidden_layers + idx if idx < 0 else idx
        if 0 <= idx < num_hidden_layers and idx not in out:
            out.append(idx)
    return sorted(out) or None


def load_image_any(x: Any) -> Optional[Image.Image]:
    if x is None:
        return None
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, str):
        p = Path(x[7:] if x.startswith("file://") else x)
        with Image.open(p) as img:
            return img.convert("RGB")
    if isinstance(x, (bytes, bytearray)):
        with Image.open(io.BytesIO(x)) as img:
            return img.convert("RGB")
    return None


def build_messages_from_prompt_completion(
    prompt: str,
    completion: str,
    has_image: bool,
    system_prompt: str = "",
) -> List[Dict[str, Any]]:
    user_content: Any = [{"type": "image"}] if has_image else []
    if prompt:
        if has_image:
            user_content.append({"type": "text", "text": prompt})
        else:
            user_content = prompt
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend([
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": completion},
    ])
    return messages


def _build_training_messages_for_subset(
    *,
    subset_name: str,
    system_prompt: str,
    prompt: str,
    completion: str,
    has_image: bool,
    tools: str = "",
    context_text: str = "",
    human_prompt: str = "",
) -> List[Dict[str, Any]]:
    subset = str(subset_name or "").strip().lower()

    if subset == "aguvis-stage-2":
        user_prompt = str(human_prompt or prompt or "").rstrip()
        if AGUVIS2_FORMAT_INSTRUCTION not in user_prompt:
            user_prompt += AGUVIS2_FORMAT_INSTRUCTION
        return build_messages_from_prompt_completion(
            prompt=user_prompt,
            completion=completion,
            has_image=has_image,
            system_prompt=system_prompt,
        )

    if subset == "apigen-mt-5k":
        combined_system = (str(system_prompt).strip() + "\n\nAvailable tools:\n" + str(tools).strip()).strip()
        user_prompt = f"Conversation so far:\n{str(context_text or prompt or '')}\n\n{APIGEN_NEXT_TURN_INSTRUCTION}"
        return build_messages_from_prompt_completion(
            prompt=user_prompt,
            completion=completion,
            has_image=False,
            system_prompt=combined_system,
        )

    return build_messages_from_prompt_completion(
        prompt=prompt,
        completion=completion,
        has_image=has_image,
        system_prompt=system_prompt,
    )


def build_messages_and_image(ex: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[Image.Image]]:
    image = None
    if ex.get("images") is not None:
        images = ex["images"]
        image = load_image_any(images[0] if isinstance(images, (list, tuple)) and images else images)
    elif ex.get("image") is not None:
        image = load_image_any(ex["image"])

    if ex.get("messages") is not None:
        return ex["messages"], image

    system_prompt = str(ex.get("system_prompt") or "")
    prompt = str(ex.get("prompt") or ex.get("question") or ex.get("text") or "")
    completion = str(ex.get("completion") or ex.get("answer") or "")
    subset_name = str(ex.get("subset_name") or "")
    tools = str(ex.get("tools") or "")
    context_text = str(ex.get("context_text") or "")
    human_prompt = str(ex.get("human_prompt") or "")

    messages = _build_training_messages_for_subset(
        subset_name=subset_name,
        system_prompt=system_prompt,
        prompt=prompt,
        completion=completion,
        has_image=image is not None,
        tools=tools,
        context_text=context_text,
        human_prompt=human_prompt,
    )
    return messages, image


class ChatBatchBuilder:
    def __init__(self, processor: Any, max_seq_len: int, head_input_mode: str):
        self.processor = processor
        self.max_seq_len = int(max_seq_len)
        self.head_input_mode = str(head_input_mode)
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.vision_token_ids = self._get_token_ids([
            "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>",
            "<|image_start|>", "<|image_end|>", "<image>", "</image>",
        ])
        self.control_token_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        self.control_token_ids.update(self._get_token_ids([
            "<|im_start|>", "<|im_end|>", "<|assistant|>", "<|user|>", "<|system|>",
        ]))

    def _token_id(self, token: str) -> Optional[int]:
        try:
            tid = self.tokenizer.convert_tokens_to_ids(token)
            return int(tid) if isinstance(tid, int) and tid >= 0 else None
        except Exception:
            return None

    def _get_token_ids(self, tokens: Sequence[str]) -> List[int]:
        ids = []
        for t in tokens:
            tid = self._token_id(t)
            if tid is not None and tid not in ids:
                ids.append(tid)
        return ids

    def _encode_prefix_len(self, text: str, image: Optional[Image.Image]) -> int:
        kwargs = {
            "text": [text],
            "return_tensors": "pt",
            "padding": False,
            "truncation": True,
            "max_length": self.max_seq_len,
        }
        if image is not None:
            kwargs["images"] = [image]
        enc = self.processor(**kwargs)
        return int(enc["attention_mask"][0].sum().item())

    def _drop_ids(self, input_ids: torch.Tensor, token_ids: Sequence[int]) -> torch.Tensor:
        drop = torch.zeros_like(input_ids, dtype=torch.bool)
        for tid in token_ids:
            drop |= input_ids.eq(int(tid))
        return (~drop).long()

    # def build_from_messages(
    #     self,
    #     messages_batch: List[List[Dict[str, Any]]],
    #     images: List[Optional[Image.Image]],
    #     labels: Optional[Sequence[int]] = None,
    # ) -> Dict[str, Any]:
    #     texts, prefix_texts = [], []
    #     for messages in messages_batch:
    #         texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    #         if self.head_input_mode in {"completion_only", "completion_text_only"}:
    #             prefix_texts.append(self.processor.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=False))
    #         else:
    #             prefix_texts.append(None)

    #     kwargs = {
    #         "text": texts,
    #         "return_tensors": "pt",
    #         "padding": True,
    #         "truncation": True,
    #         "max_length": self.max_seq_len,
    #     }
    #     if any(img is not None for img in images):
    #         kwargs["images"] = images
    #     batch = dict(self.processor(**kwargs).items())
    #     batch.pop("token_type_ids", None)

    #     input_ids = batch["input_ids"]
    #     attention_mask = batch["attention_mask"].long()

    #     if self.head_input_mode == "all_tokens":
    #         head_mask = attention_mask.clone()
    #     elif self.head_input_mode == "text_only":
    #         head_mask = attention_mask * self._drop_ids(input_ids, self.vision_token_ids)
    #     else:
    #         head_mask = torch.zeros_like(attention_mask)
    #         for i, prefix in enumerate(prefix_texts):
    #             prefix_len = self._encode_prefix_len(prefix, images[i])
    #             full_len = int(attention_mask[i].sum().item())
    #             head_mask[i, prefix_len:full_len] = 1
    #         if self.head_input_mode == "completion_text_only":
    #             drop_ids = sorted(set(self.vision_token_ids).union(self.control_token_ids))
    #             head_mask = head_mask * self._drop_ids(input_ids, drop_ids)

    #     batch["head_token_mask"] = head_mask * attention_mask
    #     if labels is not None:
    #         batch["aux_labels"] = torch.tensor([float(x) for x in labels], dtype=torch.float32)
    #     return batch

    def build_from_messages(
        self,
        messages_batch: List[List[Dict[str, Any]]],
        images: List[Optional[Image.Image]],
        labels: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        texts, prefix_texts = [], []
        for messages in messages_batch:
            texts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            )
            if self.head_input_mode in {"completion_only", "completion_text_only", "completion_first_200"}:
                prefix_texts.append(
                    self.processor.apply_chat_template(
                        messages[:-1], tokenize=False, add_generation_prompt=False
                    )
                )
            else:
                prefix_texts.append(None)

        kwargs = {
            "text": texts,
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": self.max_seq_len,
        }
        if any(img is not None for img in images):
            kwargs["images"] = images

        batch = dict(self.processor(**kwargs).items())
        batch.pop("token_type_ids", None)

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"].long()

        if self.head_input_mode == "all_tokens":
            head_mask = attention_mask.clone()

        elif self.head_input_mode == "text_only":
            head_mask = attention_mask * self._drop_ids(input_ids, self.vision_token_ids)

        else:
            head_mask = torch.zeros_like(attention_mask)
            for i, prefix in enumerate(prefix_texts):
                prefix_len = self._encode_prefix_len(prefix, images[i])
                full_len = int(attention_mask[i].sum().item())

                start = min(prefix_len, full_len)
                if self.head_input_mode in {"completion_only", "completion_text_only"}:
                    end = full_len
                elif self.head_input_mode == "completion_first_200":
                    end = min(full_len, start + 200)
                else:
                    raise ValueError(f"Unsupported head_input_mode: {self.head_input_mode}")

                head_mask[i, start:end] = 1

            if self.head_input_mode == "completion_text_only":
                drop_ids = sorted(set(self.vision_token_ids).union(self.control_token_ids))
                head_mask = head_mask * self._drop_ids(input_ids, drop_ids)

        batch["head_token_mask"] = head_mask * attention_mask
        if labels is not None:
            batch["aux_labels"] = torch.tensor([float(x) for x in labels], dtype=torch.float32)
        return batch


class VlmCollator(ChatBatchBuilder):
    def __init__(self, processor: Any, aux_label_column: str, max_seq_len: int, head_input_mode: str):
        super().__init__(processor, max_seq_len, head_input_mode)
        self.aux_label_column = aux_label_column

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        messages_batch, images, labels = [], [], []
        for ex in examples:
            messages, image = build_messages_and_image(ex)
            messages_batch.append(messages)
            images.append(image)
            labels.append(float(ex[self.aux_label_column]))
        return self.build_from_messages(messages_batch, images, labels)

class AuxHeadModule(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_hidden_layers: int,
        hidden_encoder_type: str = "strong_single",
        num_labels: int = 2,
        selected_hidden_layer_indices: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.hidden_encoder_type = str(hidden_encoder_type)
        self.selected_hidden_layer_indices = resolve_transformer_layer_indices(
            selected_hidden_layer_indices,
            num_hidden_layers,
        )

        if self.hidden_encoder_type == "lite":
            self.hid_extractor = HiddenFeatureExtractorLite(D_model=hidden_size)
        elif self.hidden_encoder_type == "strong_single":
            self.hid_extractor = StrongContentOnlyHiddenEncoder(D_model=hidden_size)
        else:
            self.hid_extractor = StrongMultiLayerHiddenTrajectoryEncoder(
                D_model=hidden_size,
                max_model_layers=num_hidden_layers,
            )

        self.requires_all_hidden_states = self.hidden_encoder_type == "strong_multi"
        
        self.head = CorrectnessHeadLite(
            D_ATT=256,
            D_CONF=128,
            D_HID=256,
            use_attn=False,
            use_conf=False,
            use_hid=True,
            num_labels=int(num_labels),
        )

    def _hid_dtype(self) -> torch.dtype:
        return next(self.hid_extractor.parameters()).dtype

    def _head_dtype(self) -> torch.dtype:
        return next(self.head.parameters()).dtype

    def _encode_single_layer(self, last_hidden: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        target_dtype = self._hid_dtype()
        seqs = [last_hidden[i][token_mask[i].bool()].to(target_dtype) for i in range(last_hidden.shape[0])]

        if self.hidden_encoder_type == "lite":
            z_hid = torch.cat([self.hid_extractor(x.unsqueeze(0)) for x in seqs], dim=0)
        else:
            z_hid = self.hid_extractor(seqs)

        return z_hid.to(self._head_dtype())

    def _encode_multi_layer(self, hidden_states: Sequence[torch.Tensor], token_mask: torch.Tensor) -> torch.Tensor:
        target_dtype = self._hid_dtype()
        layer_hidden = [hidden_states[i + 1].to(target_dtype) for i in self.selected_hidden_layer_indices]
        stacked = torch.stack(layer_hidden, dim=1)
        seqs = [stacked[b][:, token_mask[b].bool(), :] for b in range(stacked.shape[0])]
        z_hid = self.hid_extractor(seqs, layer_ids=self.selected_hidden_layer_indices)
        return z_hid.to(self._head_dtype())

    def forward(
        self,
        *,
        last_hidden: torch.Tensor,
        hidden_states: Optional[Sequence[torch.Tensor]],
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.hidden_encoder_type == "strong_multi":
            z_hid = self._encode_multi_layer(hidden_states, token_mask)
        else:
            z_hid = self._encode_single_layer(last_hidden, token_mask)
        return self.head(z_att=None, z_conf=None, z_hid=z_hid, return_penultimate=False)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device, fp_dtype: Optional[torch.dtype]) -> Dict[str, Any]:
    int_keys = {
        "input_ids",
        "position_ids",
        "cache_position",
        "labels",
        "image_grid_thw",
        "video_grid_thw",
        "head_token_mask",
        "attention_mask",
        "token_type_ids",
    }
    out = {}
    for k, v in batch.items():
        if not torch.is_tensor(v):
            out[k] = v
            continue
        t = v.to(device, non_blocking=True)
        if k in int_keys:
            out[k] = t.long()
        elif fp_dtype is not None and t.is_floating_point():
            out[k] = t.to(fp_dtype)
        else:
            out[k] = t
    return out


def count_params(module: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def compute_metrics_from_confusion(conf: torch.Tensor) -> Dict[str, float]:
    conf = conf.float()
    tp = torch.diag(conf)
    row = conf.sum(dim=1)
    col = conf.sum(dim=0)
    prec = tp / torch.clamp(col, min=1.0)
    rec = tp / torch.clamp(row, min=1.0)
    f1 = 2 * prec * rec / torch.clamp(prec + rec, min=1e-8)
    return {
        "acc": (tp.sum() / torch.clamp(conf.sum(), min=1.0)).item(),
        "macro_precision": prec.mean().item(),
        "macro_recall": rec.mean().item(),
        "macro_f1": f1.mean().item(),
    }