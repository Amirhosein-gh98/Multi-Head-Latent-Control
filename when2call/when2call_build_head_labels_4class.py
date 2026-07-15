#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Annotate When2Call train_sft + train_pref for three binary auxiliary heads.

Goal:
- Use Qwen3-30B-A3B-Instruct(-2507) as an annotator.
- For each row, decide whether the row is usable for each head and assign a binary label.
- Save a clean JSONL file (and optional parquet export) that can later be used for
  generation and auxiliary-head training.

Heads:
1) need_tool
   1 if answering the request requires an external tool / external capability.
2) can_answer_with_given_tools
   1 if, with the current conversation state and the provided tools, the assistant can proceed now.
   This includes either a valid tool call now or a direct answer when no tool is needed.
3) need_more_info
   1 if the reason the assistant cannot proceed now is missing user information.

The script asks the annotator to identify a latent behavior category first and then
maps that category deterministically to both the binary labels and the 4-class
behavior target used by the behavior pipeline. The trained head itself still has
three sigmoid outputs: tool_call, request_for_info, cannot_answer.
direct_answer is represented implicitly as [0, 0, 0].

Important:
- direct_answer is still allowed as an annotator latent category so the labeler can
  recognize rows that do not need tools.
- direct_answer rows remain fully usable. The saved dataset contains a 4-class
behavior label:
  tool_call, request_for_info, cannot_answer, direct_answer
and also explicit 3-output head targets where direct_answer maps to 0,0,0.
"""

import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------

DEFAULT_MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
DEFAULT_DATASET_ID = "nvidia/When2Call"
DEFAULT_TOKENIZER_ID = DEFAULT_MODEL_ID
DEFAULT_OUTPUT_DIR = "data/train/when2call/when2call_processed"
DEFAULT_SPLITS = ["train_sft", "train_pref"]

BEHAVIOR_TO_ID = {
    "tool_call": 0,
    "request_for_info": 1,
    "cannot_answer": 2,
    "direct_answer": 3,
}

HEAD_TARGETS = {
    "tool_call": {
        "label_tool_call": 1,
        "label_request_for_info": 0,
        "label_cannot_answer": 0,
    },
    "request_for_info": {
        "label_tool_call": 0,
        "label_request_for_info": 1,
        "label_cannot_answer": 0,
    },
    "cannot_answer": {
        "label_tool_call": 0,
        "label_request_for_info": 0,
        "label_cannot_answer": 1,
    },
    "direct_answer": {
        "label_tool_call": 0,
        "label_request_for_info": 0,
        "label_cannot_answer": 0,
    },
}

CATEGORY_TO_LABELS: Dict[str, Dict[str, int]] = {
    "tool_call": {
        "label_need_tool": 1,
        "label_can_answer_with_given_tools": 1,
        "label_need_more_info": 0,
    },
    "request_for_info": {
        "label_need_tool": 1,
        "label_can_answer_with_given_tools": 0,
        "label_need_more_info": 1,
    },
    "cannot_answer": {
        "label_need_tool": 1,
        "label_can_answer_with_given_tools": 0,
        "label_need_more_info": 0,
    },
    "direct_answer": {
        "label_need_tool": 0,
        "label_can_answer_with_given_tools": 1,
        "label_need_more_info": 0,
    },
}

CATEGORY_NORMALIZATION = {
    "tool": "tool_call",
    "toolcall": "tool_call",
    "tool_call": "tool_call",
    "tool-call": "tool_call",
    "call_tool": "tool_call",
    "request_for_info": "request_for_info",
    "request-info": "request_for_info",
    "request_info": "request_for_info",
    "follow_up": "request_for_info",
    "follow-up": "request_for_info",
    "follow up": "request_for_info",
    "followup": "request_for_info",
    "clarification": "request_for_info",
    "clarify": "request_for_info",
    "ask_for_info": "request_for_info",
    "ask-user": "request_for_info",
    "cannot_answer": "cannot_answer",
    "cannot-answer": "cannot_answer",
    "unable_to_answer": "cannot_answer",
    "unable-to-answer": "cannot_answer",
    "unable": "cannot_answer",
    "refuse": "cannot_answer",
    "refusal": "cannot_answer",
    "direct": "direct_answer",
    "direct_answer": "direct_answer",
    "direct-answer": "direct_answer",
    "answer_directly": "direct_answer",
    "answer": "direct_answer",
    "ambiguous": "ambiguous",
    "uncertain": "ambiguous",
    "unsure": "ambiguous",
    "unusable": "ambiguous",
}

TOOLCALL_PATTERNS = [
    re.compile(r"<\s*TOOLCALL\s*>", flags=re.IGNORECASE),
    re.compile(r"<\s*tool_call\s*>", flags=re.IGNORECASE),
    re.compile(r'"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:'),
]

FOLLOWUP_PATTERNS = [
    re.compile(r"could you please", flags=re.IGNORECASE),
    re.compile(r"please specify", flags=re.IGNORECASE),
    re.compile(r"please provide", flags=re.IGNORECASE),
    re.compile(r"which [a-zA-Z_ -]+\?*$", flags=re.IGNORECASE),
    re.compile(r"what [a-zA-Z_ -]+\?*$", flags=re.IGNORECASE),
    re.compile(r"i need (?:the )?(?:following|more|additional)", flags=re.IGNORECASE),
]

CANNOT_ANSWER_PATTERNS = [
    re.compile(r"unable to", flags=re.IGNORECASE),
    re.compile(r"cannot", flags=re.IGNORECASE),
    re.compile(r"can't", flags=re.IGNORECASE),
    re.compile(r"do not have access", flags=re.IGNORECASE),
    re.compile(r"don't have access", flags=re.IGNORECASE),
    re.compile(r"real-time information", flags=re.IGNORECASE),
    re.compile(r"provided tools", flags=re.IGNORECASE),
    re.compile(r"with the tools available", flags=re.IGNORECASE),
]


ANNOTATOR_SYSTEM_PROMPT = """You are a meticulous dataset annotator for tool-use supervision.
Your job is to decide whether a sample can supervise three binary auxiliary heads.
Return JSON only. Do not include markdown fences, explanations outside JSON, or extra text."""


def build_annotator_user_prompt(sample: Dict[str, Any]) -> str:
    return f"""
You are labeling a tool-use training example.

Definitions:
1) need_tool
- 1 if correctly answering the user's request requires an external tool, external API, real-time lookup, private/public database access, or another external capability beyond a pretrained LM's internal knowledge.
- 0 if the request can be answered directly without any tool.

2) can_answer_with_given_tools
- 1 if, with the current conversation state and the provided tools, the assistant can proceed now.
- This includes either:
  a) making a valid tool call now, or
  b) directly answering now without tools.
- 0 if the assistant cannot proceed now.

3) need_more_info
- 1 if the reason the assistant cannot proceed now is that the user is missing required information or parameters.
- 0 otherwise.

Latent behavior categories:
- tool_call: the correct assistant behavior is to call a tool now.
- request_for_info: the correct assistant behavior is to ask the user for missing required information.
- cannot_answer: the correct assistant behavior is to say it cannot answer with the provided tools.
- direct_answer: the correct assistant behavior is to answer directly without tool use.
- ambiguous: the row is too noisy, contradictory, or unclear to trust for supervision.

For non-ambiguous rows, the final binary labels are:
- tool_call -> need_tool=1, can_answer_with_given_tools=1, need_more_info=0
- request_for_info -> need_tool=1, can_answer_with_given_tools=0, need_more_info=1
- cannot_answer -> need_tool=1, can_answer_with_given_tools=0, need_more_info=0
- direct_answer -> need_tool=0, can_answer_with_given_tools=1, need_more_info=0

Row metadata:
- source split: {sample['source_split']}
- source row index: {sample['source_row_index']}

Provided tools:
{sample['tools_json_pretty']}

Conversation before the target response:
{sample['context_text']}

Correct / chosen assistant response:
{sample['gold_response']}

Return exactly this JSON schema:
{{
  "latent_category": "tool_call|request_for_info|cannot_answer|direct_answer|ambiguous",
  "usable_need_tool": 0 or 1,
  "usable_can_answer_with_given_tools": 0 or 1,
  "usable_need_more_info": 0 or 1,
  "confidence": 0.0 to 1.0,
  "reason": "very short reason"
}}

Rules:
- Be strict about ambiguity or contradictions.
- Use the assistant response as strong evidence of the intended correct behavior.
- If the sample is ambiguous, set latent_category to "ambiguous" and set all usable_* values to 0.
- JSON only.
""".strip()


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def content_to_text(content: Any) -> str:
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
                    # When2Call is text-only, but keep the conversion safe.
                    parts.append(f"<{item.get('type', 'content')}>" )
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p).strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def messages_to_text(messages: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for msg in messages:
        role = str(msg.get("role", "unknown")).strip().capitalize()
        txt = content_to_text(msg.get("content", ""))
        if txt:
            lines.append(f"{role}: {txt}")
    return "\n".join(lines).strip()


def last_user_text(messages: Sequence[Dict[str, Any]]) -> str:
    for msg in reversed(list(messages)):
        if str(msg.get("role", "")).lower() == "user":
            return content_to_text(msg.get("content", ""))
    return ""


def normalize_prompt_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role", "")).lower().strip()
        if role == "system":
            continue
        out.append({
            "role": role or "user",
            "content": content_to_text(msg.get("content", "")),
        })
    return out


def stringify_tools(tools: Any) -> str:
    return json.dumps(tools or [], ensure_ascii=False, indent=2, sort_keys=True)


def extract_gold_response(split_name: str, row: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    messages = row.get("messages") or []
    if not isinstance(messages, list):
        raise ValueError(f"messages must be a list, got {type(messages)}")

    if split_name == "train_sft":
        if not messages:
            raise ValueError("Empty messages in train_sft row")
        last = messages[-1]
        if str(last.get("role", "")).lower() != "assistant":
            raise ValueError("Expected last train_sft message to be assistant")
        gold_response = content_to_text(last.get("content", ""))
        prompt_messages = normalize_prompt_messages(messages[:-1])
        return prompt_messages, gold_response, None

    if split_name == "train_pref":
        chosen = row.get("chosen_response") or {}
        rejected = row.get("rejected_response") or {}
        gold_response = content_to_text(chosen.get("content", chosen))
        rejected_response = content_to_text(rejected.get("content", rejected))
        prompt_messages = normalize_prompt_messages(messages)
        return prompt_messages, gold_response, rejected_response

    raise ValueError(f"Unknown split name: {split_name}")


def obvious_response_category(text: str) -> Optional[str]:
    s = (text or "").strip()
    if not s:
        return None
    for pat in TOOLCALL_PATTERNS:
        if pat.search(s):
            return "tool_call"
    if s.endswith("?"):
        for pat in FOLLOWUP_PATTERNS:
            if pat.search(s):
                return "request_for_info"
    for pat in CANNOT_ANSWER_PATTERNS:
        if pat.search(s):
            return "cannot_answer"
    return None


_JSON_BLOCK_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


def extract_first_json_object(text: str) -> Dict[str, Any]:
    s = (text or "").strip()
    if not s:
        raise ValueError("Empty annotator output")

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        s = fence.group(1)

    match = _JSON_BLOCK_RE.search(s)
    if not match:
        raise ValueError(f"Could not find JSON object in annotator output: {text[:400]}")

    candidate = match.group(0)
    return json.loads(candidate)


def normalize_category(cat: Any) -> str:
    s = str(cat or "").strip().lower().replace(" ", "_")
    return CATEGORY_NORMALIZATION.get(s, "ambiguous")


def latent_category_to_behavior_fields(latent_category: str) -> Dict[str, Any]:
    latent_category = normalize_category(latent_category)
    if latent_category in BEHAVIOR_TO_ID:
        return {
            "behavior": latent_category,
            "behavior_class": int(BEHAVIOR_TO_ID[latent_category]),
            "usable_behavior": 1,
        }
    # ambiguous rows are intentionally excluded from the 4-class behavior target.
    return {
        "behavior": latent_category if latent_category else "ambiguous",
        "behavior_class": -1,
        "usable_behavior": 0,
    }


@dataclass
class CanonicalSample:
    sample_id: str
    source_split: str
    source_row_index: int
    source_dataset: str
    tools: Any
    tools_json: str
    prompt_messages_json: str
    context_text: str
    question: str
    gold_response: str
    rejected_response: Optional[str]
    heuristic_category: Optional[str]

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tools_json_pretty"] = self.tools_json
        return d


@dataclass
class AnnotationResult:
    latent_category: str
    usable_need_tool: int
    usable_can_answer_with_given_tools: int
    usable_need_more_info: int
    confidence: float
    reason: str
    raw_model_text: str


# --------------------------------------------------------------------------------------
# canonicalization
# --------------------------------------------------------------------------------------


def iter_canonical_samples(
    dataset_id: str,
    split_names: Sequence[str],
    max_rows_per_split: Optional[int],
) -> Iterable[CanonicalSample]:
    for split_name in split_names:
        ds = load_dataset(dataset_id, split_name, split="train")
        n = len(ds) if max_rows_per_split is None else min(len(ds), int(max_rows_per_split))
        for idx in range(n):
            row = ds[idx]
            prompt_messages, gold_response, rejected_response = extract_gold_response(split_name, row)
            tools = row.get("tools") or []
            tools_json = stringify_tools(tools)
            context_text = messages_to_text(prompt_messages)
            question = last_user_text(prompt_messages)
            heuristic = obvious_response_category(gold_response)
            yield CanonicalSample(
                sample_id=f"{split_name}:{idx}",
                source_split=split_name,
                source_row_index=int(idx),
                source_dataset=dataset_id,
                tools=tools,
                tools_json=tools_json,
                prompt_messages_json=json.dumps(prompt_messages, ensure_ascii=False),
                context_text=context_text,
                question=question,
                gold_response=gold_response,
                rejected_response=rejected_response,
                heuristic_category=heuristic,
            )


# --------------------------------------------------------------------------------------
# annotation
# --------------------------------------------------------------------------------------


def build_requests(tokenizer: AutoTokenizer, samples: Sequence[CanonicalSample]) -> List[str]:
    prompts: List[str] = []
    for sample in samples:
        row = sample.to_row()
        user_prompt = build_annotator_user_prompt(row)
        messages = [
            {"role": "system", "content": ANNOTATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return prompts


def parse_annotation(sample: CanonicalSample, raw_text: str) -> AnnotationResult:
    parsed = extract_first_json_object(raw_text)
    latent_category = normalize_category(parsed.get("latent_category"))

    # Strong obvious-tool-call override only for fully structural cases.
    if sample.heuristic_category == "tool_call":
        latent_category = "tool_call"

    if latent_category == "ambiguous":
        return AnnotationResult(
            latent_category="ambiguous",
            usable_need_tool=0,
            usable_can_answer_with_given_tools=0,
            usable_need_more_info=0,
            confidence=float(parsed.get("confidence", 0.0) or 0.0),
            reason=str(parsed.get("reason", "ambiguous"))[:400],
            raw_model_text=raw_text,
        )

    if latent_category not in CATEGORY_TO_LABELS:
        latent_category = "ambiguous"
        return AnnotationResult(
            latent_category=latent_category,
            usable_need_tool=0,
            usable_can_answer_with_given_tools=0,
            usable_need_more_info=0,
            confidence=0.0,
            reason="invalid latent category",
            raw_model_text=raw_text,
        )

    def _bit(name: str, default: int = 1) -> int:
        try:
            return int(parsed.get(name, default))
        except Exception:
            return int(default)

    return AnnotationResult(
        latent_category=latent_category,
        usable_need_tool=_bit("usable_need_tool", 1),
        usable_can_answer_with_given_tools=_bit("usable_can_answer_with_given_tools", 1),
        usable_need_more_info=_bit("usable_need_more_info", 1),
        confidence=max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0))),
        reason=str(parsed.get("reason", ""))[:400],
        raw_model_text=raw_text,
    )


# --------------------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------------------


def load_existing_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sid = str(obj.get("sample_id", ""))
                if sid:
                    done.add(sid)
            except Exception:
                continue
    return done


def append_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_jsonl_to_parquet(jsonl_path: Path, parquet_path: Path) -> None:
    ds = load_dataset("json", data_files=str(jsonl_path), split="train")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(parquet_path))


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", default=DEFAULT_DATASET_ID)
    ap.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--tokenizer_id", default=DEFAULT_TOKENIZER_ID)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max_rows_per_split", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.50)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=32000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--export_parquet", action="store_true")
    args = ap.parse_args()

    set_seed(int(args.seed))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "when2call_aux_labels.jsonl"
    parquet_path = out_dir / "when2call_aux_labels.parquet"
    stats_path = out_dir / "when2call_aux_labels_stats.json"

    done_ids = load_existing_ids(jsonl_path) if args.resume else set()
    if done_ids:
        print(f"[resume] found {len(done_ids)} already-labeled samples", flush=True)

    tokenizer_id = args.tokenizer_id or args.model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)

    llm_kwargs = {
        "model": args.model_id,
        "tokenizer": tokenizer_id,
        "trust_remote_code": True,
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_model_len": int(args.max_model_len),
        "dtype": args.dtype,
    }
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
    )

    buffer_samples: List[CanonicalSample] = []
    total_seen = 0
    total_written = 0
    stats = {
        "dataset_id": args.dataset_id,
        "model_id": args.model_id,
        "tokenizer_id": tokenizer_id,
        "quantization": args.quantization,
        "dtype": args.dtype,
        "total_seen": 0,
        "total_written": 0,
        "per_split": {},
        "latent_category_counts": {},
        "behavior_counts": {},
        "usable_counts": {
            "need_tool": 0,
            "can_answer_with_given_tools": 0,
            "need_more_info": 0,
        },
    }

    def _bump(d: Dict[str, Any], key: str, inc: int = 1) -> None:
        d[key] = int(d.get(key, 0)) + int(inc)

    def flush_batch(batch_samples: List[CanonicalSample]) -> None:
        nonlocal total_written
        if not batch_samples:
            return
        prompts = build_requests(tokenizer, batch_samples)
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        write_rows: List[Dict[str, Any]] = []
        for sample, out in zip(batch_samples, outputs):
            raw_text = out.outputs[0].text if out.outputs else ""
            try:
                ann = parse_annotation(sample, raw_text)
            except Exception as e:
                ann = AnnotationResult(
                    latent_category="ambiguous",
                    usable_need_tool=0,
                    usable_can_answer_with_given_tools=0,
                    usable_need_more_info=0,
                    confidence=0.0,
                    reason=f"parse_error: {type(e).__name__}: {e}"[:400],
                    raw_model_text=raw_text,
                )
            label_map = CATEGORY_TO_LABELS.get(ann.latent_category, {
                "label_need_tool": 0,
                "label_can_answer_with_given_tools": 0,
                "label_need_more_info": 0,
            })
            head_target_map = HEAD_TARGETS.get(ann.latent_category, {
                "label_tool_call": 0,
                "label_request_for_info": 0,
                "label_cannot_answer": 0,
            })
            row = sample.to_row()
            behavior_fields = latent_category_to_behavior_fields(ann.latent_category)
            row.update(label_map)
            row.update(head_target_map)
            row.update({
                "head_target_vector": [
                    int(head_target_map["label_tool_call"]),
                    int(head_target_map["label_request_for_info"]),
                    int(head_target_map["label_cannot_answer"]),
                ],
                "usable_need_tool": int(ann.usable_need_tool),
                "usable_can_answer_with_given_tools": int(ann.usable_can_answer_with_given_tools),
                "usable_need_more_info": int(ann.usable_need_more_info),
                "latent_category": ann.latent_category,
                "annotation_confidence": float(ann.confidence),
                "annotation_reason": ann.reason,
                "annotator_raw": ann.raw_model_text,
            })
            row.update(behavior_fields)
            write_rows.append(row)

            split_stats = stats["per_split"].setdefault(sample.source_split, {"rows": 0})
            _bump(split_stats, "rows")
            _bump(stats["latent_category_counts"], ann.latent_category)
            _bump(stats["behavior_counts"], str(behavior_fields.get("behavior", ann.latent_category)))
            _bump(stats["usable_counts"], "need_tool", ann.usable_need_tool)
            _bump(stats["usable_counts"], "can_answer_with_given_tools", ann.usable_can_answer_with_given_tools)
            _bump(stats["usable_counts"], "need_more_info", ann.usable_need_more_info)

        append_jsonl(jsonl_path, write_rows)
        total_written += len(write_rows)
        print(f"[write] appended {len(write_rows)} rows -> {jsonl_path}", flush=True)

    for sample in iter_canonical_samples(args.dataset_id, args.splits, args.max_rows_per_split):
        total_seen += 1
        stats["total_seen"] = total_seen
        if sample.sample_id in done_ids:
            continue
        buffer_samples.append(sample)
        if len(buffer_samples) >= int(args.batch_size):
            flush_batch(buffer_samples)
            buffer_samples = []

    flush_batch(buffer_samples)
    stats["total_written"] = total_written + len(done_ids)
    save_json(stats_path, stats)
    print(f"[done] total_seen={total_seen} total_new_rows={total_written}", flush=True)
    print(f"[done] stats saved to {stats_path}", flush=True)

    if args.export_parquet:
        export_jsonl_to_parquet(jsonl_path, parquet_path)
        print(f"[done] parquet saved to {parquet_path}", flush=True)


if __name__ == "__main__":
    main()
