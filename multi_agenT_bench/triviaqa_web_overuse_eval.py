# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# from __future__ import annotations

# import argparse
# import csv
# import gc
# import json
# import math
# import re
# from dataclasses import asdict
# from pathlib import Path
# from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# from tqdm.auto import tqdm
# from transformers import AutoProcessor, AutoTokenizer

# from compact_multi_agent_shared_optimized_v4_textbench import (
#     MultiAgentOrchestrator,
#     TokenUsage,
#     build_generation_row,
#     build_model_bundle,
#     debug_print,
#     evaluate_saved_row,
#     get_example_image_for_benchmark,
#     json_dump,
#     load_examples_for_benchmark,
#     response_final_answer_status,
#     write_jsonl,
# )


# # =============================================================================
# # DEFAULTS
# # =============================================================================

# BENCHMARK = "triviaqa"

# DEFAULT_DATASET_CFG = {
#     "data_mode": "hf",
#     "dataset_name": "mandarjoshi/trivia_qa",
#     "dataset_config_name": "rc",
#     "split": "validation",
#     "max_samples": 1000,
# }

# DEFAULT_SAMPLING_PROFILES = {
#     "default": {
#         "greedy": False,
#         "temperature": 0.7,
#         "top_p": 0.8,
#         "top_k": 20,
#         "repetition_penalty": 1.0,
#         "presence_penalty": 0.0,
#         "max_new_tokens": 15000,
#     },
#     "thinking": {
#         "greedy": False,
#         "temperature": 1.0,
#         "top_p": 0.95,
#         "top_k": 20,
#         "repetition_penalty": 1.0,
#         "presence_penalty": 0.0,
#         "max_new_tokens": 16000,
#     },
#     "instruct": {
#         "greedy": False,
#         "temperature": 0.7,
#         "top_p": 0.8,
#         "top_k": 20,
#         "repetition_penalty": 1.0,
#         "presence_penalty": 1.5,
#         "max_new_tokens": 15000,
#     },
# }

# DEFAULT_AUX_PROFILE = {
#     "trust_remote_code": True,
#     "prefer_unsloth_mirror": False,
#     "dtype": "bf16",
#     "max_seq_len": 32768,
#     "max_pixels": 200000,
#     "attn_implementation": "flash_attention_3",
#     "regression_threshold": 0.6,
#     "head_input_mode": "completion_text_only",
#     "hidden_encoder_type": "lite",
#     "hidden_layer_selection": "last",
#     "hidden_layer_index": None,
#     "hidden_layer_indices": None,
# }

# DEFAULT_VLLM_RUNTIME = {
#     "dtype": "bfloat16",
#     "max_model_len": 32768,
#     "tensor_parallel_size": 1,
#     "gpu_memory_utilization": 0.40,
#     "max_num_seqs": 128,
#     "enforce_eager": False,
#     "trust_remote_code": False,
#     "limit_mm_images": 1,
# }

# DEFAULT_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

# TOOL_CALL_PATTERN = re.compile(r"<\s*web_search\s*>(.*?)<\s*/\s*web_search\s*>", flags=re.IGNORECASE | re.DOTALL)


# # =============================================================================
# # ARGPARSE / GENERAL HELPERS
# # =============================================================================


# def parse_args() -> argparse.Namespace:
#     ap = argparse.ArgumentParser(description="TriviaQA tool-overuse evaluator with optional aux-head gating.")
#     ap.add_argument("--model_name_or_path", type=str, required=True)
#     ap.add_argument("--model_family", type=str, default="auto", choices=["auto", "qwen3_5", "qwen3", "qwen3_vl", "gemma4", "other"])
#     ap.add_argument("--thinking_mode", type=str, default="auto", choices=["auto", "on", "off"])
#     ap.add_argument("--aux_head_ckpt", type=str, default="")
#     ap.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_CFG["dataset_name"])
#     ap.add_argument("--dataset_config_name", type=str, default=DEFAULT_DATASET_CFG["dataset_config_name"])
#     ap.add_argument("--split", type=str, default=DEFAULT_DATASET_CFG["split"])
#     ap.add_argument("--max_samples", type=int, default=DEFAULT_DATASET_CFG["max_samples"])
#     ap.add_argument("--batch_size", type=int, default=64)
#     ap.add_argument("--thresholds", type=str, default=",".join(str(x) for x in DEFAULT_THRESHOLDS))
#     ap.add_argument("--output_dir", type=str, required=True)
#     ap.add_argument("--call_tool_if_missing_final_answer", action="store_true", default=True)
#     ap.add_argument("--no_call_tool_if_missing_final_answer", dest="call_tool_if_missing_final_answer", action="store_false")
#     ap.add_argument("--debug", action="store_true")
#     return ap.parse_args()


# def parse_thresholds(csv_text: str) -> List[float]:
#     vals: List[float] = []
#     for part in str(csv_text or "").split(","):
#         part = part.strip()
#         if not part:
#             continue
#         vals.append(float(part))
#     if not vals:
#         raise RuntimeError("No thresholds were provided.")
#     vals = sorted(set(vals))
#     return vals


# def sanitize_name_for_path(text: str) -> str:
#     s = str(text or "").strip().lower()
#     s = re.sub(r"[^a-z0-9]+", "_", s)
#     s = re.sub(r"_+", "_", s).strip("_")
#     return s or "model"


# def threshold_tag(x: float) -> str:
#     return f"{x:.2f}".rstrip("0").rstrip(".").replace(".", "p")


# def chunked(seq: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
#     for start in range(0, len(seq), batch_size):
#         yield seq[start : start + batch_size]


# def ensure_dir(path: Path) -> None:
#     path.mkdir(parents=True, exist_ok=True)


# def _benchmark_is_multimodal(benchmark: str) -> bool:
#     return str(benchmark or "").strip().lower() in {
#         "mathvista", "mathverse", "charxiv_reasoning", "screenspot_pro", "simplevqa"
#     }


# def _benchmark_is_reasoning_text(benchmark: str) -> bool:
#     return str(benchmark or "").strip().lower() in {"math", "mmlu_pro"}


# def _resolve_thinking_enabled_local(model_name_or_path: str, model_family: str, thinking_mode: str) -> bool:
#     mode = str(thinking_mode or "auto").strip().lower()
#     if mode in {"on", "true", "1", "yes"}:
#         return True
#     if mode in {"off", "false", "0", "no"}:
#         return False
#     lname = str(model_name_or_path or "").lower()
#     family = str(model_family or "auto").lower()
#     if family == "qwen3_5":
#         if any(x in lname for x in ["qwen3.5-0.8b", "qwen3.5-2b"]):
#             return False
#         return True
#     if family == "qwen3_vl":
#         return "thinking" in lname
#     if family == "qwen3":
#         return "instruct" not in lname or "thinking" in lname
#     if family == "gemma4":
#         return False
#     return "thinking" in lname


# def _official_sampling_override_for_model(*, model_name_or_path: str, model_family: str, thinking_mode: str, benchmark: str) -> Dict[str, Any]:
#     family = str(model_family or "auto").strip().lower()
#     if family == "auto":
#         lname = str(model_name_or_path or "").lower()
#         if "qwen3.5" in lname:
#             family = "qwen3_5"
#         elif "qwen3-vl" in lname:
#             family = "qwen3_vl"
#         elif "gemma-4" in lname:
#             family = "gemma4"
#         elif "qwen3" in lname:
#             family = "qwen3"
#         else:
#             family = "other"

#     thinking_enabled = _resolve_thinking_enabled_local(model_name_or_path, family, thinking_mode)
#     multimodal = _benchmark_is_multimodal(benchmark)
#     text_reasoning = _benchmark_is_reasoning_text(benchmark)

#     if family == "gemma4":
#         return {
#             "greedy": False,
#             "temperature": 1.0,
#             "top_p": 0.95,
#             "top_k": 64,
#             "repetition_penalty": 1.0,
#             "presence_penalty": 0.0,
#             "max_new_tokens": 16000,
#         }

#     if family == "qwen3_5":
#         if multimodal:
#             if thinking_enabled:
#                 return {
#                     "greedy": False,
#                     "temperature": 0.6,
#                     "top_p": 0.95,
#                     "top_k": 20,
#                     "repetition_penalty": 1.0,
#                     "presence_penalty": 1.5,
#                     "max_new_tokens": 16000,
#                 }
#             return {
#                 "greedy": False,
#                 "temperature": 0.7,
#                 "top_p": 0.8,
#                 "top_k": 20,
#                 "repetition_penalty": 1.0,
#                 "presence_penalty": 1.5,
#                 "max_new_tokens": 15000,
#             }
#         if text_reasoning:
#             if thinking_enabled:
#                 return {
#                     "greedy": False,
#                     "temperature": 0.6,
#                     "top_p": 0.95,
#                     "top_k": 20,
#                     "repetition_penalty": 1.0,
#                     "presence_penalty": 0.0,
#                     "max_new_tokens": 16000,
#                 }
#             return {
#                 "greedy": False,
#                 "temperature": 0.7,
#                 "top_p": 0.8,
#                 "top_k": 20,
#                 "repetition_penalty": 1.0,
#                 "presence_penalty": 0.0,
#                 "max_new_tokens": 15000,
#             }

#     return {}


# def _apply_runtime_override(base: Dict[str, Any]) -> Dict[str, Any]:
#     return dict(base)


# def _parquet_safe_value(value: Any) -> Any:
#     if isinstance(value, (dict, list, tuple, set)):
#         return json.dumps(value, ensure_ascii=False)
#     return value


# def save_rows_to_parquet(path: Path, rows: Iterable[Dict[str, Any]], debug: bool = False) -> bool:
#     rows = list(rows)
#     try:
#         import pandas as pd

#         flat_rows = [{k: _parquet_safe_value(v) for k, v in row.items()} for row in rows]
#         df = pd.DataFrame(flat_rows)
#         path.parent.mkdir(parents=True, exist_ok=True)
#         df.to_parquet(path, index=False)
#         debug_print(debug, f"saved {path}")
#         return True
#     except Exception as e:
#         err_path = path.with_suffix(path.suffix + ".error.txt")
#         err_path.write_text(f"Parquet export failed: {type(e).__name__}: {e}\n", encoding="utf-8")
#         debug_print(debug, f"Parquet export failed for {path}: {type(e).__name__}: {e}")
#         return False


# # =============================================================================
# # LOCAL PROMPT-BUILDER PATCHING
# # =============================================================================


# def _remove_gemma_think_prefix_local(text: str) -> str:
#     text = text or ""
#     return re.sub(r"^\s*<\|think\|>\s*\n?", "", text, count=1)


# def _patch_gemma_messages_local(messages: List[Dict[str, Any]], thinking_enabled: bool) -> List[Dict[str, Any]]:
#     patched: List[Dict[str, Any]] = []
#     for msg in messages:
#         cloned = dict(msg)
#         if isinstance(cloned.get("content"), list):
#             cloned["content"] = list(cloned["content"])
#         patched.append(cloned)

#     if not patched or patched[0].get("role") != "system":
#         if thinking_enabled:
#             patched.insert(0, {"role": "system", "content": "<|think|>"})
#         return patched

#     system_content = patched[0].get("content", "")
#     if not isinstance(system_content, str):
#         return patched

#     system_content = _remove_gemma_think_prefix_local(system_content)
#     if thinking_enabled:
#         system_content = "<|think|>\n" + system_content if system_content else "<|think|>"
#     patched[0]["content"] = system_content
#     return patched


# def _patch_chat_template_callable_local(bound_callable, model_family: str, thinking_enabled: bool):
#     def patched(messages, *args, **kwargs):
#         if model_family == "gemma4":
#             messages = _patch_gemma_messages_local(messages, thinking_enabled)
#         elif model_family in {"qwen3_5", "qwen3"}:
#             kwargs = dict(kwargs)
#             kwargs.setdefault("enable_thinking", thinking_enabled)
#         return bound_callable(messages, *args, **kwargs)
#     return patched


# def _patch_processor_for_runtime_prompting_local(processor_like, model_family: str, thinking_enabled: bool):
#     if hasattr(processor_like, "apply_chat_template"):
#         processor_like.apply_chat_template = _patch_chat_template_callable_local(
#             processor_like.apply_chat_template, model_family, thinking_enabled
#         )
#     tokenizer = getattr(processor_like, "tokenizer", None)
#     if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
#         tokenizer.apply_chat_template = _patch_chat_template_callable_local(
#             tokenizer.apply_chat_template, model_family, thinking_enabled
#         )
#     return processor_like


# def ensure_runtime_prompt_builder(runtime) -> None:
#     if getattr(runtime, "_processor", None) is not None:
#         return
#     cfg = getattr(runtime, "cfg", None)
#     model_name_or_path = getattr(cfg, "model_name_or_path", "")
#     trust_remote_code = bool(getattr(cfg, "trust_remote_code", False))
#     model_family = str(getattr(cfg, "model_family", "auto") or "auto").strip().lower()
#     if model_family == "auto":
#         lname = str(model_name_or_path or "").lower()
#         if "qwen3.5" in lname:
#             model_family = "qwen3_5"
#         elif "qwen3-vl" in lname:
#             model_family = "qwen3_vl"
#         elif "gemma-4" in lname or "gemma4" in lname:
#             model_family = "gemma4"
#         elif "qwen3" in lname:
#             model_family = "qwen3"
#         else:
#             model_family = "other"
#     thinking_enabled = _resolve_thinking_enabled_local(
#         model_name_or_path=str(model_name_or_path),
#         model_family=model_family,
#         thinking_mode=str(getattr(cfg, "thinking_mode", "auto")),
#     )

#     processor_like = None
#     load_errors = []

#     # First choice: ask vLLM for the tokenizer it is already using.
#     # This stays closest to your original working generation path and avoids
#     # re-instantiating a separate HF tokenizer/processor inside this script.
#     try:
#         llm = runtime.llm
#         for getter_name in ("get_tokenizer", "get_tokenizer_group"):
#             getter = getattr(llm, getter_name, None)
#             if getter is None:
#                 continue
#             try:
#                 candidate = getter()
#                 if getter_name == "get_tokenizer_group":
#                     candidate = getattr(candidate, "tokenizer", None) or getattr(candidate, "tokenizer_obj", None)
#                 if candidate is not None and hasattr(candidate, "apply_chat_template"):
#                     processor_like = candidate
#                     break
#             except Exception as e:
#                 load_errors.append(f"vllm.{getter_name}: {type(e).__name__}: {e}")
#         if processor_like is None:
#             tok = getattr(llm, "get_tokenizer", lambda: None)()
#             if tok is not None and hasattr(tok, "apply_chat_template"):
#                 processor_like = tok
#     except Exception as e:
#         load_errors.append(f"runtime.llm tokenizer bootstrap: {type(e).__name__}: {e}")
#         processor_like = None

#     if processor_like is None and model_family != "gemma4":
#         try:
#             processor_like = AutoProcessor.from_pretrained(
#                 model_name_or_path,
#                 trust_remote_code=trust_remote_code,
#             )
#         except Exception as e:
#             load_errors.append(f"AutoProcessor: {type(e).__name__}: {e}")
#             processor_like = None

#     if processor_like is None and model_family == "gemma4":
#         # Only as a last resort for Gemma.
#         for kwargs in (
#             {"use_fast": False},
#             {"use_fast": True},
#             {},
#         ):
#             try:
#                 processor_like = AutoTokenizer.from_pretrained(
#                     model_name_or_path,
#                     trust_remote_code=trust_remote_code,
#                     **kwargs,
#                 )
#                 break
#             except Exception as e:
#                 load_errors.append(f"AutoTokenizer kwargs={kwargs}: {type(e).__name__}: {e}")
#                 processor_like = None

#     if processor_like is None:
#         raise RuntimeError(
#             "Failed to initialize prompt builder. " + " | ".join(load_errors)
#         )

#     processor_like = _patch_processor_for_runtime_prompting_local(
#         processor_like,
#         model_family=model_family,
#         thinking_enabled=thinking_enabled,
#     )
#     runtime._processor = processor_like



# # =============================================================================
# # TOOL PROMPTING / PARSING
# # =============================================================================


# def build_tool_prompt_text(ex: Dict[str, Any]) -> str:
#     question = str(ex.get("question") or ex.get("prompt_text") or "").strip()
#     return (
#         f"{question}\n\n"
#         "You may optionally use a web search tool named WEB_SEARCH. "
#         "Use the tool only when you are genuinely uncertain and need external verification. "
#         "Do not use the tool if you already know the answer with high confidence.\n\n"
#         "If you want to use the tool, output exactly one line and nothing else in this format:\n"
#         "<web_search>your search query</web_search>\n\n"
#         "Otherwise, answer directly and put your final answer only once at the end inside \\boxed{...}."
#     )



# def build_tool_messages(ex: Dict[str, Any]) -> List[Dict[str, Any]]:
#     content: List[Dict[str, Any]] = []
#     if get_example_image_for_benchmark(ex) is not None:
#         content.append({"type": "image"})
#     content.append({"type": "text", "text": build_tool_prompt_text(ex)})
#     return [{"role": "user", "content": content}]



# def parse_tool_call(text: str) -> Tuple[bool, Optional[str]]:
#     raw = str(text or "")
#     m = TOOL_CALL_PATTERN.search(raw)
#     if m:
#         query = re.sub(r"\s+", " ", m.group(1)).strip()
#         return True, (query or None)

#     stripped = raw.strip()
#     upper = stripped.upper()
#     if upper.startswith("WEB_SEARCH:"):
#         query = stripped.split(":", 1)[1].strip()
#         return True, (query or None)
#     return False, None


# # =============================================================================
# # METRICS
# # =============================================================================


# def safe_float(x: Any, default: float = 0.0) -> float:
#     try:
#         if x is None:
#             return default
#         return float(x)
#     except Exception:
#         return default



# def safe_int(x: Any, default: int = 0) -> int:
#     try:
#         if x is None:
#             return default
#         return int(x)
#     except Exception:
#         return default



# def mean_binary(vals: Sequence[int]) -> float:
#     return float(sum(int(v) for v in vals)) / max(len(vals), 1)



# def compute_ece(y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 10) -> Optional[float]:
#     if not y_true or not y_prob or len(y_true) != len(y_prob):
#         return None
#     total = len(y_true)
#     ece = 0.0
#     for i in range(n_bins):
#         lo = i / n_bins
#         hi = (i + 1) / n_bins
#         if i == n_bins - 1:
#             idx = [j for j, p in enumerate(y_prob) if lo <= p <= hi]
#         else:
#             idx = [j for j, p in enumerate(y_prob) if lo <= p < hi]
#         if not idx:
#             continue
#         acc = sum(int(y_true[j]) for j in idx) / len(idx)
#         conf = sum(float(y_prob[j]) for j in idx) / len(idx)
#         ece += (len(idx) / total) * abs(acc - conf)
#     return float(ece)



# def compute_aux_binary_metrics(y_true: Sequence[int], y_prob: Sequence[float]) -> Dict[str, Any]:
#     metrics: Dict[str, Any] = {
#         "num_rows": len(y_true),
#         "auroc": None,
#         "average_precision": None,
#         "ece": compute_ece(y_true, y_prob),
#     }
#     try:
#         from sklearn.metrics import average_precision_score, roc_auc_score

#         if len(set(int(x) for x in y_true)) >= 2:
#             metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
#         if sum(int(x) for x in y_true) > 0:
#             metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
#     except Exception:
#         pass
#     return metrics


# # =============================================================================
# # GENERATION + EVAL HELPERS
# # =============================================================================


# def build_model_bundle_single(args: argparse.Namespace):
#     sampling_override = _official_sampling_override_for_model(
#         model_name_or_path=args.model_name_or_path,
#         model_family=args.model_family,
#         thinking_mode=args.thinking_mode,
#         benchmark=BENCHMARK,
#     )
#     runtime_profile = _apply_runtime_override({
#         **DEFAULT_VLLM_RUNTIME,
#         "model_family": args.model_family,
#         "thinking_mode": args.thinking_mode,
#     })
#     aux_profile = {
#         **DEFAULT_AUX_PROFILE,
#         "model_family": args.model_family,
#         "thinking_mode": args.thinking_mode,
#     }
#     return build_model_bundle(
#         model_name_or_path=args.model_name_or_path,
#         aux_head_ckpt=args.aux_head_ckpt,
#         runtime_profile=runtime_profile,
#         sampling_profiles=DEFAULT_SAMPLING_PROFILES,
#         aux_profile=aux_profile,
#         sampling_override=sampling_override,
#         model_family=args.model_family,
#         thinking_mode=args.thinking_mode,
#     )



# def build_empty_usage(orchestrator: MultiAgentOrchestrator, prompt_tokens: int, completion_tokens: int, generation_time_sec: float) -> Dict[str, TokenUsage]:
#     usage = orchestrator._new_usage_by_model()
#     usage["model1"].prompt_tokens += int(prompt_tokens)
#     usage["model1"].completion_tokens += int(completion_tokens)
#     usage["model1"].generation_calls += 1
#     usage["model1"].generation_time_sec += float(generation_time_sec)
#     return usage



# def run_generation_pass(
#     examples: List[Dict[str, Any]],
#     orchestrator: MultiAgentOrchestrator,
#     prompt_variant: str,
#     batch_size: int,
#     debug: bool,
# ) -> List[Dict[str, Any]]:
#     runtime = orchestrator.get_generator("model1")
#     ensure_runtime_prompt_builder(runtime)
#     bundle = orchestrator.model_bundles["model1"]
#     rows: List[Dict[str, Any]] = []

#     desc = "generate_standard" if prompt_variant == "standard_no_tool" else "generate_tool_enabled"
#     for batch in tqdm(chunked(examples, batch_size), total=math.ceil(len(examples) / max(batch_size, 1)), desc=desc, unit="batch", dynamic_ncols=True):
#         messages_list: List[List[Dict[str, Any]]] = []
#         images: List[Optional[Any]] = []
#         batch_examples: List[Dict[str, Any]] = []
#         for ex in batch:
#             if prompt_variant == "standard_no_tool":
#                 messages = orchestrator._build_messages_for_turn(ex, None)
#             elif prompt_variant == "tool_enabled":
#                 messages = build_tool_messages(ex)
#             else:
#                 raise RuntimeError(f"Unknown prompt_variant={prompt_variant}")
#             messages_list.append(messages)
#             images.append(get_example_image_for_benchmark(ex))
#             batch_examples.append(ex)

#         gens = runtime.generate_batch(
#             messages_list=messages_list,
#             images=images,
#             sampling_cfg=bundle.sampling_cfg,
#             continue_final_messages=[False] * len(batch_examples),
#         )
#         if len(gens) != len(batch_examples):
#             raise RuntimeError(f"Expected {len(batch_examples)} generations, got {len(gens)}")

#         for ex, messages, gen in zip(batch_examples, messages_list, gens):
#             usage = build_empty_usage(orchestrator, prompt_tokens=gen.prompt_tokens, completion_tokens=gen.completion_tokens, generation_time_sec=float(gen.generation_time_sec))
#             trace = [{"event": "generation", "model": "model1", "prompt_variant": prompt_variant}]
#             row = build_generation_row(
#                 benchmark=BENCHMARK,
#                 ex=ex,
#                 strategy_name=prompt_variant,
#                 final_model_name="model1",
#                 final_response=gen.text,
#                 usage_by_model=usage,
#                 trace=trace,
#                 wall_time_sec=float(gen.generation_time_sec),
#             )
#             row["messages"] = messages
#             row["prompt_variant"] = prompt_variant
#             rows.append(row)
#     debug_print(debug, f"Generated {len(rows)} rows for prompt_variant={prompt_variant}")
#     return rows



# def score_standard_rows_with_aux(
#     examples: List[Dict[str, Any]],
#     rows: List[Dict[str, Any]],
#     orchestrator: MultiAgentOrchestrator,
#     debug: bool,
#     require_aux: bool = False,
# ) -> List[Dict[str, Any]]:
#     out: List[Dict[str, Any]] = []
#     aux_runtime = orchestrator.get_aux("model1")
#     has_aux = aux_runtime is not None
#     if require_aux and not has_aux:
#         raise RuntimeError(
#             "Aux head was requested, but orchestrator.get_aux('model1') returned None. "
#             "Check that --aux_head_ckpt is set and that build_model_bundle(...) received it."
#         )
#     if has_aux and aux_runtime is not None:
#         aux_runtime.load()

#     for ex, row in tqdm(list(zip(examples, rows)), total=len(rows), desc="score_aux_standard", unit="row", dynamic_ncols=True):
#         evaluated = evaluate_saved_row(BENCHMARK, row, judge_runtime=None, judge_sampling=None)
#         info = response_final_answer_status(BENCHMARK, row.get("raw_response", ""))
#         aux_prob: Optional[float] = None
#         aux_pred: Optional[int] = None
#         if has_aux:
#             score = orchestrator._score_response(ex, "model1", str(row.get("raw_response") or ""))
#             aux_prob = float(score.prob_correct)
#             aux_pred = int(score.pred)
#             evaluated.setdefault("usage_by_model", row.get("usage_by_model", {}))
#             try:
#                 evaluated["usage_by_model"]["model1"]["aux_calls"] = safe_int(evaluated["usage_by_model"]["model1"].get("aux_calls", 0)) + 1
#                 evaluated["usage_by_model"]["model1"]["aux_scored_tokens"] = safe_int(
#                     evaluated["usage_by_model"]["model1"].get("aux_scored_tokens", 0)
#                 ) + safe_int(evaluated.get("usage_by_model", {}).get("model1", {}).get("completion_tokens", 0))
#             except Exception:
#                 pass
#         merged = {
#             **evaluated,
#             "base_has_final_answer": bool(info.get("has_final_answer", False)),
#             "base_final_answer_reason": str(info.get("reason", "unknown")),
#             "aux_enabled_for_run": bool(has_aux),
#             "aux_prob_correct": aux_prob,
#             "aux_pred": aux_pred,
#             "base_correct": 0 if evaluated.get("judge_label") is None else int(evaluated.get("judge_label", 0)),
#         }
#         out.append(merged)
#     debug_print(debug, f"Scored {len(out)} standard rows with aux")
#     return out



# def evaluate_tool_enabled_rows(
#     base_rows: List[Dict[str, Any]],
#     tool_rows: List[Dict[str, Any]],
# ) -> List[Dict[str, Any]]:
#     if len(base_rows) != len(tool_rows):
#         raise RuntimeError(f"Length mismatch: base_rows={len(base_rows)} tool_rows={len(tool_rows)}")

#     out: List[Dict[str, Any]] = []
#     for base_row, tool_row in tqdm(list(zip(base_rows, tool_rows)), total=len(tool_rows), desc="eval_tool_enabled", unit="row", dynamic_ncols=True):
#         tool_called, tool_query = parse_tool_call(tool_row.get("raw_response", ""))
#         if tool_called:
#             simulated_correct = 1
#             direct_eval = None
#         else:
#             direct_eval = evaluate_saved_row(BENCHMARK, tool_row, judge_runtime=None, judge_sampling=None)
#             simulated_correct = 0 if direct_eval.get("judge_label") is None else int(direct_eval.get("judge_label", 0))

#         merged = {
#             **tool_row,
#             "tool_called_by_model": int(tool_called),
#             "tool_query": tool_query,
#             "model_direct_eval": direct_eval,
#             "model_selftool_direct_correct": None if direct_eval is None else int(direct_eval.get("judge_label", 0) or 0),
#             "model_selftool_simulated_correct": int(simulated_correct),
#             "model_selftool_unnecessary_tool_call": int(bool(tool_called) and int(base_row.get("base_correct", 0)) == 1),
#             "model_selftool_potentially_necessary_tool_call": int(bool(tool_called) and int(base_row.get("base_correct", 0)) == 0),
#             "model_selftool_missed_incorrect_without_tool": int((not tool_called) and int(simulated_correct) == 0),
#             "normal_base_correct": int(base_row.get("base_correct", 0)),
#             "normal_aux_prob_correct": base_row.get("aux_prob_correct"),
#             "normal_base_has_final_answer": int(base_row.get("base_has_final_answer", False)),
#         }
#         out.append(merged)
#     return out


# # =============================================================================
# # THRESHOLD POLICY ANALYSIS
# # =============================================================================


# def compute_no_head_summary(base_rows: List[Dict[str, Any]], tool_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
#     n = len(base_rows)
#     num_tool_calls = sum(int(r.get("tool_called_by_model", 0)) for r in tool_rows)
#     num_unnecessary = sum(int(r.get("model_selftool_unnecessary_tool_call", 0)) for r in tool_rows)
#     num_necessary = sum(int(r.get("model_selftool_potentially_necessary_tool_call", 0)) for r in tool_rows)
#     num_missed = sum(int(r.get("model_selftool_missed_incorrect_without_tool", 0)) for r in tool_rows)
#     score = mean_binary([int(r.get("model_selftool_simulated_correct", 0)) for r in tool_rows])
#     return {
#         "num_rows": n,
#         "score_no_head": score,
#         "web_call_rate_no_head": num_tool_calls / max(n, 1),
#         "num_model_tool_calls": num_tool_calls,
#         "num_model_unnecessary_tool_calls": num_unnecessary,
#         "num_model_potentially_necessary_tool_calls": num_necessary,
#         "num_model_missed_incorrect_without_tool": num_missed,
#     }



# def compute_threshold_summary(
#     threshold: float,
#     base_rows: List[Dict[str, Any]],
#     no_head_summary: Dict[str, Any],
#     call_tool_if_missing_final_answer: bool,
# ) -> Dict[str, Any]:
#     decisions: List[int] = []
#     final_corrects: List[int] = []
#     unnecessary_calls = 0
#     necessary_calls = 0
#     missed_without_tool = 0
#     missing_final_triggered = 0

#     per_example: List[Dict[str, Any]] = []
#     for row in base_rows:
#         aux_prob = row.get("aux_prob_correct")
#         if aux_prob is None:
#             raise RuntimeError("Aux threshold analysis requires aux_prob_correct, but aux scoring is missing.")

#         low_aux = float(aux_prob) < float(threshold)
#         missing_final = not bool(row.get("base_has_final_answer", False))
#         call_tool = bool(low_aux or (call_tool_if_missing_final_answer and missing_final))
#         final_correct = 1 if call_tool else int(row.get("base_correct", 0))

#         if call_tool and missing_final and not low_aux:
#             missing_final_triggered += 1
#         if call_tool and int(row.get("base_correct", 0)) == 1:
#             unnecessary_calls += 1
#         if call_tool and int(row.get("base_correct", 0)) == 0:
#             necessary_calls += 1
#         if (not call_tool) and int(row.get("base_correct", 0)) == 0:
#             missed_without_tool += 1

#         decisions.append(int(call_tool))
#         final_corrects.append(int(final_correct))
#         per_example.append({
#             "sample_idx": row.get("sample_idx"),
#             "id": row.get("id"),
#             "threshold": float(threshold),
#             "aux_prob_correct": float(aux_prob),
#             "base_correct": int(row.get("base_correct", 0)),
#             "base_has_final_answer": int(row.get("base_has_final_answer", False)),
#             "call_tool_by_aux": int(call_tool),
#             "final_correct_after_aux_policy": int(final_correct),
#             "tool_call_unnecessary_by_aux": int(call_tool and int(row.get("base_correct", 0)) == 1),
#             "missed_incorrect_without_tool_by_aux": int((not call_tool) and int(row.get("base_correct", 0)) == 0),
#             "trigger_reason": "missing_final_or_low_aux" if (call_tool and missing_final and low_aux) else (
#                 "missing_final" if (call_tool and missing_final and not low_aux) else (
#                     "low_aux" if call_tool else "accept"
#                 )
#             ),
#         })

#     n = len(base_rows)
#     score = mean_binary(final_corrects)
#     web_call_rate = mean_binary(decisions)
#     score_no_head = float(no_head_summary.get("score_no_head", 0.0))
#     summary = {
#         "threshold": float(threshold),
#         "num_rows": n,
#         "score_ours": score,
#         "web_call_rate_ours": web_call_rate,
#         "num_aux_tool_calls": int(sum(decisions)),
#         "num_aux_unnecessary_tool_calls": int(unnecessary_calls),
#         "num_aux_potentially_necessary_tool_calls": int(necessary_calls),
#         "num_aux_missed_incorrect_without_tool": int(missed_without_tool),
#         "num_aux_missing_final_triggered_calls": int(missing_final_triggered),
#         "delta_score_vs_no_head": float(score - score_no_head),
#         "delta_web_call_rate_vs_no_head": float(web_call_rate - float(no_head_summary.get("web_call_rate_no_head", 0.0))),
#         "per_example": per_example,
#     }
#     return summary



# def select_best_threshold(threshold_summaries: List[Dict[str, Any]], no_head_summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
#     if not threshold_summaries:
#         return None
#     target_score = float(no_head_summary.get("score_no_head", 0.0))
#     feasible = [s for s in threshold_summaries if float(s.get("score_ours", 0.0)) >= target_score]
#     pool = feasible if feasible else threshold_summaries
#     pool = sorted(
#         pool,
#         key=lambda s: (
#             float(s.get("web_call_rate_ours", 1.0)),
#             -float(s.get("score_ours", 0.0)),
#             float(s.get("threshold", 0.0)),
#         ),
#     )
#     chosen = dict(pool[0])
#     chosen["selection_rule"] = (
#         "min_web_call_rate_subject_to_score_at_least_no_head"
#         if feasible else
#         "fallback_min_web_call_rate_then_max_score"
#     )
#     return chosen


# # =============================================================================
# # REPORT WRITERS
# # =============================================================================


# def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
#     ensure_dir(path.parent)
#     if not rows:
#         path.write_text("", encoding="utf-8")
#         return
#     headers = list(rows[0].keys())
#     with open(path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=headers)
#         writer.writeheader()
#         for row in rows:
#             writer.writerow({k: row.get(k) for k in headers})



# def make_markdown_summary(
#     args: argparse.Namespace,
#     aux_metrics: Optional[Dict[str, Any]],
#     no_head_summary: Dict[str, Any],
#     threshold_summaries: List[Dict[str, Any]],
#     chosen: Optional[Dict[str, Any]],
# ) -> str:
#     lines: List[str] = []
#     lines.append(f"# TriviaQA web-overuse summary for `{args.model_name_or_path}`")
#     lines.append("")
#     lines.append(f"- model_family: `{args.model_family}`")
#     lines.append(f"- thinking_mode: `{args.thinking_mode}`")
#     lines.append(f"- aux_head_ckpt: `{args.aux_head_ckpt or '<none>'}`")
#     lines.append(f"- num_rows: `{no_head_summary.get('num_rows', 0)}`")
#     lines.append(f"- thresholds: `{', '.join(threshold_tag(x) for x in parse_thresholds(args.thresholds))}`")
#     lines.append("")

#     if aux_metrics is not None:
#         lines.append("## Aux-head quality on the standard no-tool run")
#         lines.append("")
#         lines.append("| Metric | Value |")
#         lines.append("|---|---:|")
#         lines.append(f"| AUROC | {format_metric(aux_metrics.get('auroc'))} |")
#         lines.append(f"| Average Precision | {format_metric(aux_metrics.get('average_precision'))} |")
#         lines.append(f"| ECE | {format_metric(aux_metrics.get('ece'))} |")
#         lines.append("")

#     lines.append("## No-head model self-tool-use baseline")
#     lines.append("")
#     lines.append("| Metric | Value |")
#     lines.append("|---|---:|")
#     lines.append(f"| Score (No Head) | {format_metric(no_head_summary.get('score_no_head'))} |")
#     lines.append(f"| Web Call Rate (No Head) | {format_metric(no_head_summary.get('web_call_rate_no_head'))} |")
#     lines.append(f"| Model tool calls | {safe_int(no_head_summary.get('num_model_tool_calls'))} |")
#     lines.append(f"| Model unnecessary tool calls | {safe_int(no_head_summary.get('num_model_unnecessary_tool_calls'))} |")
#     lines.append(f"| Model potentially necessary tool calls | {safe_int(no_head_summary.get('num_model_potentially_necessary_tool_calls'))} |")
#     lines.append(f"| Model missed incorrect without tool | {safe_int(no_head_summary.get('num_model_missed_incorrect_without_tool'))} |")
#     lines.append("")

#     lines.append("## Threshold sweep for aux-triggered tool use")
#     lines.append("")
#     lines.append("| Threshold | Score (Ours) | Web Call Rate (Ours) | Aux tool calls | Unnecessary aux calls | Missed incorrect without tool | Δ Score vs No Head | Δ Web Call Rate vs No Head |")
#     lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
#     for s in threshold_summaries:
#         lines.append(
#             "| {thr} | {score} | {rate} | {calls} | {unnec} | {missed} | {dscore} | {drate} |".format(
#                 thr=format_metric(s.get("threshold")),
#                 score=format_metric(s.get("score_ours")),
#                 rate=format_metric(s.get("web_call_rate_ours")),
#                 calls=safe_int(s.get("num_aux_tool_calls")),
#                 unnec=safe_int(s.get("num_aux_unnecessary_tool_calls")),
#                 missed=safe_int(s.get("num_aux_missed_incorrect_without_tool")),
#                 dscore=format_signed_metric(s.get("delta_score_vs_no_head")),
#                 drate=format_signed_metric(s.get("delta_web_call_rate_vs_no_head")),
#             )
#         )
#     lines.append("")

#     if chosen is not None:
#         lines.append("## Suggested single threshold for the paper table")
#         lines.append("")
#         lines.append(f"Selection rule: `{chosen.get('selection_rule')}`")
#         lines.append("")
#         lines.append("| Threshold | Score (Ours) | Web Call Rate (Ours) | Aux unnecessary tool calls | Aux missed incorrect without tool |")
#         lines.append("|---:|---:|---:|---:|---:|")
#         lines.append(
#             "| {thr} | {score} | {rate} | {unnec} | {missed} |".format(
#                 thr=format_metric(chosen.get("threshold")),
#                 score=format_metric(chosen.get("score_ours")),
#                 rate=format_metric(chosen.get("web_call_rate_ours")),
#                 unnec=safe_int(chosen.get("num_aux_unnecessary_tool_calls")),
#                 missed=safe_int(chosen.get("num_aux_missed_incorrect_without_tool")),
#             )
#         )
#         lines.append("")

#     return "\n".join(lines) + "\n"



# def format_signed_metric(x: Any) -> str:
#     if x is None:
#         return "--"
#     try:
#         v = float(x)
#         return f"{v:+.4f}"
#     except Exception:
#         return str(x)


# def format_metric(x: Any) -> str:
#     if x is None:
#         return "--"
#     try:
#         return f"{float(x):.4f}"
#     except Exception:
#         return str(x)



# def build_paper_table_row(model_name_or_path: str, no_head_summary: Dict[str, Any], chosen: Optional[Dict[str, Any]]) -> Dict[str, Any]:
#     return {
#         "backbone": model_name_or_path,
#         "score_no_head": None if no_head_summary is None else float(no_head_summary.get("score_no_head", 0.0)),
#         "web_call_rate_no_head": None if no_head_summary is None else float(no_head_summary.get("web_call_rate_no_head", 0.0)),
#         "score_ours": None if chosen is None else float(chosen.get("score_ours", 0.0)),
#         "web_call_rate_ours": None if chosen is None else float(chosen.get("web_call_rate_ours", 0.0)),
#         "chosen_threshold": None if chosen is None else float(chosen.get("threshold", 0.0)),
#     }


# def build_latex_table_row(table_row: Dict[str, Any]) -> str:
#     def fmt(x: Any) -> str:
#         if x is None:
#             return "--"
#         return f"{float(x):.4f}"
#     return (
#         f"{table_row.get('backbone', '--')} & {fmt(table_row.get('score_no_head'))} & "
#         f"{fmt(table_row.get('web_call_rate_no_head'))} & {fmt(table_row.get('score_ours'))} & "
#         f"{fmt(table_row.get('web_call_rate_ours'))} \\"
#     )


# # =============================================================================
# # MAIN
# # =============================================================================


# def main() -> None:
#     args = parse_args()
#     thresholds = parse_thresholds(args.thresholds)

#     out_dir = Path(args.output_dir).expanduser().resolve()
#     ensure_dir(out_dir)

#     dataset_cfg = {
#         "data_mode": "hf",
#         "dataset_name": args.dataset_name,
#         "dataset_config_name": args.dataset_config_name,
#         "split": args.split,
#         "max_samples": int(args.max_samples),
#     }

#     json_dump(
#         out_dir / "run_config.json",
#         {
#             "benchmark": BENCHMARK,
#             "model_name_or_path": args.model_name_or_path,
#             "model_family": args.model_family,
#             "thinking_mode": args.thinking_mode,
#             "aux_head_ckpt": args.aux_head_ckpt,
#             "dataset_cfg": dataset_cfg,
#             "thresholds": thresholds,
#             "batch_size": int(args.batch_size),
#             "call_tool_if_missing_final_answer": bool(args.call_tool_if_missing_final_answer),
#         },
#     )

#     examples = load_examples_for_benchmark(BENCHMARK, dataset_cfg)
#     json_dump(out_dir / "dataset_summary.json", {"benchmark": BENCHMARK, "num_examples": len(examples)})

#     bundle = build_model_bundle_single(args)
#     orchestrator = MultiAgentOrchestrator(
#         benchmark=BENCHMARK,
#         model_bundles={"model1": bundle},
#         debug_mode=bool(args.debug),
#         debug_max_chars=2000,
#     )

#     if str(args.aux_head_ckpt or "").strip():
#         aux_runtime_check = orchestrator.get_aux("model1")
#         if aux_runtime_check is None:
#             raise RuntimeError(
#                 "Aux head checkpoint was provided, but the orchestrator did not create an aux runtime for model1."
#             )

#     try:
#         standard_rows_raw = run_generation_pass(
#             examples=examples,
#             orchestrator=orchestrator,
#             prompt_variant="standard_no_tool",
#             batch_size=int(args.batch_size),
#             debug=bool(args.debug),
#         )
#         standard_rows = score_standard_rows_with_aux(
#             examples=examples,
#             rows=standard_rows_raw,
#             orchestrator=orchestrator,
#             debug=bool(args.debug),
#         )

#         tool_rows_raw = run_generation_pass(
#             examples=examples,
#             orchestrator=orchestrator,
#             prompt_variant="tool_enabled",
#             batch_size=int(args.batch_size),
#             debug=bool(args.debug),
#         )
#         tool_rows = evaluate_tool_enabled_rows(base_rows=standard_rows, tool_rows=tool_rows_raw)
#     finally:
#         orchestrator.unload_all(drop_processors=False)
#         del orchestrator
#         gc.collect()

#     y_true = [int(r.get("base_correct", 0)) for r in standard_rows]
#     y_prob = [float(r.get("aux_prob_correct")) for r in standard_rows if r.get("aux_prob_correct") is not None]
#     aux_metrics = compute_aux_binary_metrics(y_true, y_prob) if len(y_true) == len(y_prob) and y_prob else None

#     no_head_summary = compute_no_head_summary(base_rows=standard_rows, tool_rows=tool_rows)
#     if not args.aux_head_ckpt:
#         raise RuntimeError(
#             "This experiment needs --aux_head_ckpt so it can simulate the aux-threshold web-call policy."
#         )
#     threshold_summaries = [
#         compute_threshold_summary(
#             threshold=thr,
#             base_rows=standard_rows,
#             no_head_summary=no_head_summary,
#             call_tool_if_missing_final_answer=bool(args.call_tool_if_missing_final_answer),
#         )
#         for thr in thresholds
#     ]
#     chosen = select_best_threshold(threshold_summaries, no_head_summary)

#     # Flatten per-threshold summaries for CSV/JSON export.
#     threshold_rows_flat: List[Dict[str, Any]] = []
#     threshold_decision_rows: List[Dict[str, Any]] = []
#     for s in threshold_summaries:
#         threshold_rows_flat.append({k: v for k, v in s.items() if k != "per_example"})
#         threshold_decision_rows.extend(list(s.get("per_example", [])))

#     merged_rows: List[Dict[str, Any]] = []
#     for base_row, tool_row in zip(standard_rows, tool_rows):
#         merged_rows.append({
#             "sample_idx": base_row.get("sample_idx"),
#             "id": base_row.get("id"),
#             "question": base_row.get("question"),
#             "gold_answer": base_row.get("gold_answer"),
#             "standard_raw_response": base_row.get("raw_response"),
#             "standard_correct": int(base_row.get("base_correct", 0)),
#             "standard_has_final_answer": int(base_row.get("base_has_final_answer", False)),
#             "standard_pred_ans_preview": base_row.get("pred_ans_preview"),
#             "aux_prob_correct": base_row.get("aux_prob_correct"),
#             "tool_enabled_raw_response": tool_row.get("raw_response"),
#             "tool_called_by_model": int(tool_row.get("tool_called_by_model", 0)),
#             "tool_query": tool_row.get("tool_query"),
#             "model_selftool_simulated_correct": int(tool_row.get("model_selftool_simulated_correct", 0)),
#             "model_selftool_unnecessary_tool_call": int(tool_row.get("model_selftool_unnecessary_tool_call", 0)),
#             "model_selftool_missed_incorrect_without_tool": int(tool_row.get("model_selftool_missed_incorrect_without_tool", 0)),
#         })

#     summary_payload = {
#         "benchmark": BENCHMARK,
#         "model_name_or_path": args.model_name_or_path,
#         "model_family": args.model_family,
#         "thinking_mode": args.thinking_mode,
#         "aux_head_ckpt": args.aux_head_ckpt,
#         "num_rows": len(examples),
#         "aux_metrics": aux_metrics,
#         "no_head_summary": no_head_summary,
#         "threshold_summaries": threshold_rows_flat,
#         "chosen_threshold_summary": chosen,
#     }

#     write_jsonl(out_dir / "standard_no_tool_rows.jsonl", standard_rows)
#     write_jsonl(out_dir / "tool_enabled_rows.jsonl", tool_rows)
#     write_jsonl(out_dir / "merged_per_example_rows.jsonl", merged_rows)
#     write_jsonl(out_dir / "threshold_decisions_per_example.jsonl", threshold_decision_rows)

#     save_rows_to_parquet(out_dir / "standard_no_tool_rows.parquet", standard_rows, debug=bool(args.debug))
#     save_rows_to_parquet(out_dir / "tool_enabled_rows.parquet", tool_rows, debug=bool(args.debug))
#     save_rows_to_parquet(out_dir / "merged_per_example_rows.parquet", merged_rows, debug=bool(args.debug))
#     save_rows_to_parquet(out_dir / "threshold_decisions_per_example.parquet", threshold_decision_rows, debug=bool(args.debug))

#     write_csv(out_dir / "threshold_summary.csv", threshold_rows_flat)
#     write_csv(out_dir / "merged_per_example_rows.csv", merged_rows)
#     write_csv(out_dir / "threshold_decisions_per_example.csv", threshold_decision_rows)

#     json_dump(out_dir / "summary.json", summary_payload)

#     paper_table_row = build_paper_table_row(args.model_name_or_path, no_head_summary, chosen)
#     json_dump(out_dir / "paper_table_row.json", paper_table_row)
#     write_csv(out_dir / "paper_table_row.csv", [paper_table_row])
#     (out_dir / "paper_table_row.tex").write_text(build_latex_table_row(paper_table_row) + "\n", encoding="utf-8")

#     md = make_markdown_summary(
#         args=args,
#         aux_metrics=aux_metrics,
#         no_head_summary=no_head_summary,
#         threshold_summaries=threshold_rows_flat,
#         chosen=chosen,
#     )
#     (out_dir / "summary.md").write_text(md, encoding="utf-8")

#     print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm.auto import tqdm
from transformers import AutoProcessor, AutoTokenizer

from compact_multi_agent_shared_optimized_v4_textbench import (
    MultiAgentOrchestrator,
    TokenUsage,
    build_generation_row,
    build_model_bundle,
    debug_print,
    evaluate_saved_row,
    get_example_image_for_benchmark,
    json_dump,
    load_examples_for_benchmark,
    response_final_answer_status,
    write_jsonl,
)


# =============================================================================
# DEFAULTS
# =============================================================================

BENCHMARK = "triviaqa"

DEFAULT_DATASET_CFG = {
    "data_mode": "hf",
    "dataset_name": "mandarjoshi/trivia_qa",
    "dataset_config_name": "rc",
    "split": "validation",
    "max_samples": 1000,
}

DEFAULT_SAMPLING_PROFILES = {
    "default": {
        "greedy": False,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "max_new_tokens": 15000,
    },
    "thinking": {
        "greedy": False,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "max_new_tokens": 16000,
    },
    "instruct": {
        "greedy": False,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
        "max_new_tokens": 15000,
    },
}

DEFAULT_AUX_PROFILE = {
    "trust_remote_code": True,
    "prefer_unsloth_mirror": False,
    "dtype": "bf16",
    "max_seq_len": 32768,
    "max_pixels": 200000,
    "attn_implementation": "flash_attention_3",
    "regression_threshold": 0.6,
    "head_input_mode": "completion_text_only",
    "hidden_encoder_type": "lite",
    "hidden_layer_selection": "last",
    "hidden_layer_index": None,
    "hidden_layer_indices": None,
}

DEFAULT_VLLM_RUNTIME = {
    "dtype": "bfloat16",
    "max_model_len": 32768,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.90,
    "max_num_seqs": 128,
    "enforce_eager": False,
    "trust_remote_code": False,
    "limit_mm_images": 1,
}

DEFAULT_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

TOOL_CALL_PATTERN = re.compile(r"<\s*web_search\s*>(.*?)<\s*/\s*web_search\s*>", flags=re.IGNORECASE | re.DOTALL)


# =============================================================================
# ARGPARSE / GENERAL HELPERS
# =============================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="TriviaQA tool-overuse evaluator with optional aux-head gating.")
    ap.add_argument("--model_name_or_path", type=str, required=True)
    ap.add_argument("--model_family", type=str, default="auto", choices=["auto", "qwen3_5", "qwen3", "qwen3_vl", "gemma4", "other"])
    ap.add_argument("--thinking_mode", type=str, default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--aux_head_ckpt", type=str, default="")
    ap.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_CFG["dataset_name"])
    ap.add_argument("--dataset_config_name", type=str, default=DEFAULT_DATASET_CFG["dataset_config_name"])
    ap.add_argument("--split", type=str, default=DEFAULT_DATASET_CFG["split"])
    ap.add_argument("--max_samples", type=int, default=DEFAULT_DATASET_CFG["max_samples"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--thresholds", type=str, default=",".join(str(x) for x in DEFAULT_THRESHOLDS))
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--call_tool_if_missing_final_answer", action="store_true", default=True)
    ap.add_argument("--no_call_tool_if_missing_final_answer", dest="call_tool_if_missing_final_answer", action="store_false")
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()


def parse_thresholds(csv_text: str) -> List[float]:
    vals: List[float] = []
    for part in str(csv_text or "").split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise RuntimeError("No thresholds were provided.")
    vals = sorted(set(vals))
    return vals


def sanitize_name_for_path(text: str) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "model"


def threshold_tag(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def chunked(seq: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(seq), batch_size):
        yield seq[start : start + batch_size]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _benchmark_is_multimodal(benchmark: str) -> bool:
    return str(benchmark or "").strip().lower() in {
        "mathvista", "mathverse", "charxiv_reasoning", "screenspot_pro", "simplevqa"
    }


def _benchmark_is_reasoning_text(benchmark: str) -> bool:
    return str(benchmark or "").strip().lower() in {"math", "mmlu_pro"}


def _resolve_thinking_enabled_local(model_name_or_path: str, model_family: str, thinking_mode: str) -> bool:
    mode = str(thinking_mode or "auto").strip().lower()
    if mode in {"on", "true", "1", "yes"}:
        return True
    if mode in {"off", "false", "0", "no"}:
        return False
    lname = str(model_name_or_path or "").lower()
    family = str(model_family or "auto").lower()
    if family == "qwen3_5":
        if any(x in lname for x in ["qwen3.5-0.8b", "qwen3.5-2b"]):
            return False
        return True
    if family == "qwen3_vl":
        return "thinking" in lname
    if family == "qwen3":
        return "instruct" not in lname or "thinking" in lname
    if family == "gemma4":
        return False
    return "thinking" in lname


def _official_sampling_override_for_model(*, model_name_or_path: str, model_family: str, thinking_mode: str, benchmark: str) -> Dict[str, Any]:
    family = str(model_family or "auto").strip().lower()
    if family == "auto":
        lname = str(model_name_or_path or "").lower()
        if "qwen3.5" in lname:
            family = "qwen3_5"
        elif "qwen3-vl" in lname:
            family = "qwen3_vl"
        elif "gemma-4" in lname:
            family = "gemma4"
        elif "qwen3" in lname:
            family = "qwen3"
        else:
            family = "other"

    thinking_enabled = _resolve_thinking_enabled_local(model_name_or_path, family, thinking_mode)
    multimodal = _benchmark_is_multimodal(benchmark)
    text_reasoning = _benchmark_is_reasoning_text(benchmark)

    if family == "gemma4":
        return {
            "greedy": False,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "max_new_tokens": 16000,
        }

    if family == "qwen3_5":
        if multimodal:
            if thinking_enabled:
                return {
                    "greedy": False,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "repetition_penalty": 1.0,
                    "presence_penalty": 1.5,
                    "max_new_tokens": 16000,
                }
            return {
                "greedy": False,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.0,
                "presence_penalty": 1.5,
                "max_new_tokens": 15000,
            }
        if text_reasoning:
            if thinking_enabled:
                return {
                    "greedy": False,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "repetition_penalty": 1.0,
                    "presence_penalty": 0.0,
                    "max_new_tokens": 16000,
                }
            return {
                "greedy": False,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.0,
                "presence_penalty": 0.0,
                "max_new_tokens": 15000,
            }

    return {}


def _apply_runtime_override(base: Dict[str, Any]) -> Dict[str, Any]:
    return dict(base)


def _parquet_safe_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False)
    return value


def save_rows_to_parquet(path: Path, rows: Iterable[Dict[str, Any]], debug: bool = False) -> bool:
    rows = list(rows)
    try:
        import pandas as pd

        flat_rows = [{k: _parquet_safe_value(v) for k, v in row.items()} for row in rows]
        df = pd.DataFrame(flat_rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        debug_print(debug, f"saved {path}")
        return True
    except Exception as e:
        err_path = path.with_suffix(path.suffix + ".error.txt")
        err_path.write_text(f"Parquet export failed: {type(e).__name__}: {e}\n", encoding="utf-8")
        debug_print(debug, f"Parquet export failed for {path}: {type(e).__name__}: {e}")
        return False



def release_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


# =============================================================================
# LOCAL PROMPT-BUILDER PATCHING
# =============================================================================


def _remove_gemma_think_prefix_local(text: str) -> str:
    text = text or ""
    return re.sub(r"^\s*<\|think\|>\s*\n?", "", text, count=1)


def _patch_gemma_messages_local(messages: List[Dict[str, Any]], thinking_enabled: bool) -> List[Dict[str, Any]]:
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

    system_content = _remove_gemma_think_prefix_local(system_content)
    if thinking_enabled:
        system_content = "<|think|>\n" + system_content if system_content else "<|think|>"
    patched[0]["content"] = system_content
    return patched


def _patch_chat_template_callable_local(bound_callable, model_family: str, thinking_enabled: bool):
    def patched(messages, *args, **kwargs):
        if model_family == "gemma4":
            messages = _patch_gemma_messages_local(messages, thinking_enabled)
        elif model_family in {"qwen3_5", "qwen3"}:
            kwargs = dict(kwargs)
            kwargs.setdefault("enable_thinking", thinking_enabled)
        return bound_callable(messages, *args, **kwargs)
    return patched


def _patch_processor_for_runtime_prompting_local(processor_like, model_family: str, thinking_enabled: bool):
    if hasattr(processor_like, "apply_chat_template"):
        processor_like.apply_chat_template = _patch_chat_template_callable_local(
            processor_like.apply_chat_template, model_family, thinking_enabled
        )
    tokenizer = getattr(processor_like, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        tokenizer.apply_chat_template = _patch_chat_template_callable_local(
            tokenizer.apply_chat_template, model_family, thinking_enabled
        )
    return processor_like


def ensure_runtime_prompt_builder(runtime) -> None:
    if getattr(runtime, "_processor", None) is not None:
        return
    cfg = getattr(runtime, "cfg", None)
    model_name_or_path = getattr(cfg, "model_name_or_path", "")
    trust_remote_code = bool(getattr(cfg, "trust_remote_code", False))
    model_family = str(getattr(cfg, "model_family", "auto") or "auto").strip().lower()
    if model_family == "auto":
        lname = str(model_name_or_path or "").lower()
        if "qwen3.5" in lname:
            model_family = "qwen3_5"
        elif "qwen3-vl" in lname:
            model_family = "qwen3_vl"
        elif "gemma-4" in lname or "gemma4" in lname:
            model_family = "gemma4"
        elif "qwen3" in lname:
            model_family = "qwen3"
        else:
            model_family = "other"
    thinking_enabled = _resolve_thinking_enabled_local(
        model_name_or_path=str(model_name_or_path),
        model_family=model_family,
        thinking_mode=str(getattr(cfg, "thinking_mode", "auto")),
    )

    processor_like = None
    load_errors = []

    # First choice: ask vLLM for the tokenizer it is already using.
    # This stays closest to your original working generation path and avoids
    # re-instantiating a separate HF tokenizer/processor inside this script.
    try:
        llm = runtime.llm
        for getter_name in ("get_tokenizer", "get_tokenizer_group"):
            getter = getattr(llm, getter_name, None)
            if getter is None:
                continue
            try:
                candidate = getter()
                if getter_name == "get_tokenizer_group":
                    candidate = getattr(candidate, "tokenizer", None) or getattr(candidate, "tokenizer_obj", None)
                if candidate is not None and hasattr(candidate, "apply_chat_template"):
                    processor_like = candidate
                    break
            except Exception as e:
                load_errors.append(f"vllm.{getter_name}: {type(e).__name__}: {e}")
        if processor_like is None:
            tok = getattr(llm, "get_tokenizer", lambda: None)()
            if tok is not None and hasattr(tok, "apply_chat_template"):
                processor_like = tok
    except Exception as e:
        load_errors.append(f"runtime.llm tokenizer bootstrap: {type(e).__name__}: {e}")
        processor_like = None

    if processor_like is None and model_family != "gemma4":
        try:
            processor_like = AutoProcessor.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
            )
        except Exception as e:
            load_errors.append(f"AutoProcessor: {type(e).__name__}: {e}")
            processor_like = None

    if processor_like is None and model_family == "gemma4":
        # Only as a last resort for Gemma.
        for kwargs in (
            {"use_fast": False},
            {"use_fast": True},
            {},
        ):
            try:
                processor_like = AutoTokenizer.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=trust_remote_code,
                    **kwargs,
                )
                break
            except Exception as e:
                load_errors.append(f"AutoTokenizer kwargs={kwargs}: {type(e).__name__}: {e}")
                processor_like = None

    if processor_like is None:
        raise RuntimeError(
            "Failed to initialize prompt builder. " + " | ".join(load_errors)
        )

    processor_like = _patch_processor_for_runtime_prompting_local(
        processor_like,
        model_family=model_family,
        thinking_enabled=thinking_enabled,
    )
    runtime._processor = processor_like



# =============================================================================
# TOOL PROMPTING / PARSING
# =============================================================================


def build_tool_prompt_text(ex: Dict[str, Any]) -> str:
    question = str(ex.get("question") or ex.get("prompt_text") or "").strip()
    return (
        f"{question}\n\n"
        "You may optionally use a web search tool named WEB_SEARCH. "
        "Use the tool only when you are genuinely uncertain and need external verification. "
        "Do not use the tool if you already know the answer with high confidence.\n\n"
        "If you want to use the tool, output exactly one line and nothing else in this format:\n"
        "<web_search>your search query</web_search>\n\n"
        "Otherwise, answer directly and put your final answer only once at the end inside \\boxed{...}."
    )



def build_tool_messages(ex: Dict[str, Any]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    if get_example_image_for_benchmark(ex) is not None:
        content.append({"type": "image"})
    content.append({"type": "text", "text": build_tool_prompt_text(ex)})
    return [{"role": "user", "content": content}]



def parse_tool_call(text: str) -> Tuple[bool, Optional[str]]:
    raw = str(text or "")
    m = TOOL_CALL_PATTERN.search(raw)
    if m:
        query = re.sub(r"\s+", " ", m.group(1)).strip()
        return True, (query or None)

    stripped = raw.strip()
    upper = stripped.upper()
    if upper.startswith("WEB_SEARCH:"):
        query = stripped.split(":", 1)[1].strip()
        return True, (query or None)
    return False, None


# =============================================================================
# METRICS
# =============================================================================


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default



def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default



def mean_binary(vals: Sequence[int]) -> float:
    return float(sum(int(v) for v in vals)) / max(len(vals), 1)



def compute_ece(y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 10) -> Optional[float]:
    if not y_true or not y_prob or len(y_true) != len(y_prob):
        return None
    total = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        if i == n_bins - 1:
            idx = [j for j, p in enumerate(y_prob) if lo <= p <= hi]
        else:
            idx = [j for j, p in enumerate(y_prob) if lo <= p < hi]
        if not idx:
            continue
        acc = sum(int(y_true[j]) for j in idx) / len(idx)
        conf = sum(float(y_prob[j]) for j in idx) / len(idx)
        ece += (len(idx) / total) * abs(acc - conf)
    return float(ece)



def compute_aux_binary_metrics(y_true: Sequence[int], y_prob: Sequence[float]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "num_rows": len(y_true),
        "auroc": None,
        "average_precision": None,
        "ece": compute_ece(y_true, y_prob),
    }
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        if len(set(int(x) for x in y_true)) >= 2:
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        if sum(int(x) for x in y_true) > 0:
            metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        pass
    return metrics


# =============================================================================
# GENERATION + EVAL HELPERS
# =============================================================================


def build_model_bundle_single(args: argparse.Namespace, *, vllm_gpu_memory_utilization: Optional[float] = None):
    sampling_override = _official_sampling_override_for_model(
        model_name_or_path=args.model_name_or_path,
        model_family=args.model_family,
        thinking_mode=args.thinking_mode,
        benchmark=BENCHMARK,
    )
    runtime_profile = {
        **DEFAULT_VLLM_RUNTIME,
        "model_family": args.model_family,
        "thinking_mode": args.thinking_mode,
    }
    if vllm_gpu_memory_utilization is not None:
        runtime_profile["gpu_memory_utilization"] = float(vllm_gpu_memory_utilization)
    runtime_profile = _apply_runtime_override(runtime_profile)
    aux_profile = {
        **DEFAULT_AUX_PROFILE,
        "model_family": args.model_family,
        "thinking_mode": args.thinking_mode,
    }
    return build_model_bundle(
        model_name_or_path=args.model_name_or_path,
        aux_head_ckpt=args.aux_head_ckpt,
        runtime_profile=runtime_profile,
        sampling_profiles=DEFAULT_SAMPLING_PROFILES,
        aux_profile=aux_profile,
        sampling_override=sampling_override,
        model_family=args.model_family,
        thinking_mode=args.thinking_mode,
    )



def build_empty_usage(orchestrator: MultiAgentOrchestrator, prompt_tokens: int, completion_tokens: int, generation_time_sec: float) -> Dict[str, TokenUsage]:
    usage = orchestrator._new_usage_by_model()
    usage["model1"].prompt_tokens += int(prompt_tokens)
    usage["model1"].completion_tokens += int(completion_tokens)
    usage["model1"].generation_calls += 1
    usage["model1"].generation_time_sec += float(generation_time_sec)
    return usage



def run_generation_pass(
    examples: List[Dict[str, Any]],
    orchestrator: MultiAgentOrchestrator,
    prompt_variant: str,
    batch_size: int,
    debug: bool,
) -> List[Dict[str, Any]]:
    runtime = orchestrator.get_generator("model1")
    ensure_runtime_prompt_builder(runtime)
    bundle = orchestrator.model_bundles["model1"]
    rows: List[Dict[str, Any]] = []

    desc = "generate_standard" if prompt_variant == "standard_no_tool" else "generate_tool_enabled"
    for batch in tqdm(chunked(examples, batch_size), total=math.ceil(len(examples) / max(batch_size, 1)), desc=desc, unit="batch", dynamic_ncols=True):
        messages_list: List[List[Dict[str, Any]]] = []
        images: List[Optional[Any]] = []
        batch_examples: List[Dict[str, Any]] = []
        for ex in batch:
            if prompt_variant == "standard_no_tool":
                messages = orchestrator._build_messages_for_turn(ex, None)
            elif prompt_variant == "tool_enabled":
                messages = build_tool_messages(ex)
            else:
                raise RuntimeError(f"Unknown prompt_variant={prompt_variant}")
            messages_list.append(messages)
            images.append(get_example_image_for_benchmark(ex))
            batch_examples.append(ex)

        gens = runtime.generate_batch(
            messages_list=messages_list,
            images=images,
            sampling_cfg=bundle.sampling_cfg,
            continue_final_messages=[False] * len(batch_examples),
        )
        if len(gens) != len(batch_examples):
            raise RuntimeError(f"Expected {len(batch_examples)} generations, got {len(gens)}")

        for ex, messages, gen in zip(batch_examples, messages_list, gens):
            usage = build_empty_usage(orchestrator, prompt_tokens=gen.prompt_tokens, completion_tokens=gen.completion_tokens, generation_time_sec=float(gen.generation_time_sec))
            trace = [{"event": "generation", "model": "model1", "prompt_variant": prompt_variant}]
            row = build_generation_row(
                benchmark=BENCHMARK,
                ex=ex,
                strategy_name=prompt_variant,
                final_model_name="model1",
                final_response=gen.text,
                usage_by_model=usage,
                trace=trace,
                wall_time_sec=float(gen.generation_time_sec),
            )
            row["messages"] = messages
            row["prompt_variant"] = prompt_variant
            rows.append(row)
    debug_print(debug, f"Generated {len(rows)} rows for prompt_variant={prompt_variant}")
    return rows



def score_standard_rows_with_aux(
    examples: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    orchestrator: MultiAgentOrchestrator,
    debug: bool,
    require_aux: bool = False,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    aux_runtime = orchestrator.get_aux("model1")
    has_aux = aux_runtime is not None
    if require_aux and not has_aux:
        raise RuntimeError(
            "Aux head was requested, but orchestrator.get_aux('model1') returned None. "
            "Check that --aux_head_ckpt is set and that build_model_bundle(...) received it."
        )
    if has_aux and aux_runtime is not None:
        aux_runtime.load()

    for ex, row in tqdm(list(zip(examples, rows)), total=len(rows), desc="score_aux_standard", unit="row", dynamic_ncols=True):
        evaluated = evaluate_saved_row(BENCHMARK, row, judge_runtime=None, judge_sampling=None)
        info = response_final_answer_status(BENCHMARK, row.get("raw_response", ""))
        aux_prob: Optional[float] = None
        aux_pred: Optional[int] = None
        if has_aux:
            score = orchestrator._score_response(ex, "model1", str(row.get("raw_response") or ""))
            aux_prob = float(score.prob_correct)
            aux_pred = int(score.pred)
            evaluated.setdefault("usage_by_model", row.get("usage_by_model", {}))
            try:
                evaluated["usage_by_model"]["model1"]["aux_calls"] = safe_int(evaluated["usage_by_model"]["model1"].get("aux_calls", 0)) + 1
                evaluated["usage_by_model"]["model1"]["aux_scored_tokens"] = safe_int(
                    evaluated["usage_by_model"]["model1"].get("aux_scored_tokens", 0)
                ) + safe_int(evaluated.get("usage_by_model", {}).get("model1", {}).get("completion_tokens", 0))
            except Exception:
                pass
        merged = {
            **evaluated,
            "base_has_final_answer": bool(info.get("has_final_answer", False)),
            "base_final_answer_reason": str(info.get("reason", "unknown")),
            "aux_enabled_for_run": bool(has_aux),
            "aux_prob_correct": aux_prob,
            "aux_pred": aux_pred,
            "base_correct": 0 if evaluated.get("judge_label") is None else int(evaluated.get("judge_label", 0)),
        }
        out.append(merged)
    debug_print(debug, f"Scored {len(out)} standard rows with aux")
    return out



def evaluate_tool_enabled_rows(
    base_rows: List[Dict[str, Any]],
    tool_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if len(base_rows) != len(tool_rows):
        raise RuntimeError(f"Length mismatch: base_rows={len(base_rows)} tool_rows={len(tool_rows)}")

    out: List[Dict[str, Any]] = []
    for base_row, tool_row in tqdm(list(zip(base_rows, tool_rows)), total=len(tool_rows), desc="eval_tool_enabled", unit="row", dynamic_ncols=True):
        tool_called, tool_query = parse_tool_call(tool_row.get("raw_response", ""))
        if tool_called:
            simulated_correct = 1
            direct_eval = None
        else:
            direct_eval = evaluate_saved_row(BENCHMARK, tool_row, judge_runtime=None, judge_sampling=None)
            simulated_correct = 0 if direct_eval.get("judge_label") is None else int(direct_eval.get("judge_label", 0))

        merged = {
            **tool_row,
            "tool_called_by_model": int(tool_called),
            "tool_query": tool_query,
            "model_direct_eval": direct_eval,
            "model_selftool_direct_correct": None if direct_eval is None else int(direct_eval.get("judge_label", 0) or 0),
            "model_selftool_simulated_correct": int(simulated_correct),
            "model_selftool_unnecessary_tool_call": int(bool(tool_called) and int(base_row.get("base_correct", 0)) == 1),
            "model_selftool_potentially_necessary_tool_call": int(bool(tool_called) and int(base_row.get("base_correct", 0)) == 0),
            "model_selftool_missed_incorrect_without_tool": int((not tool_called) and int(simulated_correct) == 0),
            "normal_base_correct": int(base_row.get("base_correct", 0)),
            "normal_aux_prob_correct": base_row.get("aux_prob_correct"),
            "normal_base_has_final_answer": int(base_row.get("base_has_final_answer", False)),
        }
        out.append(merged)
    return out


# =============================================================================
# THRESHOLD POLICY ANALYSIS
# =============================================================================


def compute_no_head_summary(base_rows: List[Dict[str, Any]], tool_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(base_rows)
    num_tool_calls = sum(int(r.get("tool_called_by_model", 0)) for r in tool_rows)
    num_unnecessary = sum(int(r.get("model_selftool_unnecessary_tool_call", 0)) for r in tool_rows)
    num_necessary = sum(int(r.get("model_selftool_potentially_necessary_tool_call", 0)) for r in tool_rows)
    num_missed = sum(int(r.get("model_selftool_missed_incorrect_without_tool", 0)) for r in tool_rows)
    score = mean_binary([int(r.get("model_selftool_simulated_correct", 0)) for r in tool_rows])
    return {
        "num_rows": n,
        "score_no_head": score,
        "web_call_rate_no_head": num_tool_calls / max(n, 1),
        "num_model_tool_calls": num_tool_calls,
        "num_model_unnecessary_tool_calls": num_unnecessary,
        "num_model_potentially_necessary_tool_calls": num_necessary,
        "num_model_missed_incorrect_without_tool": num_missed,
    }



def compute_threshold_summary(
    threshold: float,
    base_rows: List[Dict[str, Any]],
    no_head_summary: Dict[str, Any],
    call_tool_if_missing_final_answer: bool,
) -> Dict[str, Any]:
    decisions: List[int] = []
    final_corrects: List[int] = []
    unnecessary_calls = 0
    necessary_calls = 0
    missed_without_tool = 0
    missing_final_triggered = 0

    per_example: List[Dict[str, Any]] = []
    for row in base_rows:
        aux_prob = row.get("aux_prob_correct")
        if aux_prob is None:
            raise RuntimeError("Aux threshold analysis requires aux_prob_correct, but aux scoring is missing.")

        low_aux = float(aux_prob) < float(threshold)
        missing_final = not bool(row.get("base_has_final_answer", False))
        call_tool = bool(low_aux or (call_tool_if_missing_final_answer and missing_final))
        final_correct = 1 if call_tool else int(row.get("base_correct", 0))

        if call_tool and missing_final and not low_aux:
            missing_final_triggered += 1
        if call_tool and int(row.get("base_correct", 0)) == 1:
            unnecessary_calls += 1
        if call_tool and int(row.get("base_correct", 0)) == 0:
            necessary_calls += 1
        if (not call_tool) and int(row.get("base_correct", 0)) == 0:
            missed_without_tool += 1

        decisions.append(int(call_tool))
        final_corrects.append(int(final_correct))
        per_example.append({
            "sample_idx": row.get("sample_idx"),
            "id": row.get("id"),
            "threshold": float(threshold),
            "aux_prob_correct": float(aux_prob),
            "base_correct": int(row.get("base_correct", 0)),
            "base_has_final_answer": int(row.get("base_has_final_answer", False)),
            "call_tool_by_aux": int(call_tool),
            "final_correct_after_aux_policy": int(final_correct),
            "tool_call_unnecessary_by_aux": int(call_tool and int(row.get("base_correct", 0)) == 1),
            "missed_incorrect_without_tool_by_aux": int((not call_tool) and int(row.get("base_correct", 0)) == 0),
            "trigger_reason": "missing_final_or_low_aux" if (call_tool and missing_final and low_aux) else (
                "missing_final" if (call_tool and missing_final and not low_aux) else (
                    "low_aux" if call_tool else "accept"
                )
            ),
        })

    n = len(base_rows)
    score = mean_binary(final_corrects)
    web_call_rate = mean_binary(decisions)
    score_no_head = float(no_head_summary.get("score_no_head", 0.0))
    summary = {
        "threshold": float(threshold),
        "num_rows": n,
        "score_ours": score,
        "web_call_rate_ours": web_call_rate,
        "num_aux_tool_calls": int(sum(decisions)),
        "num_aux_unnecessary_tool_calls": int(unnecessary_calls),
        "num_aux_potentially_necessary_tool_calls": int(necessary_calls),
        "num_aux_missed_incorrect_without_tool": int(missed_without_tool),
        "num_aux_missing_final_triggered_calls": int(missing_final_triggered),
        "delta_score_vs_no_head": float(score - score_no_head),
        "delta_web_call_rate_vs_no_head": float(web_call_rate - float(no_head_summary.get("web_call_rate_no_head", 0.0))),
        "per_example": per_example,
    }
    return summary



def select_best_threshold(threshold_summaries: List[Dict[str, Any]], no_head_summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not threshold_summaries:
        return None
    target_score = float(no_head_summary.get("score_no_head", 0.0))
    feasible = [s for s in threshold_summaries if float(s.get("score_ours", 0.0)) >= target_score]
    pool = feasible if feasible else threshold_summaries
    pool = sorted(
        pool,
        key=lambda s: (
            float(s.get("web_call_rate_ours", 1.0)),
            -float(s.get("score_ours", 0.0)),
            float(s.get("threshold", 0.0)),
        ),
    )
    chosen = dict(pool[0])
    chosen["selection_rule"] = (
        "min_web_call_rate_subject_to_score_at_least_no_head"
        if feasible else
        "fallback_min_web_call_rate_then_max_score"
    )
    return chosen


# =============================================================================
# REPORT WRITERS
# =============================================================================


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in headers})



def make_markdown_summary(
    args: argparse.Namespace,
    aux_metrics: Optional[Dict[str, Any]],
    no_head_summary: Dict[str, Any],
    threshold_summaries: List[Dict[str, Any]],
    chosen: Optional[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append(f"# TriviaQA web-overuse summary for `{args.model_name_or_path}`")
    lines.append("")
    lines.append(f"- model_family: `{args.model_family}`")
    lines.append(f"- thinking_mode: `{args.thinking_mode}`")
    lines.append(f"- aux_head_ckpt: `{args.aux_head_ckpt or '<none>'}`")
    lines.append(f"- num_rows: `{no_head_summary.get('num_rows', 0)}`")
    lines.append(f"- thresholds: `{', '.join(threshold_tag(x) for x in parse_thresholds(args.thresholds))}`")
    lines.append("")

    if aux_metrics is not None:
        lines.append("## Aux-head quality on the standard no-tool run")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        lines.append(f"| AUROC | {format_metric(aux_metrics.get('auroc'))} |")
        lines.append(f"| Average Precision | {format_metric(aux_metrics.get('average_precision'))} |")
        lines.append(f"| ECE | {format_metric(aux_metrics.get('ece'))} |")
        lines.append("")

    lines.append("## No-head model self-tool-use baseline")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Score (No Head) | {format_metric(no_head_summary.get('score_no_head'))} |")
    lines.append(f"| Web Call Rate (No Head) | {format_metric(no_head_summary.get('web_call_rate_no_head'))} |")
    lines.append(f"| Model tool calls | {safe_int(no_head_summary.get('num_model_tool_calls'))} |")
    lines.append(f"| Model unnecessary tool calls | {safe_int(no_head_summary.get('num_model_unnecessary_tool_calls'))} |")
    lines.append(f"| Model potentially necessary tool calls | {safe_int(no_head_summary.get('num_model_potentially_necessary_tool_calls'))} |")
    lines.append(f"| Model missed incorrect without tool | {safe_int(no_head_summary.get('num_model_missed_incorrect_without_tool'))} |")
    lines.append("")

    lines.append("## Threshold sweep for aux-triggered tool use")
    lines.append("")
    lines.append("| Threshold | Score (Ours) | Web Call Rate (Ours) | Aux tool calls | Unnecessary aux calls | Missed incorrect without tool | Δ Score vs No Head | Δ Web Call Rate vs No Head |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in threshold_summaries:
        lines.append(
            "| {thr} | {score} | {rate} | {calls} | {unnec} | {missed} | {dscore} | {drate} |".format(
                thr=format_metric(s.get("threshold")),
                score=format_metric(s.get("score_ours")),
                rate=format_metric(s.get("web_call_rate_ours")),
                calls=safe_int(s.get("num_aux_tool_calls")),
                unnec=safe_int(s.get("num_aux_unnecessary_tool_calls")),
                missed=safe_int(s.get("num_aux_missed_incorrect_without_tool")),
                dscore=format_signed_metric(s.get("delta_score_vs_no_head")),
                drate=format_signed_metric(s.get("delta_web_call_rate_vs_no_head")),
            )
        )
    lines.append("")

    if chosen is not None:
        lines.append("## Suggested single threshold for the paper table")
        lines.append("")
        lines.append(f"Selection rule: `{chosen.get('selection_rule')}`")
        lines.append("")
        lines.append("| Threshold | Score (Ours) | Web Call Rate (Ours) | Aux unnecessary tool calls | Aux missed incorrect without tool |")
        lines.append("|---:|---:|---:|---:|---:|")
        lines.append(
            "| {thr} | {score} | {rate} | {unnec} | {missed} |".format(
                thr=format_metric(chosen.get("threshold")),
                score=format_metric(chosen.get("score_ours")),
                rate=format_metric(chosen.get("web_call_rate_ours")),
                unnec=safe_int(chosen.get("num_aux_unnecessary_tool_calls")),
                missed=safe_int(chosen.get("num_aux_missed_incorrect_without_tool")),
            )
        )
        lines.append("")

    return "\n".join(lines) + "\n"



def format_signed_metric(x: Any) -> str:
    if x is None:
        return "--"
    try:
        v = float(x)
        return f"{v:+.4f}"
    except Exception:
        return str(x)


def format_metric(x: Any) -> str:
    if x is None:
        return "--"
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)



def build_paper_table_row(model_name_or_path: str, no_head_summary: Dict[str, Any], chosen: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "backbone": model_name_or_path,
        "score_no_head": None if no_head_summary is None else float(no_head_summary.get("score_no_head", 0.0)),
        "web_call_rate_no_head": None if no_head_summary is None else float(no_head_summary.get("web_call_rate_no_head", 0.0)),
        "score_ours": None if chosen is None else float(chosen.get("score_ours", 0.0)),
        "web_call_rate_ours": None if chosen is None else float(chosen.get("web_call_rate_ours", 0.0)),
        "chosen_threshold": None if chosen is None else float(chosen.get("threshold", 0.0)),
    }


def build_latex_table_row(table_row: Dict[str, Any]) -> str:
    def fmt(x: Any) -> str:
        if x is None:
            return "--"
        return f"{float(x):.4f}"
    return (
        f"{table_row.get('backbone', '--')} & {fmt(table_row.get('score_no_head'))} & "
        f"{fmt(table_row.get('web_call_rate_no_head'))} & {fmt(table_row.get('score_ours'))} & "
        f"{fmt(table_row.get('web_call_rate_ours'))} \\"
    )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)

    out_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(out_dir)

    dataset_cfg = {
        "data_mode": "hf",
        "dataset_name": args.dataset_name,
        "dataset_config_name": args.dataset_config_name,
        "split": args.split,
        "max_samples": int(args.max_samples),
    }

    json_dump(
        out_dir / "run_config.json",
        {
            "benchmark": BENCHMARK,
            "model_name_or_path": args.model_name_or_path,
            "model_family": args.model_family,
            "thinking_mode": args.thinking_mode,
            "aux_head_ckpt": args.aux_head_ckpt,
            "dataset_cfg": dataset_cfg,
            "thresholds": thresholds,
            "batch_size": int(args.batch_size),
            "call_tool_if_missing_final_answer": bool(args.call_tool_if_missing_final_answer),
        },
    )

    examples = load_examples_for_benchmark(BENCHMARK, dataset_cfg)
    json_dump(out_dir / "dataset_summary.json", {"benchmark": BENCHMARK, "num_examples": len(examples)})

    generation_bundle = build_model_bundle_single(args, vllm_gpu_memory_utilization=0.90)
    generation_orchestrator = MultiAgentOrchestrator(
        benchmark=BENCHMARK,
        model_bundles={"model1": generation_bundle},
        debug_mode=bool(args.debug),
        debug_max_chars=2000,
    )

    try:
        standard_rows_raw = run_generation_pass(
            examples=examples,
            orchestrator=generation_orchestrator,
            prompt_variant="standard_no_tool",
            batch_size=int(args.batch_size),
            debug=bool(args.debug),
        )
        tool_rows_raw = run_generation_pass(
            examples=examples,
            orchestrator=generation_orchestrator,
            prompt_variant="tool_enabled",
            batch_size=int(args.batch_size),
            debug=bool(args.debug),
        )

        write_jsonl(out_dir / "standard_no_tool_rows_raw_generation.jsonl", standard_rows_raw)
        write_jsonl(out_dir / "tool_enabled_rows_raw_generation.jsonl", tool_rows_raw)
        save_rows_to_parquet(out_dir / "standard_no_tool_rows_raw_generation.parquet", standard_rows_raw, debug=bool(args.debug))
        save_rows_to_parquet(out_dir / "tool_enabled_rows_raw_generation.parquet", tool_rows_raw, debug=bool(args.debug))
    finally:
        generation_orchestrator.unload_all(drop_processors=False)
        del generation_orchestrator
        del generation_bundle
        release_memory()

    aux_bundle = build_model_bundle_single(args)
    aux_orchestrator = MultiAgentOrchestrator(
        benchmark=BENCHMARK,
        model_bundles={"model1": aux_bundle},
        debug_mode=bool(args.debug),
        debug_max_chars=2000,
    )

    if str(args.aux_head_ckpt or "").strip():
        aux_runtime_check = aux_orchestrator.get_aux("model1")
        if aux_runtime_check is None:
            raise RuntimeError(
                "Aux head checkpoint was provided, but the orchestrator did not create an aux runtime for model1."
            )

    try:
        standard_rows = score_standard_rows_with_aux(
            examples=examples,
            rows=standard_rows_raw,
            orchestrator=aux_orchestrator,
            debug=bool(args.debug),
        )
        tool_rows = evaluate_tool_enabled_rows(base_rows=standard_rows, tool_rows=tool_rows_raw)
    finally:
        aux_orchestrator.unload_all(drop_processors=False)
        del aux_orchestrator
        del aux_bundle
        release_memory()

    y_true = [int(r.get("base_correct", 0)) for r in standard_rows]
    y_prob = [float(r.get("aux_prob_correct")) for r in standard_rows if r.get("aux_prob_correct") is not None]
    aux_metrics = compute_aux_binary_metrics(y_true, y_prob) if len(y_true) == len(y_prob) and y_prob else None

    no_head_summary = compute_no_head_summary(base_rows=standard_rows, tool_rows=tool_rows)
    if not args.aux_head_ckpt:
        raise RuntimeError(
            "This experiment needs --aux_head_ckpt so it can simulate the aux-threshold web-call policy."
        )
    threshold_summaries = [
        compute_threshold_summary(
            threshold=thr,
            base_rows=standard_rows,
            no_head_summary=no_head_summary,
            call_tool_if_missing_final_answer=bool(args.call_tool_if_missing_final_answer),
        )
        for thr in thresholds
    ]
    chosen = select_best_threshold(threshold_summaries, no_head_summary)

    # Flatten per-threshold summaries for CSV/JSON export.
    threshold_rows_flat: List[Dict[str, Any]] = []
    threshold_decision_rows: List[Dict[str, Any]] = []
    for s in threshold_summaries:
        threshold_rows_flat.append({k: v for k, v in s.items() if k != "per_example"})
        threshold_decision_rows.extend(list(s.get("per_example", [])))

    merged_rows: List[Dict[str, Any]] = []
    for base_row, tool_row in zip(standard_rows, tool_rows):
        merged_rows.append({
            "sample_idx": base_row.get("sample_idx"),
            "id": base_row.get("id"),
            "question": base_row.get("question"),
            "gold_answer": base_row.get("gold_answer"),
            "standard_raw_response": base_row.get("raw_response"),
            "standard_correct": int(base_row.get("base_correct", 0)),
            "standard_has_final_answer": int(base_row.get("base_has_final_answer", False)),
            "standard_pred_ans_preview": base_row.get("pred_ans_preview"),
            "aux_prob_correct": base_row.get("aux_prob_correct"),
            "tool_enabled_raw_response": tool_row.get("raw_response"),
            "tool_called_by_model": int(tool_row.get("tool_called_by_model", 0)),
            "tool_query": tool_row.get("tool_query"),
            "model_selftool_simulated_correct": int(tool_row.get("model_selftool_simulated_correct", 0)),
            "model_selftool_unnecessary_tool_call": int(tool_row.get("model_selftool_unnecessary_tool_call", 0)),
            "model_selftool_missed_incorrect_without_tool": int(tool_row.get("model_selftool_missed_incorrect_without_tool", 0)),
        })

    summary_payload = {
        "benchmark": BENCHMARK,
        "model_name_or_path": args.model_name_or_path,
        "model_family": args.model_family,
        "thinking_mode": args.thinking_mode,
        "aux_head_ckpt": args.aux_head_ckpt,
        "num_rows": len(examples),
        "aux_metrics": aux_metrics,
        "no_head_summary": no_head_summary,
        "threshold_summaries": threshold_rows_flat,
        "chosen_threshold_summary": chosen,
    }

    write_jsonl(out_dir / "standard_no_tool_rows.jsonl", standard_rows)
    write_jsonl(out_dir / "tool_enabled_rows.jsonl", tool_rows)
    write_jsonl(out_dir / "merged_per_example_rows.jsonl", merged_rows)
    write_jsonl(out_dir / "threshold_decisions_per_example.jsonl", threshold_decision_rows)

    save_rows_to_parquet(out_dir / "standard_no_tool_rows.parquet", standard_rows, debug=bool(args.debug))
    save_rows_to_parquet(out_dir / "tool_enabled_rows.parquet", tool_rows, debug=bool(args.debug))
    save_rows_to_parquet(out_dir / "merged_per_example_rows.parquet", merged_rows, debug=bool(args.debug))
    save_rows_to_parquet(out_dir / "threshold_decisions_per_example.parquet", threshold_decision_rows, debug=bool(args.debug))

    write_csv(out_dir / "threshold_summary.csv", threshold_rows_flat)
    write_csv(out_dir / "merged_per_example_rows.csv", merged_rows)
    write_csv(out_dir / "threshold_decisions_per_example.csv", threshold_decision_rows)

    json_dump(out_dir / "summary.json", summary_payload)

    paper_table_row = build_paper_table_row(args.model_name_or_path, no_head_summary, chosen)
    json_dump(out_dir / "paper_table_row.json", paper_table_row)
    write_csv(out_dir / "paper_table_row.csv", [paper_table_row])
    (out_dir / "paper_table_row.tex").write_text(build_latex_table_row(paper_table_row) + "\n", encoding="utf-8")

    md = make_markdown_summary(
        args=args,
        aux_metrics=aux_metrics,
        no_head_summary=no_head_summary,
        threshold_summaries=threshold_rows_flat,
        chosen=chosen,
    )
    (out_dir / "summary.md").write_text(md, encoding="utf-8")

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
