#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def _sanitize_name_for_path(x: str) -> str:
    s = str(x).strip().lower()
    s = s.replace("/", "__").replace("\\", "__")
    s = s.replace(" ", "_").replace(".", "p").replace("-", "_")
    while "___" in s:
        s = s.replace("___", "__")
    return s


def _format_threshold_for_path(x: float) -> str:
    return f"{float(x):.2f}".replace(".", "p")


def _runtime_mode_tag(model_family: str, thinking_mode: str) -> str:
    fam = _sanitize_name_for_path(model_family or "auto")
    think = _sanitize_name_for_path(thinking_mode or "auto")
    return f"fam_{fam}__think_{think}"


def _parse_threshold_list(csv: str) -> List[float]:
    vals: List[float] = []
    for part in str(csv).split(","):
        s = part.strip()
        if not s:
            continue
        vals.append(float(s))
    if not vals:
        raise RuntimeError("No thresholds provided")
    return sorted(set(vals))


def _compute_results_root(
    *,
    output_base_root: str,
    benchmark: str,
    model1_name_or_path: str,
    model2_name_or_path: str,
    model1_model_family: str,
    model1_thinking_mode: str,
    model2_model_family: str,
    model2_thinking_mode: str,
    model1_aux_thresholds: List[float],
    model2_aux_threshold: float,
) -> Path:
    model1_tag = _sanitize_name_for_path(model1_name_or_path)
    model2_tag = _sanitize_name_for_path(model2_name_or_path)
    model1_mode_tag = _runtime_mode_tag(model1_model_family, model1_thinking_mode)
    model2_mode_tag = _runtime_mode_tag(model2_model_family, model2_thinking_mode)
    thr_list_tag = "_".join(_format_threshold_for_path(x) for x in model1_aux_thresholds)
    thr2_tag = _format_threshold_for_path(model2_aux_threshold)
    run_tag = f"{model1_tag}__{model1_mode_tag}__{model2_tag}__{model2_mode_tag}__thr1s_{thr_list_tag}__thr2_{thr2_tag}_lite"
    return Path(output_base_root) / run_tag / f"{benchmark}_split"


def _replace_constant(src: str, name: str, value_literal: str) -> str:
    pattern = re.compile(rf"(?m)^{re.escape(name)}\s*=\s*(?:.+(?:\n(?![A-Z_][A-Z0-9_]*\s*=).+)*)")
    repl = f"{name} = {value_literal}"
    out, count = pattern.subn(repl, src, count=1)
    if count != 1:
        raise RuntimeError(f"Could not patch constant {name!r}")
    return out


def _make_temp_generate_script(
    *,
    generate_script: Path,
    benchmark: str,
    strategy_names: str,
    model1_name_or_path: str,
    model2_name_or_path: str,
    model1_model_family: str,
    model1_thinking_mode: str,
    model2_model_family: str,
    model2_thinking_mode: str,
    model1_aux_head_ckpt: str,
    model1_aux_thresholds_csv: str,
    model2_aux_threshold: float,
    output_base_root: str,
) -> Path:
    src = generate_script.read_text(encoding="utf-8")
    src = _replace_constant(src, "BENCHMARK", repr(benchmark))
    src = _replace_constant(src, "STRATEGY_NAMES", repr(strategy_names))
    src = _replace_constant(src, "MODEL1_NAME_OR_PATH", repr(model1_name_or_path))
    src = _replace_constant(src, "MODEL2_NAME_OR_PATH", repr(model2_name_or_path))
    src = _replace_constant(src, "MODEL1_MODEL_FAMILY", repr(model1_model_family))
    src = _replace_constant(src, "MODEL1_THINKING_MODE", repr(model1_thinking_mode))
    src = _replace_constant(src, "MODEL2_MODEL_FAMILY", repr(model2_model_family))
    src = _replace_constant(src, "MODEL2_THINKING_MODE", repr(model2_thinking_mode))
    src = _replace_constant(src, "MODEL1_AUX_HEAD_CKPT", repr(model1_aux_head_ckpt))
    src = _replace_constant(src, "MODEL1_AUX_THRESHOLDS", repr(_parse_threshold_list(model1_aux_thresholds_csv)))
    src = _replace_constant(src, "MODEL2_AUX_THRESHOLD", repr(float(model2_aux_threshold)))
    src = _replace_constant(src, "OUTPUT_BASE_ROOT", repr(output_base_root))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    tmp_path = generate_script.with_name(f"{generate_script.stem}__tmp_run_{stamp}.py")
    tmp_path.write_text(src, encoding="utf-8")
    return tmp_path


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Run compact multi-agent generation, then evaluate the results end to end.")
    ap.add_argument("--benchmark", type=str, default="simplevqa", choices=["mathvista", "mathverse", "charxiv_reasoning", "screenspot_pro", "simplevqa", "triviaqa", "math", "mmlu_pro"])
    ap.add_argument("--model1_name_or_path", type=str, required=True)
    ap.add_argument("--model2_name_or_path", type=str, required=True)
    ap.add_argument("--model1_aux_head_ckpt", type=str, required=True)
    ap.add_argument("--model1_model_family", type=str, default="auto", choices=["auto", "qwen3_5", "qwen3", "qwen3_vl", "gemma4", "other"])
    ap.add_argument("--model1_thinking_mode", type=str, default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--model2_model_family", type=str, default="auto", choices=["auto", "qwen3_5", "qwen3", "qwen3_vl", "gemma4", "other"])
    ap.add_argument("--model2_thinking_mode", type=str, default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--model1_params", type=float, default=0.0)
    ap.add_argument("--model2_params", type=float, default=0.0)
    ap.add_argument("--strategy_names", type=str, required=True, help="Comma-separated strategy names")
    ap.add_argument("--model1_aux_thresholds", type=str, default="0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--model2_aux_threshold", type=float, default=0.80)
    ap.add_argument(
        "--output_base_root",
        type=str,
        default="eval_outputs/multi_agent_compact",
    )
    ap.add_argument(
        "--generate_script",
        type=str,
        default=str(here / "compact_multi_agent_generate.py"),
    )
    ap.add_argument(
        "--evaluate_script",
        type=str,
        default=str(here / "compact_multi_agent_evaluate.py"),
    )
    ap.add_argument("--judge_batch_size", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--keep_temp_generate_script", action="store_true")
    return ap.parse_args()


def _run(cmd: List[str], *, cwd: Path) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    args = parse_args()

    launch_cwd = Path.cwd().resolve()
    generate_script = Path(args.generate_script).expanduser().resolve()
    evaluate_script = Path(args.evaluate_script).expanduser().resolve()
    output_base_root = str(Path(args.output_base_root).expanduser().resolve())
    model1_aux_head_ckpt = str(Path(args.model1_aux_head_ckpt).expanduser().resolve())
    if not generate_script.exists():
        raise RuntimeError(f"Generate script not found: {generate_script}")
    if not evaluate_script.exists():
        raise RuntimeError(f"Evaluate script not found: {evaluate_script}")

    thresholds = _parse_threshold_list(args.model1_aux_thresholds)
    results_root = _compute_results_root(
        output_base_root=output_base_root,
        benchmark=args.benchmark,
        model1_name_or_path=args.model1_name_or_path,
        model2_name_or_path=args.model2_name_or_path,
        model1_model_family=args.model1_model_family,
        model1_thinking_mode=args.model1_thinking_mode,
        model2_model_family=args.model2_model_family,
        model2_thinking_mode=args.model2_thinking_mode,
        model1_aux_thresholds=thresholds,
        model2_aux_threshold=float(args.model2_aux_threshold),
    )

    temp_generate_script = _make_temp_generate_script(
        generate_script=generate_script,
        benchmark=args.benchmark,
        strategy_names=args.strategy_names,
        model1_name_or_path=args.model1_name_or_path,
        model2_name_or_path=args.model2_name_or_path,
        model1_model_family=args.model1_model_family,
        model1_thinking_mode=args.model1_thinking_mode,
        model2_model_family=args.model2_model_family,
        model2_thinking_mode=args.model2_thinking_mode,
        model1_aux_head_ckpt=model1_aux_head_ckpt,
        model1_aux_thresholds_csv=args.model1_aux_thresholds,
        model2_aux_threshold=float(args.model2_aux_threshold),
        output_base_root=output_base_root,
    )

    run_manifest = {
        "benchmark": args.benchmark,
        "model1_name_or_path": args.model1_name_or_path,
        "model2_name_or_path": args.model2_name_or_path,
        "model1_aux_head_ckpt": model1_aux_head_ckpt,
        "model1_model_family": args.model1_model_family,
        "model1_thinking_mode": args.model1_thinking_mode,
        "model2_model_family": args.model2_model_family,
        "model2_thinking_mode": args.model2_thinking_mode,
        "model1_params": float(args.model1_params),
        "model2_params": float(args.model2_params),
        "strategy_names": args.strategy_names,
        "model1_aux_thresholds": thresholds,
        "model2_aux_threshold": float(args.model2_aux_threshold),
        "output_base_root": output_base_root,
        "results_root": str(results_root),
        "generate_script": str(generate_script),
        "evaluate_script": str(evaluate_script),
        "temp_generate_script": str(temp_generate_script),
        "judge_batch_size": int(args.judge_batch_size),
        "overwrite": bool(args.overwrite),
        "resume_mode": not bool(args.no_resume),
        "debug": bool(args.debug),
    }

    results_root.mkdir(parents=True, exist_ok=True)
    manifest_path = results_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        gen_cmd = [
            sys.executable,
            str(temp_generate_script),
            "--benchmark",
            args.benchmark,
            "--strategy_names",
            args.strategy_names,
            "--model1_aux_thresholds",
            args.model1_aux_thresholds,
            "--model1_model_family",
            args.model1_model_family,
            "--model1_thinking_mode",
            args.model1_thinking_mode,
            "--model2_model_family",
            args.model2_model_family,
            "--model2_thinking_mode",
            args.model2_thinking_mode,
        ]
        if args.overwrite:
            gen_cmd.append("--overwrite")
        if args.no_resume:
            gen_cmd.append("--no_resume")
        if args.debug:
            gen_cmd.append("--debug")
        _run(gen_cmd, cwd=launch_cwd)

        eval_cmd = [
            sys.executable,
            str(evaluate_script),
            "--results_root",
            str(results_root),
            "--benchmark",
            args.benchmark,
            "--strategy_names",
            args.strategy_names,
            "--judge_batch_size",
            str(args.judge_batch_size),
            "--model1_params",
            str(args.model1_params),
            "--model2_params",
            str(args.model2_params),
        ]
        if args.debug:
            eval_cmd.append("--debug")
        _run(eval_cmd, cwd=launch_cwd)

        print(f"\n[DONE] Results root: {results_root}", flush=True)
        print(f"[DONE] Manifest: {manifest_path}", flush=True)
    finally:
        if not args.keep_temp_generate_script:
            try:
                if temp_generate_script.exists():
                    temp_generate_script.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    main()
