
#!/usr/bin/env python3


"""
ensemble.py — ZeroTwo Local Multi-LLM Ensemble System
======================================================
Powered by Ollama · Runs 100% locally

Usage:
    python3 ensemble.py "your question"
    python3 ensemble.py "explain quicksort"   --gpu-mode light
    python3 ensemble.py "write a sort fn"     --gpu-mode balanced --type coding
    python3 ensemble.py "what is entropy?"    --gpu-mode aggressive --verbose
    python3 ensemble.py "fix this code"       --model codellama
    python3 ensemble.py "compare A and B"     --no-judge
    python3 ensemble.py --list-models

GPU modes:
    --gpu-mode light       1 model · no judge · minimum VRAM  (safest)
    --gpu-mode balanced    2-3 models · sequential · judge on  (default)
    --gpu-mode aggressive  full panel · parallel allowed · max perf

Other flags:
    --model MODEL    Query a single model only (short name or full name)
    --type TYPE      Force query type: coding|math|reasoning|creative|general
    --all-models     Skip smart routing; query every available model
    --no-judge       Skip judge synthesis; print individual responses
    --verbose        Show each model's response before the final answer
    --parallel       Enable parallel execution (overrides GPU mode default)
    --list-models    Show configured models and Ollama availability
"""

import sys
import argparse
import time
from typing import Optional

import config as _cfg
from config import (
    HARDWARE, MODEL_REGISTRY,
    GPU_MODE_CONFIGS, apply_gpu_mode, get_gpu_cfg,
)
from ollama_client import check_ollama_running, list_available_models
from router import classify_query, select_models, get_judge, routing_summary
from executor import run_panel, filter_successful
from judge import run_judge_streaming

try:
    import shutil
    _COLS = shutil.get_terminal_size().columns
except Exception:
    _COLS = 80

IS_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not IS_TTY:
        return text
    return f"\033[{code}m{text}\033[0m"


CYAN = lambda t: _c("96", t)
YELLOW = lambda t: _c("93", t)
GREEN = lambda t: _c("92", t)
RED = lambda t: _c("91", t)
BOLD = lambda t: _c("1", t)
DIM = lambda t: _c("2", t)
PINK = lambda t: _c("38;2;255;105;180", t)
LRED = lambda t: _c("38;2;220;50;50", t)
RULE = lambda: print(DIM("─" * min(_COLS, 80)))


_ZEROTWO_TEXT = r"""
    __                            
    | |                           
    | | ___ _ __ ___   ___  _ __  
    | |/ _ \ '_ ` _ \ / _ \| '_ \ 
    | |  __/ | | | | | (_) | | | |
    |_|\___|_| |_| |_|\___/|_| |_|

    """


def _banner() -> None:
    """Print the ZeroTwo banner in yellow."""
    for line in _ZEROTWO_TEXT.splitlines():
        print(YELLOW(line))
    print(YELLOW("  " + "═" * 68))
    print(YELLOW("    Local Multi-LLM Ensemble System  ·  Powered by Ollama  ·  100% Local"))
    print(YELLOW("  " + "═" * 68))
    print()


def _progress(msg: str, is_error: bool = False) -> None:
    if is_error:
        print(RED(f"  {msg}"))
    else:
        print(DIM(f"  {msg}"))


def cmd_list_models(available: list) -> None:
    print()
    print(BOLD("  Configured Models"))
    RULE()
    for m in MODEL_REGISTRY:
        found = any(
            m.name == a or m.name.split(":")[0] == a.split(":")[0]
            for a in available
        )
        status = GREEN("● available") if found else RED("○ not pulled")
        judge_tag = YELLOW(" [judge]") if m.is_judge else ""
        caps = DIM(", ".join(m.capabilities))
        short = DIM(f"--model {m.name.split(':')[0]}")
        vram = DIM(f"~{m.size_gb}GB VRAM")
        print(f"  {status}  {BOLD(m.display_name)}{judge_tag}  {vram}")
        print(f"           {DIM(m.name)}  |  {caps}")
        print(f"           {short}")
    print()
    extra = [
        a for a in available
        if not any(m.name.split(":")[0] == a.split(":")[0] for m in MODEL_REGISTRY)
    ]
    if extra:
        print(DIM(f"  Other pulled (not in registry): {', '.join(extra)}"))
    print()
    print(BOLD("  GPU Modes"))
    RULE()
    for name, cfg in GPU_MODE_CONFIGS.items():
        marker = YELLOW("▶ ") if name == _cfg.GPU_MODE else "  "
        print(f"{marker}{BOLD(name):20s}  {cfg['description']}")
    print()


def _resolve_model(name_hint: str, available: list):
    hint_base = name_hint.split(":")[0].lower()

    for m in MODEL_REGISTRY:
        if (
            m.name.lower() == name_hint.lower()
            or m.name.split(":")[0].lower() == hint_base
            or m.display_name.lower() == name_hint.lower()
        ):
            ollama_name = next(
                (a for a in available
                 if a.split(":")[0].lower() == m.name.split(":")[0].lower()),
                None,
            )
            return m, ollama_name

    ollama_name = next(
        (a for a in available if a.split(":")[0].lower() == hint_base),
        None,
    )
    if ollama_name:
        from config import ModelConfig
        mc = ModelConfig(
            name=ollama_name,
            display_name=ollama_name,
            capabilities=["general"],
        )
        return mc, ollama_name

    return None, None


def run_single(question: str, model_hint: str) -> str:
    from ollama_client import stream_model as _stream

    if not check_ollama_running():
        print(RED("\n  ✗ Ollama is not running. Start it with: ollama serve\n"))
        return ""

    available = list_available_models()
    if not available:
        print(RED("\n  ✗ No models found in Ollama.\n"))
        return ""

    model_cfg, ollama_name = _resolve_model(model_hint, available)
    if model_cfg is None or ollama_name is None:
        print(RED(f"\n  ✗ Model '{model_hint}' not found in Ollama.\n"))
        print(DIM("  Run --list-models to see what is available.\n"))
        return ""

    gpu_cfg = get_gpu_cfg()
    ctx_cap = gpu_cfg["num_ctx_cap"]
    pred_cap = gpu_cfg["num_predict_cap"]

    print(BOLD("\n  Single-Model Query"))
    RULE()
    print(DIM(f"  Model     : {model_cfg.display_name}  ({ollama_name})"))
    print(
        DIM(f"  GPU mode  : {_cfg.GPU_MODE}  (ctx={min(model_cfg.num_ctx, ctx_cap)}, predict={min(model_cfg.num_predict, pred_cap)})"))
    print(DIM(f"  Prompt    : {question[:72]}{'…' if len(question) > 72 else ''}"))
    print()
    RULE()
    print()

    start = time.perf_counter()
    full_text: list = []
    try:
        for token in _stream(
                model_name=ollama_name,
                prompt=question,
                temperature=model_cfg.temperature,
                num_ctx=min(model_cfg.num_ctx, ctx_cap),
                num_predict=min(model_cfg.num_predict, pred_cap),
        ):
            print(token, end="", flush=True)
            full_text.append(token)
    except Exception as e:
        print(RED(f"\n  ✗ Stream error: {e}\n"))
        return ""

    elapsed = time.perf_counter() - start
    print()
    RULE()
    print(DIM(f"  Done in {elapsed:.1f}s"))
    print()
    return "".join(full_text).strip()


def run_ensemble(
        question: str,
        force_type: Optional[str] = None,
        all_models: bool = False,
        no_judge: bool = False,
        verbose: bool = False,
        parallel: bool = False,
) -> str:
    total_start = time.perf_counter()

    if parallel:
        HARDWARE["allow_parallel"] = True
        if _cfg.GPU_MODE != "aggressive":
            print(DIM(
                f"  ⚠  --parallel forced on in {_cfg.GPU_MODE} mode. "
                "Risk of VRAM overflow on 6GB card."
            ))

    if not check_ollama_running():
        print(RED("\n  ✗ Ollama is not running. Start it with: ollama serve\n"))
        return ""

    available = list_available_models()
    if not available:
        print(RED("\n  ✗ No models found in Ollama. Pull at least one model first.\n"))
        return ""

    query_type = force_type if force_type else classify_query(question)
    panel = select_models(query_type, available, force_all=all_models)
    judge = None if no_judge else get_judge(available)

    if not panel:
        print(RED("\n  ✗ No suitable models available for this query type.\n"))
        print(DIM("  Run --list-models to check which models are pulled.\n"))
        return ""

    print(BOLD("\n  Routing Decision"))
    RULE()
    print(routing_summary(question, query_type, panel, judge))
    print()
    mode_label = "parallel" if HARDWARE["allow_parallel"] else "sequential"
    print(DIM(f"  Execution : {mode_label}"))
    print()

    print(BOLD("  Querying Models"))
    RULE()
    responses = run_panel(panel, question, progress_cb=_progress)
    successful = filter_successful(responses)
    print()
    print(DIM(f"  {len(successful)}/{len(panel)} models responded successfully."))

    if not successful:
        print(RED("\n  ✗ All models failed. Check Ollama logs.\n"))
        return ""

    if verbose or no_judge:
        print()
        print(BOLD("  Individual Responses"))
        for r in successful:
            RULE()
            print(YELLOW(f"  [{r.display_name}]  ({r.elapsed_sec:.1f}s)"))
            print()
            for line in r.text.strip().splitlines():
                print(f"    {line}")
            print()

    if no_judge:
        elapsed = time.perf_counter() - total_start
        print(DIM(f"\n  Total time: {elapsed:.1f}s\n"))
        return "\n\n".join(f"[{r.display_name}]\n{r.text}" for r in successful)

    if judge is None:
        print(RED("\n  ✗ No judge available. Use --no-judge to skip synthesis.\n"))
        return ""

    judge_avail = any(
        judge.name == a or judge.name.split(":")[0] == a.split(":")[0]
        for a in available
    )
    if not judge_avail:
        print(RED(f"\n  ✗ Judge model '{judge.name}' is not pulled in Ollama.\n"))
        return ""

    print()
    RULE()
    print(BOLD("  Final Answer  ") + DIM(f"(synthesized by {judge.display_name})"))
    RULE()
    print()

    final_answer = run_judge_streaming(
        judge=judge,
        question=question,
        responses=successful,
        print_tokens=True,
    )

    elapsed = time.perf_counter() - total_start
    print()
    RULE()
    print(DIM(f"  Total time: {elapsed:.1f}s"))
    print()
    return final_answer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ensemble.py",
        description="ZeroTwo — Local Multi-LLM Ensemble System (Ollama)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "question",
        nargs="?",
        help="Prompt to send to the model(s)",
    )
    p.add_argument(
        "--gpu-mode",
        dest="gpu_mode",
        choices=["light", "balanced", "aggressive"],
        default="balanced",
        metavar="MODE",
        help="VRAM safety mode: light | balanced (default) | aggressive",
    )
    p.add_argument(
        "--model",
        metavar="MODEL",
        default=None,
        help="Query ONE model only (short name e.g. deepseek-r1, codellama)",
    )
    p.add_argument(
        "--type",
        dest="force_type",
        choices=["coding", "math", "reasoning", "creative", "structured", "general"],
        default=None,
        help="Override automatic query-type detection",
    )
    p.add_argument(
        "--all-models",
        action="store_true",
        help="Query every available model (still respects GPU panel cap)",
    )
    p.add_argument(
        "--no-judge",
        action="store_true",
        help="Print individual responses without judge synthesis",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show each model response before the final answer",
    )
    p.add_argument(
        "--parallel",
        action="store_true",
        help="Force parallel execution (overrides GPU mode default; risky on 6GB)",
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="Show configured models, Ollama availability, and GPU modes, then exit",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    apply_gpu_mode(args.gpu_mode)
    _banner()

    if args.list_models:
        if not check_ollama_running():
            print(RED("  ✗ Ollama is not running. Start it with: ollama serve\n"))
            sys.exit(1)
        cmd_list_models(list_available_models())
        sys.exit(0)

    if not args.question:
        parser.print_help()
        print()
        sys.exit(1)

    if args.model:
        result = run_single(question=args.question, model_hint=args.model)
        sys.exit(0 if result else 1)

    result = run_ensemble(
        question=args.question,
        force_type=args.force_type,
        all_models=args.all_models,
        no_judge=args.no_judge,
        verbose=args.verbose,
        parallel=args.parallel,
    )
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
