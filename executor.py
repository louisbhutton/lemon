

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import config as _cfg
from config import ModelConfig, HARDWARE
from ollama_client import ModelResponse, query_model


ProgressCallback = Callable[[str, bool], None]   # (message, is_error)


def run_panel(
    models: list,
    prompt: str,
    progress_cb: Optional[ProgressCallback] = None,
) -> list:

    if not models:
        return []

    if HARDWARE["allow_parallel"]:
        return _run_parallel(models, prompt, progress_cb)
    return _run_sequential(models, prompt, progress_cb)


def _run_sequential(
    models: list,
    prompt: str,
    progress_cb: Optional[ProgressCallback],
) -> list:
    results: list = []
    gpu_cfg = _cfg.get_gpu_cfg()
    delay   = HARDWARE["inter_model_delay_sec"]

    # Apply GPU-mode caps to each model's context/predict window
    ctx_cap     = gpu_cfg["num_ctx_cap"]
    predict_cap = gpu_cfg["num_predict_cap"]

    for i, model in enumerate(models):
        if progress_cb:
            progress_cb(f"Querying {model.display_name}...", False)

        response = query_model(
            model_name=model.name,
            display_name=model.display_name,
            prompt=prompt,
            temperature=model.temperature,
            top_p=model.top_p,
            num_ctx=min(model.num_ctx, ctx_cap),
            num_predict=min(model.num_predict, predict_cap),
        )
        results.append(response)

        if progress_cb:
            if response.success:
                progress_cb(
                    f"  ✓ {model.display_name} responded in {response.elapsed_sec:.1f}s",
                    False,
                )
            else:
                progress_cb(
                    f"  ✗ {model.display_name} failed: {response.error}",
                    True,
                )

        if i < len(models) - 1 and delay > 0:
            time.sleep(delay)

    return results


def _run_parallel(
    models: list,
    prompt: str,
    progress_cb: Optional[ProgressCallback],
) -> list:

    max_workers = min(HARDWARE.get("max_workers", 1), 2)  # hard safety cap
    gpu_cfg     = _cfg.get_gpu_cfg()
    ctx_cap     = gpu_cfg["num_ctx_cap"]
    predict_cap = gpu_cfg["num_predict_cap"]
    results: list = []
    lock = threading.Lock()

    def _query(model: ModelConfig) -> ModelResponse:
        return query_model(
            model_name=model.name,
            display_name=model.display_name,
            prompt=prompt,
            temperature=model.temperature,
            top_p=model.top_p,
            num_ctx=min(model.num_ctx, ctx_cap),
            num_predict=min(model.num_predict, predict_cap),
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_query, m): m for m in models}
        for future in as_completed(futures):
            try:
                response = future.result()
            except Exception as e:
                m = futures[future]
                response = ModelResponse(
                    model=m.name,
                    display_name=m.display_name,
                    text="",
                    elapsed_sec=0.0,
                    success=False,
                    error=str(e),
                )
            with lock:
                results.append(response)
            if progress_cb:
                if response.success:
                    progress_cb(
                        f"  ✓ {response.display_name} responded in {response.elapsed_sec:.1f}s",
                        False,
                    )
                else:
                    progress_cb(
                        f"  ✗ {response.display_name} failed: {response.error}",
                        True,
                    )

    return results


def filter_successful(responses: list) -> list:
    return [r for r in responses if r.success and r.text.strip()]


def format_responses_for_judge(responses: list) -> str:
    parts = []
    for i, r in enumerate(responses, 1):
        parts.append(f"--- Response {i} ({r.display_name}) ---\n{r.text.strip()}")
    return "\n\n".join(parts)
