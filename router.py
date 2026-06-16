

from typing import Optional

import config as _cfg
from config import (
    ModelConfig,
    MODEL_REGISTRY,
    ROUTING_KEYWORDS,
    ROUTE_CAPABILITY_MAP,
    MODELS_PER_ROUTE,
)


def classify_query(query: str) -> str:

    query_lower = query.lower()
    scores: dict = {category: 0 for category in ROUTING_KEYWORDS}

    for category, keywords in ROUTING_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                scores[category] += 1

    best = max(scores, key=lambda c: scores[c])
    return best if scores[best] > 0 else "general"


def select_models(
    query_type: str,
    available_model_names: list,
    force_all: bool = False,
) -> list:

    gpu_cfg = _cfg.get_gpu_cfg()


    candidates = [
        m for m in MODEL_REGISTRY
        if m.enabled
        and not m.is_judge
        and _is_available(m.name, available_model_names)
    ]

    if not candidates:
        return []

   
    if gpu_cfg.get("prefer_light_models"):
        candidates = sorted(candidates, key=lambda m: m.size_gb)

    if force_all:
        
        gpu_cap = gpu_cfg["max_panel_size"]
        return candidates[:gpu_cap]

  
    preferred_caps = set(ROUTE_CAPABILITY_MAP.get(query_type, ["general"]))
    route_max = MODELS_PER_ROUTE.get(query_type, 3)
    gpu_cap   = gpu_cfg["max_panel_size"]
    effective_max = min(route_max, gpu_cap)

    def score(m: ModelConfig) -> int:
        return len(set(m.capabilities) & preferred_caps)

   
    ranked = sorted(
        candidates,
        key=lambda m: (-score(m), m.size_gb if gpu_cfg.get("prefer_light_models") else 0),
    )
    selected = ranked[:effective_max]

   
    if not selected and candidates:
        selected = [candidates[0]]

    return selected


def get_judge(available_model_names: list) -> Optional[ModelConfig]:
    """
    Return the designated judge model if available and GPU mode allows it.
    Returns None if GPU mode has judge_enabled=False.
    Falls back to smallest available model if primary judge is missing.
    """
    gpu_cfg = _cfg.get_gpu_cfg()
    if not gpu_cfg.get("judge_enabled", True):
        return None

    for m in MODEL_REGISTRY:
        if m.is_judge and _is_available(m.name, available_model_names):
            return m

    # Fallback: smallest available enabled model
    available_models = [
        m for m in MODEL_REGISTRY
        if m.enabled and _is_available(m.name, available_model_names)
    ]
    if available_models:
        return min(available_models, key=lambda m: m.size_gb)

    return None


def _is_available(model_name: str, available: list) -> bool:
    if model_name in available:
        return True
    base = model_name.split(":")[0]
    return any(a.split(":")[0] == base for a in available)


def routing_summary(
    query: str,
    query_type: str,
    selected: list,
    judge: Optional[ModelConfig],
) -> str:
    gpu_cfg   = _cfg.get_gpu_cfg()
    panel_str = ", ".join(m.display_name for m in selected) or "none"
    judge_str = judge.display_name if judge else "disabled"
    return (
        f"  GPU mode   : {_cfg.GPU_MODE.upper()}  — {gpu_cfg['description']}\n"
        f"  Query type : {query_type.upper()}\n"
        f"  Panel      : {panel_str}\n"
        f"  Judge      : {judge_str}"
    )