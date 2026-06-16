

from typing import Optional

import config as _cfg
from config import ModelConfig, JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE
from ollama_client import ModelResponse, query_model, stream_model


def build_judge_prompt(question: str, responses: list) -> str:
    """Compose the full user-turn prompt the judge will receive."""
    formatted = []
    for i, r in enumerate(responses, 1):
        formatted.append(f"--- Response {i} ({r.display_name}) ---\n{r.text.strip()}")
    return JUDGE_USER_TEMPLATE.format(
        question=question,
        responses="\n\n".join(formatted),
    )


def run_judge_streaming(
    judge: ModelConfig,
    question: str,
    responses: list,
    print_tokens: bool = True,
) -> str:

    gpu_cfg  = _cfg.get_gpu_cfg()
    num_ctx  = gpu_cfg["judge_num_ctx"]
    num_pred = gpu_cfg["judge_num_predict"]

    prompt = build_judge_prompt(question, responses)
    full_text: list = []

    try:
        for token in stream_model(
            model_name=judge.name,
            prompt=prompt,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            temperature=0.4,           # low temp for accurate synthesis
            num_ctx=num_ctx,
            num_predict=num_pred,
        ):
            if print_tokens:
                print(token, end="", flush=True)
            full_text.append(token)
    except Exception as e:
        msg = f"\n[Judge streaming error: {e}]"
        if print_tokens:
            print(msg)
        return msg

    return "".join(full_text).strip()


def run_judge_blocking(
    judge: ModelConfig,
    question: str,
    responses: list,
) -> ModelResponse:

    gpu_cfg  = _cfg.get_gpu_cfg()
    num_ctx  = gpu_cfg["judge_num_ctx"]
    num_pred = gpu_cfg["judge_num_predict"]

    prompt = build_judge_prompt(question, responses)
    return query_model(
        model_name=judge.name,
        display_name=judge.display_name,
        prompt=prompt,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        temperature=0.4,
        num_ctx=num_ctx,
        num_predict=num_pred,
    )
