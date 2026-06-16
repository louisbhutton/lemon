"""
ollama_client.py — Low-level HTTP client for the Ollama local API.
Handles requests, timeouts, streaming, and error recovery.
"""

import json
import time
import requests
from typing import Optional, Generator
from dataclasses import dataclass

from config import OLLAMA_API_GENERATE, OLLAMA_API_TAGS, HARDWARE


@dataclass
class ModelResponse:
    model: str
    display_name: str
    text: str
    elapsed_sec: float
    success: bool
    error: Optional[str] = None

    def __str__(self) -> str:
        status = f"✓ {self.elapsed_sec:.1f}s" if self.success else f"✗ {self.error}"
        return f"[{self.display_name}] ({status})\n{self.text}"


def check_ollama_running() -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        resp = requests.get(OLLAMA_API_TAGS, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def list_available_models() -> list:
    """Return list of model name strings currently pulled in Ollama."""
    try:
        resp = requests.get(OLLAMA_API_TAGS, timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def query_model(
    model_name: str,
    display_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_ctx: int = 4096,
    num_predict: int = 1024,
    timeout: Optional[int] = None,
) -> ModelResponse:
    """Send a prompt to an Ollama model and return a ModelResponse. Non-streaming."""
    if timeout is None:
        timeout = HARDWARE["request_timeout_sec"]

    payload: dict = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }
    if system_prompt:
        payload["system"] = system_prompt

    start = time.perf_counter()
    try:
        resp = requests.post(OLLAMA_API_GENERATE, json=payload, timeout=timeout)
        resp.raise_for_status()
        data    = resp.json()
        text    = data.get("response", "").strip()
        elapsed = time.perf_counter() - start

        if not text:
            return ModelResponse(
                model=model_name, display_name=display_name,
                text="", elapsed_sec=elapsed, success=False,
                error="Empty response from model",
            )
        return ModelResponse(
            model=model_name, display_name=display_name,
            text=text, elapsed_sec=elapsed, success=True,
        )

    except requests.exceptions.Timeout:
        return ModelResponse(
            model=model_name, display_name=display_name, text="",
            elapsed_sec=time.perf_counter() - start, success=False,
            error=f"Timeout after {timeout}s",
        )
    except requests.exceptions.ConnectionError:
        return ModelResponse(
            model=model_name, display_name=display_name, text="",
            elapsed_sec=time.perf_counter() - start, success=False,
            error="Connection refused — is Ollama running?",
        )
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = resp.json().get("error", str(e))
        except Exception:
            detail = str(e)
        return ModelResponse(
            model=model_name, display_name=display_name, text="",
            elapsed_sec=time.perf_counter() - start, success=False,
            error=f"HTTP {resp.status_code}: {detail}",
        )
    except Exception as e:
        return ModelResponse(
            model=model_name, display_name=display_name, text="",
            elapsed_sec=time.perf_counter() - start, success=False,
            error=str(e),
        )


def stream_model(
    model_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    num_ctx: int = 4096,
    num_predict: int = 1024,
    timeout: Optional[int] = None,
) -> Generator[str, None, None]:
    """Stream tokens from an Ollama model as a generator. Yields chunks as they arrive."""
    if timeout is None:
        timeout = HARDWARE["request_timeout_sec"]

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }
    if system_prompt:
        payload["system"] = system_prompt

    try:
        with requests.post(
            OLLAMA_API_GENERATE,
            json=payload,
            stream=True,
            timeout=timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        yield f"\n[Stream error: {e}]"
