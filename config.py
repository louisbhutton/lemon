
from dataclasses import dataclass
from typing import Optional

# designed specifically for gtx 1660ti 

HARDWARE: dict = {
    "vram_gb": 6,
    "ram_gb": 32,
    "request_timeout_sec": 120,
    "inter_model_delay_sec": 0.5,  
    "allow_parallel": False,         
    "max_workers": 1,
}


#  ollama server configuration


OLLAMA_BASE_URL   = "http://localhost:11434"
OLLAMA_API_GENERATE = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_API_TAGS     = f"{OLLAMA_BASE_URL}/api/tags"


# model registry


@dataclass
class ModelConfig:
    name: str
    display_name: str
    capabilities: list     
    temperature: float = 0.7
    top_p: float = 0.9
    num_ctx: int = 4096
    num_predict: int = 1024
    enabled: bool = True
    is_judge: bool = False
    size_gb: float = 5.0   
    notes: str = ""


MODEL_REGISTRY: list = [
    ModelConfig(
        name="deepseek-r1:7b",
        display_name="DeepSeek R1 7B",
        capabilities=["reasoning", "math", "analysis", "general"],
        temperature=0.6,
        num_predict=2048,
        is_judge=True,
        size_gb=4.7,
        notes="Chain-of-thought reasoning; best for synthesis",
    ),
    ModelConfig(
        name="qwen2.5:7b",
        display_name="Qwen 2.5 7B",
        capabilities=["coding", "structured", "analysis", "general"],
        temperature=0.5,
        num_predict=1536,
        size_gb=4.5,
        notes="Excellent at structured output and code",
    ),
    ModelConfig(
        name="llama3.2",
        display_name="Llama 3.2",
        capabilities=["general", "creative", "conversation", "summarization"],
        temperature=0.75,
        size_gb=2.0,          # 3B model — lighter
        notes="Balanced general-purpose assistant",
    ),
    ModelConfig(
        name="mistral:7b",
        display_name="Mistral 7B",
        capabilities=["general", "analysis", "reasoning", "creative"],
        temperature=0.7,
        size_gb=4.5,
        notes="Strong at instruction following and reasoning",
    ),
    ModelConfig(
        name="gemma:7b",
        display_name="Gemma 7B",
        capabilities=["general", "creative", "conversation"],
        temperature=0.8,
        size_gb=5.0,
        notes="Good creative and conversational model",
    ),
    ModelConfig(
        name="codellama:7b",
        display_name="CodeLlama 7B",
        capabilities=["coding", "debugging", "technical"],
        temperature=0.3,
        num_predict=2048,
        size_gb=4.5,
        notes="Specialized for code generation and debugging",
    ),
]


#  routing config

ROUTING_KEYWORDS: dict = {
    "coding": [
        "code", "function", "class", "def ", "bug", "error", "debug",
        "python", "javascript", "typescript", "java", "c++", "rust",
        "sql", "script", "implement", "refactor", "algorithm", "syntax",
        "compile", "runtime", "api", "library", "framework", "unittest",
    ],
    "math": [
        "calculate", "compute", "equation", "integral", "derivative",
        "matrix", "probability", "statistics", "formula", "proof",
        "algebra", "geometry", "trigonometry", "calculus", "number",
    ],
    "reasoning": [
        "why", "explain", "analyze", "compare", "evaluate", "argue",
        "pros and cons", "difference between", "which is better",
        "reason", "cause", "effect", "logic", "infer", "deduce",
    ],
    "creative": [
        "write", "story", "poem", "creative", "imagine", "fictional",
        "narrative", "character", "plot", "essay", "blog", "caption",
    ],
    "structured": [
        "json", "xml", "yaml", "table", "list", "format", "structured",
        "extract", "parse", "schema",
    ],
}

ROUTE_CAPABILITY_MAP: dict = {
    "coding":     ["coding", "debugging", "technical"],
    "math":       ["reasoning", "math", "analysis"],
    "reasoning":  ["reasoning", "analysis", "general"],
    "creative":   ["creative", "general", "conversation"],
    "structured": ["structured", "coding", "analysis"],
    "general":    ["general", "reasoning", "creative"],
}

MODELS_PER_ROUTE: dict = {
    "coding":     3,
    "math":       2,
    "reasoning":  3,
    "creative":   3,
    "structured": 2,
    "general":    3,
}


#  GPU mode system

GPU_MODE: str = "balanced"   

GPU_MODE_CONFIGS: dict = {
    "light": {
        # Execution
        "allow_parallel":        False,
        "max_workers":           1,
        "inter_model_delay_sec": 1.5,
        "request_timeout_sec":   90,
        # Panel
        "max_panel_size":        1,
        "prefer_light_models":   True,    
        # Per-model caps
        "num_ctx_cap":           2048,
        "num_predict_cap":       512,
        # Judge
        "judge_enabled":         False,
        "judge_num_ctx":         2048,
        "judge_num_predict":     512,
        "description": "Ultra-safe · 1 model · No judge · Minimum VRAM",
    },
    "balanced": {
        "allow_parallel":        False,
        "max_workers":           1,
        "inter_model_delay_sec": 0.5,
        "request_timeout_sec":   120,
        "max_panel_size":        3,
        "prefer_light_models":   False,
        "num_ctx_cap":           4096,
        "num_predict_cap":       1024,
        "judge_enabled":         True,
        "judge_num_ctx":         6144,
        "judge_num_predict":     1536,
        "description": "Recommended · 2-3 models · Sequential · Judge enabled",
    },
    "aggressive": {
        "allow_parallel":        True,
        "max_workers":           2,
        "inter_model_delay_sec": 0.2,
        "request_timeout_sec":   180,
        "max_panel_size":        6,
        "prefer_light_models":   False,
        "num_ctx_cap":           8192,
        "num_predict_cap":       2048,
        "judge_enabled":         True,
        "judge_num_ctx":         8192,
        "judge_num_predict":     2048,
        "description": "Max performance · All models · Parallel allowed · High VRAM",
    },
}


def get_gpu_cfg() -> dict:
    """Return the active GPU mode config dict. Always reads current GPU_MODE."""
    return GPU_MODE_CONFIGS[GPU_MODE]


def apply_gpu_mode(mode: str) -> None:
    """
    Activate a GPU mode by patching HARDWARE and updating GPU_MODE.
    Call this once at program start, before any model runs.

    Args:
        mode: "light" | "balanced" | "aggressive"
    """
    global GPU_MODE
    if mode not in GPU_MODE_CONFIGS:
        raise ValueError(
            f"Unknown --gpu-mode {mode!r}. "
            f"Choose: {' | '.join(GPU_MODE_CONFIGS)}"
        )
    GPU_MODE = mode
    cfg = GPU_MODE_CONFIGS[mode]
    HARDWARE["allow_parallel"]        = cfg["allow_parallel"]
    HARDWARE["max_workers"]           = cfg["max_workers"]
    HARDWARE["inter_model_delay_sec"] = cfg["inter_model_delay_sec"]
    HARDWARE["request_timeout_sec"]   = cfg["request_timeout_sec"]



#  Judge prompt templates

JUDGE_SYSTEM_PROMPT = """You are an expert AI response evaluator and synthesizer.
You will receive a user question and multiple responses from different AI models.
Your task is to:
1. Evaluate each response for accuracy, completeness, clarity, and relevance.
2. Identify the strongest points from each response.
3. Produce a single, refined, comprehensive answer that:
   - Is more accurate and complete than any individual response
   - Preserves the best reasoning and insights
   - Eliminates errors, redundancies, or weak content
   - Is clearly written and well-structured
Do NOT mention the individual models or that you are synthesizing responses.
Output ONLY the final refined answer, nothing else."""

JUDGE_USER_TEMPLATE = """USER QUESTION:
{question}

MODEL RESPONSES:
{responses}

Produce the best possible final answer based on the above."""
