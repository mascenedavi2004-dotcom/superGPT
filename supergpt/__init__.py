# supergpt — Frontier LLM Training Framework
from supergpt.core.config import GPTConfig, TrainConfig, get_model_config
from supergpt.core.model import GPT

__all__ = ["GPT", "GPTConfig", "TrainConfig", "get_model_config"]
__version__ = "2.0.0"
