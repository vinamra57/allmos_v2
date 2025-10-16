"""
Simple user-facing LLM API.

This is a thin wrapper around LLMEngine that provides a clean,
nano-vLLM-compatible interface.
"""
from engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """
    High-level LLM inference API.

    Usage:
        >>> from llm import LLM
        >>> from sampling_params import SamplingParams
        >>>
        >>> llm = LLM("~/huggingface/Qwen3-0.6B/", enforce_eager=True)
        >>> outputs = llm.generate(
        ...     prompts=["Hello, world!"],
        ...     sampling_params=SamplingParams(temperature=0.8, max_tokens=100)
        ... )
        >>> print(outputs[0]['text'])
    """
    pass
