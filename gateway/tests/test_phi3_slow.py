"""Slow integration test loading real Phi-3 Mini via HFLocalBackend."""

import pytest
import torch

from slm_gateway.config import Settings
from slm_gateway.backends.hf_local import HFLocalBackend


@pytest.mark.slow
@pytest.mark.asyncio
async def test_phi3_real_inference():
    """Verify in-process loading and generation with 4-bit quantized Phi-3 Mini."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is not available for 4-bit Phi-3 test")

    cfg = Settings(
        BACKEND="hf_local",
        MODEL_ID="microsoft/Phi-3-mini-4k-instruct",
        QUANTIZE="4bit",
    )
    backend = HFLocalBackend(cfg)

    try:
        await backend.load()
        assert backend.is_ready()
        assert backend.get_model_name() == "microsoft/Phi-3-mini-4k-instruct"

        messages = [
            {"role": "user", "content": "What is 2 + 2? Answer in one word."}
        ]
        content, prompt_tokens, completion_tokens, finish_reason = await backend.generate(
            messages=messages,
            temperature=0.0,
            max_tokens=30,
        )

        assert isinstance(content, str)
        assert len(content.strip()) > 0
        assert prompt_tokens > 0
        assert completion_tokens > 0
        assert finish_reason in ("stop", "length")
        assert "4" in content or "four" in content.lower()
    finally:
        await backend.close()
