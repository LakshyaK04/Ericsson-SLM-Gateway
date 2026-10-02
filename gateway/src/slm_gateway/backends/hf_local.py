"""In-process HuggingFace backend using Phi-3-mini and bitsandbytes."""

import asyncio
import logging
from threading import Thread
from typing import AsyncIterator, Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from ..config import Settings
from .base import LLMBackend

logger = logging.getLogger(__name__)


class HFLocalBackend(LLMBackend):
    """Local in-process LLM backend running Phi-3 Mini."""

    def __init__(self, config: Settings):
        self.config = config
        self.model_id = config.MODEL_ID
        self.quantize = config.QUANTIZE
        self.max_context = config.MAX_CONTEXT_LENGTH
        self.tokenizer = None
        self.model = None
        self._semaphore = asyncio.Semaphore(1)
        self._ready = False

    async def load(self) -> None:
        """Load tokenizer and quantized model into GPU/CPU memory."""
        logger.info("Initializing HFLocalBackend with model: %s", self.model_id)

        # Run model loading in thread to avoid blocking event loop during startup
        await asyncio.to_thread(self._load_sync)
        self._ready = True
        logger.info("HFLocalBackend loaded and ready.")

    def _load_sync(self) -> None:
        logger.info("Loading tokenizer for %s...", self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        has_cuda = torch.cuda.is_available()

        if self.quantize == "4bit":
            if not has_cuda:
                raise RuntimeError(
                    "CUDA GPU is required for 4-bit quantization with bitsandbytes. "
                    "Set QUANTIZE=none or use BACKEND=openai_compatible."
                )

            logger.info("Configuring 4-bit NF4 quantization for GPU: %s", torch.cuda.get_device_name(0))
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=quant_config,
                device_map=self.config.DEVICE,
            )
        else:
            device = "cuda" if has_cuda else "cpu"
            logger.info("Loading model without quantization on device: %s", device)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if has_cuda else torch.float32,
                device_map=device,
            )

        self.model.eval()

    def is_ready(self) -> bool:
        return self._ready and self.model is not None and self.tokenizer is not None

    def get_model_name(self) -> str:
        return self.model_id

    def _trim_messages_if_needed(self, messages: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
        """Trim oldest messages if total tokens exceed max_context."""
        active_messages = list(messages)
        preserve_system = len(active_messages) > 0 and active_messages[0].get("role") == "system"

        while True:
            prompt_str = self.tokenizer.apply_chat_template(
                active_messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            tokens = self.tokenizer.encode(prompt_str)
            if len(tokens) + max_tokens <= self.max_context:
                return active_messages

            # If only 1 message remains (or 1 system + 1 user), we can't trim further
            min_count = 2 if preserve_system else 1
            if len(active_messages) <= min_count:
                logger.warning(
                    "Prompt with %d tokens exceeds context limit %d even after trimming.",
                    len(tokens),
                    self.max_context,
                )
                raise ValueError(
                    f"Prompt length ({len(tokens)} tokens) plus max_tokens ({max_tokens}) "
                    f"exceeds maximum context window ({self.max_context} tokens)."
                )

            # Trim the oldest non-system message
            trim_idx = 1 if preserve_system else 0
            dropped = active_messages.pop(trim_idx)
            logger.warning("Context limit approached: trimmed message (%s): %s...", dropped.get("role"), dropped.get("content", "")[:30])

    def _generate_sync(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Tuple[str, int, int, str]:
        trimmed_messages = self._trim_messages_if_needed(messages, max_tokens)

        inputs = self.tokenizer.apply_chat_template(
            trimmed_messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        if isinstance(inputs, torch.Tensor):
            input_ids = inputs.to(self.model.device)
            attention_mask = None
        else:
            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.model.device)

        prompt_tokens = input_ids.shape[-1]

        generate_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask

        if temperature <= 0.0:
            generate_kwargs["do_sample"] = False
        else:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        with torch.no_grad():
            outputs = self.model.generate(**generate_kwargs)

        new_tokens = outputs[0][prompt_tokens:]
        completion_tokens = len(new_tokens)

        # Determine finish reason
        if completion_tokens >= max_tokens:
            finish_reason = "length"
        else:
            finish_reason = "stop"

        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response_text, prompt_tokens, completion_tokens, finish_reason

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 512,
    ) -> Tuple[str, int, int, str]:
        """Generate response with single-concurrency lock on GPU."""
        async with self._semaphore:
            return await asyncio.to_thread(
                self._generate_sync,
                messages,
                temperature,
                top_p,
                max_tokens,
            )

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 512,
    ) -> AsyncIterator[str]:
        """Execute token streaming chat completion protected by single-concurrency GPU lock."""
        async with self._semaphore:
            trimmed_messages = self._trim_messages_if_needed(messages, max_tokens)
            inputs = self.tokenizer.apply_chat_template(
                trimmed_messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if isinstance(inputs, torch.Tensor):
                input_ids = inputs.to(self.model.device)
                attention_mask = None
            else:
                input_ids = inputs["input_ids"].to(self.model.device)
                attention_mask = inputs.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.model.device)

            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            generate_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": max_tokens,
                "pad_token_id": self.tokenizer.eos_token_id,
                "streamer": streamer,
            }
            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask

            if temperature <= 0.0:
                generate_kwargs["do_sample"] = False
            else:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = temperature
                generate_kwargs["top_p"] = top_p

            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()

            def run_generation():
                try:
                    with torch.no_grad():
                        self.model.generate(**generate_kwargs)
                except Exception as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, exc)

            def stream_tokens():
                try:
                    for text in streamer:
                        if text:
                            loop.call_soon_threadsafe(queue.put_nowait, text)
                except Exception as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, exc)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            gen_thread = Thread(target=run_generation, daemon=True)
            stream_thread = Thread(target=stream_tokens, daemon=True)
            gen_thread.start()
            stream_thread.start()

            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item

    async def close(self) -> None:
        """Unload model and free GPU memory."""
        logger.info("Unloading HFLocalBackend...")
        self._ready = False
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
