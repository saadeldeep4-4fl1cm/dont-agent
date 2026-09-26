"""Model loading and text generation for Don't Agent.

Loads Qwen2.5-Coder-1.5B (plus an optional LoRA adapter) once, then exposes
a single `generate(messages)` function used by the agent loop.

Environment:
    MODEL_ID       base model id or local path (default "Qwen/Qwen2.5-Coder-1.5B")
    ADAPTER_ID     optional LoRA adapter id/path loaded on top of the base model
    MAX_NEW_TOKENS max tokens per generation (default "512")
"""

from __future__ import annotations

import os
import threading

import torch

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-Coder-1.5B")
ADAPTER_ID = os.environ.get("ADAPTER_ID", "").strip() or None
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))

SYSTEM_PROMPT = (
    "You are Don't Agent, an expert AI programming agent. "
    "You write clean, correct code, explain your reasoning step by step, "
    "and break complex tasks into clear actionable steps. "
    "You are helpful, precise, and honest about your limitations."
)

_model = None
_tokenizer = None
_lock = threading.Lock()


def _load():
    """Load model + tokenizer once (thread-safe)."""
    global _model, _tokenizer
    with _lock:
        if _model is not None:
            return _model, _tokenizer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig

            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=(
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                ),
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=quant,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            # CPU: float16 halves RAM (~3GB for 1.5B) vs float32 (~6GB).
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True,
            )
        if ADAPTER_ID:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, ADAPTER_ID)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _model, _tokenizer = model, tokenizer
        return _model, _tokenizer


def is_loaded() -> bool:
    return _model is not None


def generate(messages: list[dict]) -> str:
    """Generate a reply for chat messages.

    Each message is {"role": "user"|"assistant"|"system", "content": str}.
    The Don't Agent system prompt is prepended when none was supplied.
    """
    model, tokenizer = _load()
    msgs = list(messages)
    if not msgs or msgs[0].get("role") != "system":
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
    prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return text.strip()
