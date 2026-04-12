#!/usr/bin/env python3
"""MLX inference bridge — loads model + optional LoRA adapter, generates text.

Protocol: reads JSON request from stdin, writes JSON response to stdout.

Request:
    {
        "model": "mlx-community/gemma-4-26b-a4b-it-4bit",
        "adapter_path": "/path/to/adapters",  // optional
        "prompt": "user prompt text",
        "system_prompt": "system prompt text",  // optional
        "prompt_mode": "chat" | "raw",          // default: "chat"
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.92,
        "repetition_penalty": 1.05,
        "logit_bias": {"word": -4.0, ...}       // optional
    }

Response:
    {
        "text": "generated text",
        "prompt_tokens": 123,
        "generation_tokens": 456,
        "generation_tps": 12.3,
        "peak_memory_gb": 18.5,
        "finish_reason": "stop"
    }
"""
import json
import sys
import time


def build_logits_processor(logit_bias, tokenizer):
    """Build an MLX logits processor from a word->bias map.

    For each word in the bias map, find all token IDs that encode that word
    (or contain it as a prefix), and apply the bias additively to those logits.
    """
    if not logit_bias:
        return None

    import mlx.core as mx

    # Build token_id -> bias mapping
    token_biases = {}
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}

    for word, bias in logit_bias.items():
        word_lower = word.lower()
        for token_str, token_id in vocab.items():
            # Match tokens that are the word itself or common tokenizer variants
            clean = token_str.lower().strip("▁Ġ ")
            if clean == word_lower:
                token_biases[token_id] = token_biases.get(token_id, 0.0) + bias

    if not token_biases:
        return None

    # Pre-compute as arrays for fast application
    bias_ids = list(token_biases.keys())
    bias_vals = mx.array([token_biases[i] for i in bias_ids], dtype=mx.float32)
    bias_ids_arr = mx.array(bias_ids, dtype=mx.int32)

    def processor(tokens, logits):
        # Add bias to specific token logits
        updates = logits[..., bias_ids_arr] + bias_vals
        logits[..., bias_ids_arr] = updates
        return logits

    return processor


def _make_repetition_processor(penalty, context_size=100):
    """Build a logits processor that penalises recently generated tokens."""
    import mlx.core as mx

    generated_tokens = []

    def processor(tokens, logits):
        generated_tokens.append(int(tokens[-1]) if tokens.size > 0 else 0)
        recent = generated_tokens[-context_size:]
        if recent:
            ids = mx.array(list(set(recent)), dtype=mx.int32)
            penalties = mx.where(
                logits[..., ids] > 0,
                logits[..., ids] / penalty,
                logits[..., ids] * penalty,
            )
            logits[..., ids] = penalties
        return logits

    return processor


def main():
    req = json.loads(sys.stdin.read())

    model_path = req["model"]
    adapter_path = req.get("adapter_path")
    prompt_text = req["prompt"]
    system_prompt = req.get("system_prompt")
    prompt_mode = req.get("prompt_mode", "chat")
    max_tokens = req.get("max_tokens", 2048)
    temperature = req.get("temperature", 0.7)
    top_p = req.get("top_p", 0.92)
    repetition_penalty = req.get("repetition_penalty", 1.05)
    logit_bias = req.get("logit_bias")
    seed = req.get("seed")

    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    # Set seed for reproducibility if provided
    if seed is not None:
        mx.random.seed(seed)

    model, tokenizer = load(model_path, adapter_path=adapter_path)

    # Format prompt based on mode
    if prompt_mode == "raw":
        # Raw mode: send prompt verbatim, no chat template, no message wrapping
        formatted = prompt_text
    else:
        # Chat mode: build messages and apply chat template
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt_text})

        if hasattr(tokenizer, "apply_chat_template"):
            formatted = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            formatted = ""
            if system_prompt:
                formatted += system_prompt + "\n\n"
            formatted += prompt_text

    sampler = make_sampler(temp=temperature, top_p=top_p)

    # Build logits processor for vocabulary bias
    logits_processor = build_logits_processor(logit_bias, tokenizer)

    # Chain repetition penalty as a logits processor
    processors = []
    if repetition_penalty > 1.0:
        processors.append(
            _make_repetition_processor(repetition_penalty, context_size=100)
        )
    if logits_processor is not None:
        processors.append(logits_processor)

    t0 = time.time()

    gen_kwargs = dict(
        max_tokens=max_tokens,
        sampler=sampler,
    )
    if processors:
        gen_kwargs["logits_processors"] = processors

    full_text = ""
    last_resp = None
    for resp in stream_generate(
        model,
        tokenizer,
        formatted,
        **gen_kwargs,
    ):
        full_text += resp.text
        last_resp = resp

    elapsed = time.time() - t0

    result = {
        "text": full_text.strip(),
        "prompt_tokens": last_resp.prompt_tokens if last_resp else 0,
        "generation_tokens": last_resp.generation_tokens if last_resp else 0,
        "generation_tps": last_resp.generation_tps if last_resp else 0.0,
        "peak_memory_gb": last_resp.peak_memory if last_resp else 0.0,
        "finish_reason": last_resp.finish_reason if last_resp else "unknown",
        "elapsed_ms": int(elapsed * 1000),
    }

    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
