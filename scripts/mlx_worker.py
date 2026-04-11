#!/usr/bin/env python3
"""Persistent MLX inference worker — loads model once, serves many requests.

Protocol: reads one JSON request per line from stdin, writes one JSON response
per line to stdout. The worker exits when stdin is closed or it reads an empty line.

This replaces the one-shot mlx_generate.py for batch workloads like eval and
ablation, where model loading (15GB, ~10s) dominates per-generation cost.

Request (one JSON object per line):
    {"prompt": "...", "system_prompt": "...", "prompt_mode": "chat|raw",
     "max_tokens": 2048, "temperature": 0.7, "top_p": 0.92,
     "repetition_penalty": 1.05, "seed": 42, "logit_bias": {...}}

Response (one JSON object per line):
    {"text": "...", "prompt_tokens": N, "generation_tokens": N,
     "generation_tps": F, "peak_memory_gb": F, "finish_reason": "stop",
     "elapsed_ms": N}

Startup: first line must be a JSON config:
    {"model": "mlx-community/gemma-4-26b-a4b-it-4bit",
     "adapter_path": "/path/to/adapters"}  // optional
"""
import json
import sys
import time


def build_logits_processor(logit_bias, tokenizer):
    """Build an MLX logits processor from a word->bias map."""
    if not logit_bias:
        return None

    import mlx.core as mx

    token_biases = {}
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}

    for word, bias in logit_bias.items():
        word_lower = word.lower()
        for token_str, token_id in vocab.items():
            clean = token_str.lower().strip("\u2581\u0120 ")
            if clean == word_lower:
                token_biases[token_id] = token_biases.get(token_id, 0.0) + bias

    if not token_biases:
        return None

    bias_ids = list(token_biases.keys())
    bias_vals = mx.array([token_biases[i] for i in bias_ids], dtype=mx.float32)
    bias_ids_arr = mx.array(bias_ids, dtype=mx.int32)

    def processor(tokens, logits):
        updates = logits[..., bias_ids_arr] + bias_vals
        logits[..., bias_ids_arr] = updates
        return logits

    return processor


def main():
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    # Read startup config (first line)
    startup_line = sys.stdin.readline().strip()
    if not startup_line:
        sys.exit(0)

    startup = json.loads(startup_line)
    model_path = startup["model"]
    adapter_path = startup.get("adapter_path")

    # Load model once
    sys.stderr.write(f"worker: loading {model_path}")
    if adapter_path:
        sys.stderr.write(f" + adapter {adapter_path}")
    sys.stderr.write("\n")
    sys.stderr.flush()

    model, tokenizer = load(model_path, adapter_path=adapter_path)

    sys.stderr.write("worker: ready\n")
    sys.stderr.flush()

    # Signal readiness
    ready = json.dumps({"status": "ready"})
    sys.stdout.write(ready + "\n")
    sys.stdout.flush()

    # Process requests
    for line in sys.stdin:
        line = line.strip()
        if not line:
            break

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            err = json.dumps({"error": f"invalid JSON: {e}"})
            sys.stdout.write(err + "\n")
            sys.stdout.flush()
            continue

        prompt_text = req.get("prompt", "")
        system_prompt = req.get("system_prompt")
        prompt_mode = req.get("prompt_mode", "chat")
        max_tokens = req.get("max_tokens", 2048)
        temperature = req.get("temperature", 0.7)
        top_p = req.get("top_p", 0.92)
        repetition_penalty = req.get("repetition_penalty", 1.05)
        logit_bias = req.get("logit_bias")
        seed = req.get("seed")

        # Set seed for reproducibility
        if seed is not None:
            mx.random.seed(seed)

        # Format prompt
        if prompt_mode == "raw":
            formatted = prompt_text
        else:
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
        logits_processor = build_logits_processor(logit_bias, tokenizer)

        gen_kwargs = dict(max_tokens=max_tokens, sampler=sampler)
        if logits_processor is not None:
            gen_kwargs["logits_processors"] = [logits_processor]

        t0 = time.time()

        full_text = ""
        last_resp = None
        for resp in stream_generate(model, tokenizer, formatted, **gen_kwargs):
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

        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()

    sys.stderr.write("worker: shutting down\n")
    sys.stderr.flush()


if __name__ == "__main__":
    main()
