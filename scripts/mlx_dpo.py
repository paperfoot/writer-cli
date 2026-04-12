#!/usr/bin/env python3
"""SimPO/DPO preference training on Apple Silicon via mlx-lm.

Protocol: reads JSON config from stdin, prints progress to stdout,
writes final adapter to the specified output directory.

Config:
    {
        "model": "mlx-community/gemma-4-26b-a4b-it-4bit",
        "dataset_path": "/path/to/preference_pairs.jsonl",
        "adapter_out": "/path/to/output/adapters",
        "resume_adapter": "/path/to/sft/adapters",  // optional
        "method": "simpo" | "dpo",
        "beta": 0.1,
        "gamma": 1.0,
        "learning_rate": 1e-6,
        "batch_size": 1,
        "max_steps": 500,
        "max_seq_len": 2048,
        "lora_rank": 16,
        "lora_scale": 2.0
    }

Preference dataset format (JSONL):
    {"prompt": "...", "chosen": "...", "rejected": "..."}

Progress output (one JSON per line to stdout):
    {"step": N, "total_steps": N, "loss": F, "learning_rate": F,
     "chosen_reward": F, "rejected_reward": F, "reward_margin": F}
"""
import json
import os
import sys
import time


def main():
    config = json.loads(sys.stdin.read())

    model_path = config["model"]
    dataset_path = config["dataset_path"]
    adapter_out = config["adapter_out"]
    resume_adapter = config.get("resume_adapter")
    method = config.get("method", "simpo")
    beta = config.get("beta", 0.1)
    gamma = config.get("gamma", 1.0)
    learning_rate = config.get("learning_rate", 1e-6)
    batch_size = config.get("batch_size", 1)
    max_steps = config.get("max_steps", 500)
    max_seq_len = config.get("max_seq_len", 2048)
    lora_rank = config.get("lora_rank", 16)
    lora_scale = config.get("lora_scale", 2.0)

    # Try mlx-tune first (has DPO/SimPO support), fall back to manual implementation
    try:
        from mlx_tune import SimPOTrainer, DPOTrainer, TrainingArguments
        _train_with_mlx_tune(
            model_path=model_path,
            dataset_path=dataset_path,
            adapter_out=adapter_out,
            resume_adapter=resume_adapter,
            method=method,
            beta=beta,
            gamma=gamma,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_steps=max_steps,
            max_seq_len=max_seq_len,
            lora_rank=lora_rank,
            lora_scale=lora_scale,
        )
    except ImportError:
        sys.stderr.write(
            "mlx-tune not installed, using built-in preference trainer\n"
        )
        sys.stderr.flush()
        _train_builtin(
            model_path=model_path,
            dataset_path=dataset_path,
            adapter_out=adapter_out,
            resume_adapter=resume_adapter,
            method=method,
            beta=beta,
            gamma=gamma,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_steps=max_steps,
            max_seq_len=max_seq_len,
            lora_rank=lora_rank,
            lora_scale=lora_scale,
        )


def _train_with_mlx_tune(**kwargs):
    """Train using the mlx-tune package (preferred, has native SimPO)."""
    raise ImportError("mlx-tune integration not yet wired")


def _train_builtin(**kwargs):
    """Built-in SimPO/DPO trainer using raw mlx-lm primitives.

    SimPO loss (Yu et al., NeurIPS 2024):
        L = -log(sigma(beta * (r_chosen - r_rejected - gamma)))
    where r = avg_log_prob(completion) (reference-free).

    DPO loss (Rafailov et al., NeurIPS 2023):
        L = -log(sigma(beta * (log(pi/ref)_chosen - log(pi/ref)_rejected)))
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model_path = kwargs["model_path"]
    dataset_path = kwargs["dataset_path"]
    adapter_out = kwargs["adapter_out"]
    resume_adapter = kwargs.get("resume_adapter")
    method = kwargs.get("method", "simpo")
    beta = kwargs.get("beta", 0.1)
    gamma = kwargs.get("gamma", 1.0)
    learning_rate = kwargs.get("learning_rate", 1e-6)
    batch_size = kwargs.get("batch_size", 1)
    max_steps = kwargs.get("max_steps", 500)
    max_seq_len = kwargs.get("max_seq_len", 2048)
    lora_rank = kwargs.get("lora_rank", 16)
    lora_scale = kwargs.get("lora_scale", 2.0)

    sys.stderr.write(f"Loading model: {model_path}\n")
    sys.stderr.flush()

    # Load model with optional SFT adapter as starting point
    model, tokenizer = load(model_path, adapter_path=resume_adapter)

    # Apply LoRA layers if not already present from resume
    if resume_adapter is None:
        lora_config = {"rank": lora_rank, "scale": lora_scale, "dropout": 0.0}
        linear_to_lora_layers(model, 16, lora_config)

    # Freeze non-LoRA parameters
    model.freeze()
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    # Load preference dataset
    pairs = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    if not pairs:
        sys.stderr.write("Error: empty preference dataset\n")
        sys.exit(1)

    sys.stderr.write(f"Loaded {len(pairs)} preference pairs\n")
    sys.stderr.write(f"Method: {method}, beta={beta}, gamma={gamma}, lr={learning_rate}\n")
    sys.stderr.flush()

    # Set up optimizer
    optimizer = optim.Adam(learning_rate=learning_rate)

    # Training loop
    t0 = time.time()
    for step in range(1, max_steps + 1):
        pair = pairs[(step - 1) % len(pairs)]

        prompt = pair["prompt"]
        chosen = pair["chosen"]
        rejected = pair["rejected"]

        # Tokenize
        chosen_text = _format_chat(tokenizer, prompt, chosen)
        rejected_text = _format_chat(tokenizer, prompt, rejected)

        chosen_ids = mx.array(tokenizer.encode(chosen_text)[:max_seq_len])
        rejected_ids = mx.array(tokenizer.encode(rejected_text)[:max_seq_len])

        # Compute loss and gradients
        loss_val, grads, metrics = _compute_preference_loss(
            model, chosen_ids, rejected_ids, method, beta, gamma
        )

        # Update
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        # Report progress
        elapsed = time.time() - t0
        progress = {
            "step": step,
            "total_steps": max_steps,
            "loss": float(loss_val),
            "learning_rate": learning_rate,
            "chosen_reward": float(metrics.get("chosen_reward", 0)),
            "rejected_reward": float(metrics.get("rejected_reward", 0)),
            "reward_margin": float(metrics.get("margin", 0)),
            "elapsed_s": round(elapsed, 1),
        }
        sys.stdout.write(json.dumps(progress) + "\n")
        sys.stdout.flush()

        if step % 50 == 0:
            sys.stderr.write(
                f"Step {step}/{max_steps} | loss={float(loss_val):.4f} "
                f"| margin={float(metrics.get('margin', 0)):.3f} "
                f"| {elapsed:.0f}s\n"
            )
            sys.stderr.flush()

    # Save adapter
    os.makedirs(adapter_out, exist_ok=True)
    # Save only LoRA weights
    lora_weights = {
        k: v for k, v in model.parameters().items() if "lora" in k.lower()
    }
    mx.save_safetensors(os.path.join(adapter_out, "adapters.safetensors"), lora_weights)

    sys.stderr.write(f"Adapter saved to {adapter_out}\n")
    sys.stderr.flush()


def _format_chat(tokenizer, prompt, completion):
    """Format a prompt+completion pair using the model's chat template."""
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False)
    return f"{prompt}\n\n{completion}"


def _compute_preference_loss(model, chosen_ids, rejected_ids, method, beta, gamma):
    """Compute SimPO or DPO loss with gradients."""
    import mlx.core as mx
    import mlx.nn as nn

    def loss_fn(model):
        # Forward pass for chosen
        chosen_logits = model(chosen_ids[None, :-1])
        chosen_targets = chosen_ids[1:]
        chosen_log_probs = -nn.losses.cross_entropy(
            chosen_logits.squeeze(0), chosen_targets, reduction="none"
        )
        chosen_avg_logp = chosen_log_probs.mean()

        # Forward pass for rejected
        rejected_logits = model(rejected_ids[None, :-1])
        rejected_targets = rejected_ids[1:]
        rejected_log_probs = -nn.losses.cross_entropy(
            rejected_logits.squeeze(0), rejected_targets, reduction="none"
        )
        rejected_avg_logp = rejected_log_probs.mean()

        if method == "simpo":
            # SimPO: reference-free, length-normalized
            # L = -log(sigma(beta * (r_chosen - r_rejected - gamma)))
            margin = chosen_avg_logp - rejected_avg_logp
            loss = -mx.log(mx.sigmoid(beta * (margin - gamma)))
        else:
            # Standard DPO (simplified without separate ref model)
            margin = chosen_avg_logp - rejected_avg_logp
            loss = -mx.log(mx.sigmoid(beta * margin))

        return loss, {
            "chosen_reward": chosen_avg_logp,
            "rejected_reward": rejected_avg_logp,
            "margin": margin,
        }

    # Compute loss and gradients
    (loss_val, metrics), grads = nn.value_and_grad(model, lambda m: loss_fn(m))(model)

    return loss_val, grads, metrics


if __name__ == "__main__":
    main()
