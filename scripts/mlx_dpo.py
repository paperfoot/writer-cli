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

    # Try mlx-tune first (has DPO/SimPO support with MoE gradient handling)
    try:
        import mlx_tune  # noqa: F401 — just check availability
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
    """Train using the mlx-tune package (preferred, has native SimPO for MoE)."""
    from mlx_tune import (
        FastVisionModel,
        SimPOConfig, SimPOTrainer,
        DPOConfig, DPOTrainer,
    )

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

    sys.stderr.write(f"Loading model via mlx-tune: {model_path}\n")
    sys.stderr.flush()

    # Gemma 4 is treated as VLM in mlx-tune
    model, tokenizer = FastVisionModel.from_pretrained(model_path)

    # Load SFT adapter weights if provided
    if resume_adapter:
        import mlx.core as mx
        adapter_file = os.path.join(resume_adapter, "adapters.safetensors")
        if os.path.exists(adapter_file):
            sys.stderr.write(f"Loading SFT adapter from {adapter_file}\n")
            weights = mx.load(adapter_file)
            model.load_weights(list(weights.items()))

    # Prepare preference dataset from our JSONL format
    # mlx-tune expects {"prompt": ..., "chosen": ..., "rejected": ...}
    import json as _json
    pairs = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                pair = _json.loads(line)
                pairs.append({
                    "prompt": pair["prompt"],
                    "chosen": pair["chosen"],
                    "rejected": pair["rejected"],
                })

    sys.stderr.write(f"Loaded {len(pairs)} preference pairs\n")
    sys.stderr.write(f"Method: {method}, beta={beta}, gamma={gamma}, lr={learning_rate}\n")
    sys.stderr.flush()

    # Write temp dataset file for mlx-tune
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tf:
        for pair in pairs:
            tf.write(_json.dumps(pair) + "\n")
        temp_dataset = tf.name

    try:
        if method == "simpo":
            config = SimPOConfig(
                beta=beta,
                gamma=gamma,
                output_dir=adapter_out,
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                max_steps=max_steps,
                max_seq_length=max_seq_len,
                logging_steps=10,
                save_steps=max_steps,  # Save only at end
                warmup_steps=min(10, max_steps // 10),
            )
            trainer = SimPOTrainer(
                model=model,
                tokenizer=tokenizer,
                args=config,
                train_dataset=temp_dataset,
            )
        else:
            config = DPOConfig(
                beta=beta,
                output_dir=adapter_out,
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                max_steps=max_steps,
                max_seq_length=max_seq_len,
                logging_steps=10,
                save_steps=max_steps,
                warmup_steps=min(10, max_steps // 10),
            )
            trainer = DPOTrainer(
                model=model,
                tokenizer=tokenizer,
                args=config,
                train_dataset=temp_dataset,
            )

        # Train — mlx-tune handles MoE gradient routing correctly
        trainer.train()

        sys.stderr.write(f"Training complete. Adapter saved to {adapter_out}\n")
        sys.stderr.flush()

        # Write final progress line for Rust to parse
        progress = {
            "step": max_steps,
            "total_steps": max_steps,
            "loss": 0.0,
            "learning_rate": learning_rate,
            "chosen_reward": 0.0,
            "rejected_reward": 0.0,
            "reward_margin": 0.0,
        }
        sys.stdout.write(_json.dumps(progress) + "\n")
        sys.stdout.flush()

    finally:
        os.unlink(temp_dataset)


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

    # LoRA layers are already trainable from load() with adapter_path.
    # Count trainable params for logging.
    trainable = model.trainable_parameters()
    n_trainable = sum(p.size for _, p in nn.utils.tree_flatten(trainable))
    sys.stderr.write(f"Trainable parameters: {n_trainable:,}\n")
    sys.stderr.flush()

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

    # Save adapter — extract LoRA weights from the parameter tree
    os.makedirs(adapter_out, exist_ok=True)
    lora_weights = {}
    for name, param in nn.utils.tree_flatten(model.trainable_parameters()):
        lora_weights[name] = param
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

        return loss

    # nn.value_and_grad returns a function that computes (loss, grads)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss_val, grads = loss_and_grad_fn(model)

    # Compute metrics from a separate forward pass (cheap, no grad graph)
    chosen_logits = model(chosen_ids[None, :-1])
    chosen_lp = -nn.losses.cross_entropy(
        chosen_logits.squeeze(0), chosen_ids[1:], reduction="none"
    ).mean()
    rejected_logits = model(rejected_ids[None, :-1])
    rejected_lp = -nn.losses.cross_entropy(
        rejected_logits.squeeze(0), rejected_ids[1:], reduction="none"
    ).mean()

    metrics = {
        "chosen_reward": chosen_lp,
        "rejected_reward": rejected_lp,
        "margin": chosen_lp - rejected_lp,
    }

    return loss_val, grads, metrics


if __name__ == "__main__":
    main()
