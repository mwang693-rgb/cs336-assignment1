import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .AdamW import AdamW
from .Transformer_lm import Transformer_lm
from .checkpointing import load_checkpoint, save_checkpoint
from .cross_entropy import cross_entropy
from .data_loading import data_loading
from .gradient_clipping import clip_gradient_norm
from .learning_rate_schedule import get_lr_cosine_schedule


def open_tokens(path):
    if path.suffix.lower() == ".npy":
        return np.load(path, mmap_mode="r", allow_pickle=False)
    return np.memmap(path, dtype="<u2", mode="r")


@torch.no_grad()
def validation_loss(model, tokens, batch_size, device, batches):
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = data_loading(tokens, batch_size, model.context_length, device)
        losses.append(cross_entropy(model(x), y))
    model.train()
    return torch.stack(losses).mean().item()


def main():
    parser = argparse.ArgumentParser(description="Train the CS336 Transformer LM.")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--valid-data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--rope-theta", type=float, default=10000)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=1000)
    args = parser.parse_args()

    warmup = args.warmup_steps if args.warmup_steps is not None else min(100, args.steps // 10)
    if not 0 <= warmup <= args.steps or min(args.steps, args.eval_every, args.eval_batches, args.save_every) < 1:
        parser.error("steps, eval and save intervals must be positive; warmup must be within steps")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_tokens, valid_tokens = open_tokens(args.train_data), open_tokens(args.valid_data)
    for tokens in (train_tokens, valid_tokens):
        if tokens.ndim != 1 or len(tokens) <= args.context_length or tokens.dtype.kind not in "iu":
            parser.error("each dataset must be a 1D integer array longer than the context")

    model = Transformer_lm(
        d_model=args.d_model,
        num_head=args.num_heads,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layer=args.num_layers,
        rope_theta=args.rope_theta,
    ).to(args.device)
    optimizer = AdamW(
        model.parameters(), lr=args.max_lr, betas=(args.beta1, args.beta2),
        eps=args.eps, weight_decay=args.weight_decay,
    )
    start = load_checkpoint(args.resume, model, optimizer) if args.resume else 0

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if args.log is not None:
        args.log.parent.mkdir(parents=True, exist_ok=True)
    model.train()
    started = time.perf_counter()
    for iteration in range(start, args.steps):
        lr = get_lr_cosine_schedule(iteration, args.max_lr, args.min_lr, warmup, args.steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = data_loading(train_tokens, args.batch_size, args.context_length, args.device)
        optimizer.zero_grad()
        loss = cross_entropy(model(x), y)
        loss.backward()
        clip_gradient_norm(model.parameters(), args.max_grad_norm)
        optimizer.step()

        completed = iteration + 1
        if completed % args.eval_every == 0 or completed == args.steps:
            metrics = {
                "step": completed,
                "lr": lr,
                "train_loss": loss.detach().item(),
                "valid_loss": validation_loss(model, valid_tokens, args.batch_size, args.device, args.eval_batches),
                "elapsed_seconds": time.perf_counter() - started,
            }
            line = json.dumps(metrics)
            print(line, flush=True)
            if args.log is not None:
                with args.log.open("a", encoding="utf-8") as log:
                    log.write(line + "\n")

        if completed % args.save_every == 0 or completed == args.steps:
            save_checkpoint(model, optimizer, completed, args.checkpoint)


if __name__ == "__main__":
    main()
