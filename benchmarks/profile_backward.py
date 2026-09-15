# Copyright © 2025 Apple Inc.
"""
Profile the backward pass of the default training loss.

Measures forward, backward, and optimizer-update time separately for
`default_loss` (cross entropy) so we can see how much of the step the
CE backward dominates.

Runs multiple independent profiling rounds to account for cold starts,
kernel compilation, allocator warmup, and run-to-run variance.

Usage:
    python3 -m benchmarks.profile_backward --model Qwen/Qwen3-0.6B
    python3 -m benchmarks.profile_backward --model Qwen/Qwen3-0.6B --runs 5 --iters 10

Notes:
  - Uses random token batches (no dataset needed).
  - Reports per-phase ms (min/median/std across runs) and the backward
    fraction of total step time.
"""

import argparse
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers

from mlx_lm.tuner.trainer import default_loss
from mlx_lm.utils import load


def make_random_batch(batch_size: int, seq_length: int, vocab_size: int):
    batch = mx.random.randint(0, vocab_size, shape=(batch_size, seq_length))
    lengths = mx.array([(0, seq_length)] * batch_size, dtype=mx.int32)
    return batch, lengths


def profile_once(model, batch, lengths, optimizer, iters, warmup):
    """Run one profiling round: warmup then time each phase."""
    loss_value_and_grad = nn.value_and_grad(model, default_loss)

    # Warmup: kernel compilation, allocator, Metal pipeline state
    for _ in range(warmup):
        (lvalue, toks), grad = loss_value_and_grad(model, batch, lengths)
        optimizer.update(model, grad)
        mx.eval(model.state, grad)

    # Forward + loss
    t_fwd = 0.0
    for _ in range(iters):
        tic = time.perf_counter()
        lvalue, toks = default_loss(model, batch, lengths)
        mx.eval(lvalue, toks)
        t_fwd += time.perf_counter() - tic

    # Full value_and_grad (fwd + bwd)
    t_vg = 0.0
    lvalue = mx.array(0.0)
    toks = mx.array(0)
    grad = None
    for _ in range(iters):
        tic = time.perf_counter()
        (lvalue, toks), grad = loss_value_and_grad(model, batch, lengths)
        mx.eval(lvalue, toks, grad)
        t_vg += time.perf_counter() - tic

    # Optimizer update only
    t_opt = 0.0
    for _ in range(iters):
        tic = time.perf_counter()
        optimizer.update(model, grad)
        mx.eval(model.state)
        t_opt += time.perf_counter() - tic

    return {
        "fwd_ms": 1000 * t_fwd / iters,
        "bwd_ms": 1000 * (t_vg - t_fwd) / iters,
        "opt_ms": 1000 * t_opt / iters,
        "total_ms": 1000 * (t_vg + t_opt) / iters,
        "loss": float(lvalue) if lvalue is not None else 0.0,
    }


def summarize(values):
    """Return (min, median, stdev) for a list of floats."""
    if len(values) < 2:
        v = values[0]
        return v, v, 0.0
    return min(values), statistics.median(values), statistics.stdev(values)


def fmt_stat(label, values):
    lo, med, sd = summarize(values)
    spread = f"±{sd:.2f}" if sd > 0 else ""
    return f"{label:20s} {med:8.2f} ms  (min {lo:.2f}{spread})"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "-m", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--iters", type=int, default=10,
                        help="Timed iterations per run")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup iterations per run")
    parser.add_argument("--runs", type=int, default=5,
                        help="Independent profiling rounds")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    mx.random.seed(0)

    loaded = load(
        args.model,
        return_config=True,
        trust_remote_code=args.trust_remote_code,
    )
    if len(loaded) != 3:
        raise ValueError("load() with return_config=True must return 3 items")
    model, tokenizer, config = loaded
    model.train()
    vocab_size = config["vocab_size"]

    batch, lengths = make_random_batch(args.batch_size, args.seq_length, vocab_size)
    optimizer = optimizers.Adam(learning_rate=1e-5)

    dev = mx.default_device()
    print(f"Model:   {args.model}")
    print(f"Batch:   {args.batch_size} x {args.seq_length}, vocab={vocab_size}")
    print(f"Device:  {'Metal' if dev.type == mx.DeviceType.gpu else 'CPU'}")
    print(f"Runs:    {args.runs} x {args.iters} iters  (warmup {args.warmup})")
    print()

    fwd_list, bwd_list, opt_list, total_list = [], [], [], []
    loss = 0.0

    for run in range(args.runs):
        try:
            stats = profile_once(model, batch, lengths, optimizer, args.iters, args.warmup)
        except Exception as e:
            print(f"Run {run + 1} failed: {e}")
            raise SystemExit(1)

        fwd_list.append(stats["fwd_ms"])
        bwd_list.append(stats["bwd_ms"])
        opt_list.append(stats["opt_ms"])
        total_list.append(stats["total_ms"])
        loss = stats["loss"]

        # Reset peak memory tracker between runs so each run is independent
        mx.reset_peak_memory()

        print(f"  run {run + 1}/{args.runs}:  "
              f"fwd {stats['fwd_ms']:7.2f}  bwd {stats['bwd_ms']:7.2f}  "
              f"opt {stats['opt_ms']:7.2f}  total {stats['total_ms']:7.2f} ms")

    print()
    print(fmt_stat("Forward + loss", fwd_list))
    print(fmt_stat("Backward", bwd_list))
    print(fmt_stat("Optimizer", opt_list))
    print(fmt_stat("Total step", total_list))
    print()

    bwd_med = statistics.median(bwd_list)
    total_med = statistics.median(total_list)
    print(f"Loss:                    {loss:.4f}")
    print(f"Backward fraction:       {bwd_med / total_med * 100:.1f}%")
    print(f"Peak memory (last run):  {mx.get_peak_memory() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
