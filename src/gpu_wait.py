"""gpu_wait.py — Wait for GPU memory to free up instead of crashing on OOM.

When the cluster GPU is shared with another user who's running their own training,
our allocations can OOM as soon as the other process spikes its usage. Crashing the
whole training (and losing the multi-GB checkpoint progress) is wasteful. Instead:

  1. wait_for_gpu_memory(min_free_mb)
       Block until ≥min_free_mb is available on the current CUDA device. Used
       before heavy allocations: model.to(device), .from_pretrained, optimizer init.

  2. run_with_oom_retry(fn, ..., min_free_mb=N)
       Call fn(); on torch.cuda.OutOfMemoryError, empty cache, wait for memory,
       retry. Used around the call sites above + per-batch training step.

The training script then becomes resilient to co-tenants: instead of "OOM → crash",
we get "OOM → empty cache → sleep until co-tenant frees memory → resume from
exactly where we were". The only externally visible effect is paused iterations
during the wait.
"""
from __future__ import annotations
import time
import torch


def free_mb() -> int:
    """MB available on the current CUDA device (after empty_cache)."""
    if not torch.cuda.is_available():
        return 0
    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info()
    return free // (1024 * 1024)


def wait_for_gpu_memory(min_free_mb: int, poll_interval: float = 30.0,
                        max_wait: float | None = None, label: str = "") -> int:
    """Block until at least min_free_mb MB are free on the current CUDA device.

    Returns the actual free MB when satisfied. Raises TimeoutError if max_wait set
    and exceeded. Logs progress every poll_interval seconds so the tmux pane shows
    we are alive and waiting (not silently hung).
    """
    if not torch.cuda.is_available():
        return 0
    tag = f":{label}" if label else ""
    start = time.time()
    last_log = 0.0
    while True:
        cur = free_mb()
        if cur >= min_free_mb:
            elapsed = int(time.time() - start)
            if elapsed > 0:
                print(f"  [gpu-wait{tag}] OK, free={cur} MB ≥ {min_free_mb} MB after {elapsed}s wait",
                      flush=True)
            return cur
        elapsed = time.time() - start
        if max_wait is not None and elapsed > max_wait:
            raise TimeoutError(
                f"GPU memory wait timed out after {elapsed:.0f}s: "
                f"free={cur} MB, needed {min_free_mb} MB"
            )
        # Log every poll_interval so the wait is visible; first call always logs.
        if elapsed - last_log >= poll_interval - 0.5 or last_log == 0.0:
            print(f"  [gpu-wait{tag}] free={cur} MB < {min_free_mb} MB needed; "
                  f"sleeping {poll_interval:.0f}s (waited {int(elapsed)}s)...", flush=True)
            last_log = elapsed
        time.sleep(poll_interval)


def run_with_oom_retry(fn, *args, min_free_mb: int = 1024,
                       max_retries: int = 200, poll_interval: float = 30.0,
                       label: str = "", **kwargs):
    """Call fn(*args, **kwargs). If torch.cuda.OutOfMemoryError, empty cache, wait
    for at least min_free_mb to become available, and retry. Returns fn's result.

    Default max_retries=200 with poll_interval=30s gives ~100 minutes of total
    patience — enough to outwait most co-tenant training cycles, but bounded so a
    truly dead GPU doesn't hang forever.
    """
    tag = f":{label}" if label else ""
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except torch.cuda.OutOfMemoryError as e:
            if attempt >= max_retries:
                print(f"  [oom-retry{tag}] exhausted {max_retries} retries; reraising",
                      flush=True)
                raise
            torch.cuda.empty_cache()
            cur = free_mb()
            msg = e.args[0][:120].replace("\n", " ") if e.args else "?"
            print(f"  [oom-retry{tag}] attempt {attempt + 1}/{max_retries}: OOM "
                  f"(free={cur} MB). {msg}. Waiting for ≥{min_free_mb} MB...",
                  flush=True)
            try:
                wait_for_gpu_memory(min_free_mb, poll_interval=poll_interval,
                                    max_wait=poll_interval * max_retries, label=label)
            except TimeoutError:
                # Don't reraise here — let the next fn() attempt fail and exhaust retries naturally.
                pass
