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

**Garde-fou self-OOM** : si le déficit de mémoire vient de NOTRE propre process
(model trop gros pour la GPU, pas un co-tenant qui squatte), attendre est un
deadlock — personne ne libérera rien. Détecté via ``_self_holds_dominant_share``
qui interroge ``nvidia-smi`` pour les compute-apps, et abandonne le wait dans ce
cas. Le caller (OOM-retry, train step skip) propage alors l'erreur, ce qui est
le bon comportement (un batch_size trop grand doit échouer visiblement, pas
boucler à l'infini).
"""
from __future__ import annotations
import os
import subprocess
import time
import torch


def free_mb() -> int:
    """MB available on the current CUDA device (after empty_cache)."""
    if not torch.cuda.is_available():
        return 0
    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info()
    return free // (1024 * 1024)


def _self_gpu_mb() -> int:
    """MB de GPU occupés par le process courant (via nvidia-smi).

    Retourne 0 si nvidia-smi indisponible ou si notre PID n'apparaît pas dans
    les compute-apps (cas rare : early init avant que CUDA ait alloué).
    """
    my_pid = os.getpid()
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == my_pid:
                return int(parts[1])
    except Exception:
        pass
    return 0


def _self_holds_dominant_share(min_free_mb: int) -> tuple[bool, int]:
    """True si notre process tient plus que (total - min_free_mb) / 2.

    Heuristique : si on monopolise plus de la moitié des MB qui devraient être
    libres pour qu'un autre tenant nous laisse passer, c'est qu'on n'est pas
    en compétition avec un co-tenant — c'est *nous* le problème. Pas la peine
    d'attendre. Renvoie aussi le self_mb pour le logging.
    """
    self_mb = _self_gpu_mb()
    if self_mb <= 0:
        return False, 0
    if not torch.cuda.is_available():
        return False, self_mb
    free, total = torch.cuda.mem_get_info()
    total_mb = total // (1024 * 1024)
    # Si > 50% du déficit nous appartient → on est dominant.
    deficit = max(min_free_mb - (free // (1024 * 1024)), 1)
    return (self_mb >= total_mb // 2) or (self_mb >= 2 * deficit), self_mb


def wait_for_gpu_memory(min_free_mb: int, poll_interval: float = 30.0,
                        max_wait: float | None = None, label: str = "") -> int:
    """Block until at least min_free_mb MB are free on the current CUDA device.

    Returns the actual free MB when satisfied. Raises TimeoutError if max_wait set
    and exceeded. Logs progress every poll_interval seconds so the tmux pane shows
    we are alive and waiting (not silently hung).

    Garde-fou self-OOM : si après vérification c'est *notre* process qui
    monopolise la VRAM (pas un co-tenant), on raise TimeoutError immédiatement
    plutôt que d'attendre indéfiniment — il n'y a personne qui libérera.
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

        # Self-OOM detection : si on tient la VRAM nous-mêmes, le wait ne servira à rien.
        is_self, self_mb = _self_holds_dominant_share(min_free_mb)
        if is_self:
            elapsed = int(time.time() - start)
            print(f"  [gpu-wait{tag}] free={cur} MB < {min_free_mb} MB, mais *nous* tenons "
                  f"{self_mb} MB → pas de co-tenant à attendre. Abandon (waited {elapsed}s).",
                  flush=True)
            raise TimeoutError(
                f"Self-OOM detected: free={cur} MB < {min_free_mb} MB needed, "
                f"current process holds {self_mb} MB. Reduce batch_size / model size."
            )

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
