"""LoRA (Low-Rank Adaptation) utilities for frozen transformer encoders.

Wraps nn.Linear layers in the target modules with:
    W_out = W_frozen @ x + alpha/r * B @ A @ x
where A ~ N(0, 1/r), B = 0 (so the adapted layer starts as identity).

Usage:
    apply_lora(model, target_modules=["q_proj", "v_proj"], rank=8, alpha=16)
    lora_params = get_lora_parameters(model)
    # Only optimize lora_params in the optimizer.

Reference: Hu et al. 2021, LoRA: Low-Rank Adaptation of Large Language Models.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps an existing frozen nn.Linear with a low-rank adapter."""

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        assert rank > 0
        in_features  = linear.in_features
        out_features = linear.out_features

        self.linear = linear
        self.rank   = rank
        self.scale  = alpha / rank

        # A: down-projection; B: up-projection (zero-init → identity start)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        delta = (x @ self.lora_A.T) @ self.lora_B.T
        return base + self.scale * delta


def _replace_linear(module: nn.Module, name: str, target_suffixes: list[str],
                    rank: int, alpha: float) -> None:
    """Recursively replace matching nn.Linear layers with LoRALinear."""
    for child_name, child in list(module.named_children()):
        full_name = f"{name}.{child_name}" if name else child_name
        if isinstance(child, nn.Linear) and any(
            child_name.endswith(s) for s in target_suffixes
        ):
            setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha))
        else:
            _replace_linear(child, full_name, target_suffixes, rank, alpha)


def apply_lora(
    model: nn.Module,
    target_modules: list[str],
    rank: int = 8,
    alpha: float = 16.0,
) -> int:
    """Replace all matching linear layers with LoRALinear adapters.

    Returns the number of adapters inserted.
    """
    _replace_linear(model, "", target_modules, rank, alpha)
    count = sum(1 for m in model.modules() if isinstance(m, LoRALinear))
    print(f"LoRA: inserted {count} adapters (rank={rank}, alpha={alpha}, "
          f"target={target_modules})")
    return count


def get_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return only the LoRA adapter parameters (lora_A + lora_B)."""
    params = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            params.extend([m.lora_A, m.lora_B])
    return params


def count_lora_params(model: nn.Module) -> int:
    return sum(p.numel() for p in get_lora_parameters(model))
