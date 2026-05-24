"""Smoke test for models/vjepa2_llama3_vlm.py — syntax + token-pool math only.

Does NOT load V-JEPA or LLaMA weights (would need ~20 GB). Validates that:
  1. The module parses cleanly.
  2. The _pool_vjepa_tokens helper produces the expected shapes.
  3. train_trackB.py still parses after the edits.
"""
import ast, sys, types, importlib.util, os

ROOT = os.path.dirname(os.path.abspath(__file__))

# 1) AST parse both files
for path in ("models/vjepa2_llama3_vlm.py", "train_trackB.py"):
    full = os.path.join(ROOT, path)
    with open(full, encoding="utf-8") as f:
        src = f.read()
    try:
        ast.parse(src)
        print(f"OK  parse  {path}")
    except SyntaxError as e:
        print(f"FAIL parse {path}: {e}")
        sys.exit(1)

# 2) Token-pool math — stub transformers.VJEPA2Model so import does not download.
sys.path.insert(0, ROOT)
stub_tf = types.ModuleType("transformers")
stub_tf.VJEPA2Model = type("Stub", (), {})
sys.modules["transformers"] = stub_tf

spec = importlib.util.spec_from_file_location(
    "vjepa2_llama3_vlm", os.path.join(ROOT, "models/vjepa2_llama3_vlm.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import torch
B, T_v, side, D = 2, 8, 14, 1408
x = torch.randn(B, T_v * side * side, D)

cases = [
    ("none",       8 * 14 * 14),
    ("spatial2x",  8 *  7 *  7),
    ("spatial4x",  8 *  4 *  4),
    ("temporal2x", 4 * 14 * 14),
    ("both",       4 *  7 *  7),
]
for mode, expected in cases:
    y = mod._pool_vjepa_tokens(x, mode)
    assert y.shape == (B, expected, D), f"{mode}: got {tuple(y.shape)}, want (B, {expected}, D)"
    print(f"OK  pool   {mode:11s} -> {tuple(y.shape)}")

# 3) Temporal upsampler sanity
T_in = 4
v = torch.randn(B, T_in, 3, 224, 224)
v_up = mod._linear_upsample(v, mod._VJEPA_TARGET_FRAMES)
assert v_up.shape == (B, mod._VJEPA_TARGET_FRAMES, 3, 224, 224), v_up.shape
print(f"OK  upsamp 4 -> {mod._VJEPA_TARGET_FRAMES} frames, shape {tuple(v_up.shape)}")

print("\nAll smoke checks passed.")
