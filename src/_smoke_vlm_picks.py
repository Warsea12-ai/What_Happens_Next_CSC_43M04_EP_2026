"""Smoke test: parse + basic helper validation for the 2 new VLM picks.

Does NOT load real weights (would need GPU + 8GB downloads). Validates:
  1. Each new model file parses cleanly.
  2. train_trackB.py still parses after the build_model edits.
  3. The Qwen patchify helper produces the expected shapes.
  4. The pixel_shuffle helper (InternVL) produces the expected shapes.
"""
import ast, os, sys, types, importlib.util

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# 1) AST parse all touched files
TOUCHED = [
    "models/qwen25vl_3b_video.py",
    "models/internvl25_4b_video.py",
    "train_trackB.py",
]
for rel in TOUCHED:
    path = os.path.join(ROOT, rel)
    with open(path, encoding="utf-8") as f:
        src = f.read()
    try:
        ast.parse(src)
        print(f"OK  parse  {rel}")
    except SyntaxError as e:
        print(f"FAIL parse {rel}: {e}")
        sys.exit(1)

# 2) Stub heavy deps so the module-level imports don't try to download anything,
#    then import the helpers we want to unit-test.
stub_tf = types.ModuleType("transformers")
stub_tf.Qwen2_5_VLForConditionalGeneration = type("Stub", (), {})
stub_tf.AutoModel = type("Stub", (), {})
sys.modules["transformers"] = stub_tf

stub_peft = types.ModuleType("peft")
stub_peft.LoraConfig = type("Stub", (), {})
stub_peft.get_peft_model = lambda *a, **kw: None
sys.modules["peft"] = stub_peft

# Import qwen module helpers
spec_q = importlib.util.spec_from_file_location(
    "qwen25vl_3b_video", os.path.join(ROOT, "models/qwen25vl_3b_video.py")
)
mq = importlib.util.module_from_spec(spec_q)
spec_q.loader.exec_module(mq)

import torch

# 3) Qwen patchify shape check: 1 batch × 8 frames × 3 × 224 × 224
B, T, C, H, W = 1, 8, 3, 224, 224
x = torch.randn(B, T, C, H, W)
patches, (Tp, Hp, Wp) = mq._qwen_patchify(x)
expected_num_patches = B * (T // 2) * (H // 14) * (W // 14)
expected_dim = C * 2 * 14 * 14
assert patches.shape == (expected_num_patches, expected_dim), \
    f"qwen_patchify: got {patches.shape}, want ({expected_num_patches}, {expected_dim})"
assert (Tp, Hp, Wp) == (T // 2, H // 14, W // 14), f"grid mismatch: {(Tp, Hp, Wp)}"
print(f"OK  qwen_patchify  -> patches {tuple(patches.shape)}, grid ({Tp},{Hp},{Wp})")

# 4) InternVL pixel_shuffle shape check: B × N × C, square spatial grid
spec_i = importlib.util.spec_from_file_location(
    "internvl25_4b_video", os.path.join(ROOT, "models/internvl25_4b_video.py")
)
mi = importlib.util.module_from_spec(spec_i)
spec_i.loader.exec_module(mi)

# Simulate InternViT output after CLS removal: (B, 32*32, 1024) for 448/14=32
B, side, Cv = 2, 32, 1024
xv = torch.randn(B, side * side, Cv)
y = mi._pixel_shuffle(xv, scale_factor=0.5)
# Expected: (B, (32*0.5)*(32*0.5), 1024 * 4) = (B, 16*16, 4096) = (B, 256, 4096)
expect = (B, 16 * 16, Cv * 4)
assert y.shape == expect, f"pixel_shuffle: got {y.shape}, want {expect}"
print(f"OK  pixel_shuffle  scale=0.5 -> {tuple(y.shape)} (4x token compression)")

print("\nAll smoke checks passed.")
