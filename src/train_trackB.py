"""
Track B training script — pretrained models, fine-tuning with differential LR.

Key differences from train_trackA.py:
  - Differential learning rate: backbone gets lr * backbone_lr_scale (e.g. 0.1)
    so pretrained weights are only gently updated while the new head learns faster
  - Longer warmup (10 epochs) to avoid corrupting pretrained weights early on
  - Same augmentation pipeline (flip, color jitter, RandomResizedCrop)

Run from src/::

    uv run python train_trackB.py experiment=track_B_swin
"""

from __future__ import annotations

# ── Reduce CUDA allocator fragmentation ──────────────────────────────────────
# Several tight-VRAM runs (Qwen2-VL-2B at 16GB) OOM in Adam.step() with
# hundreds of MB reserved-but-unallocated. expandable_segments lets the
# allocator grow segments dynamically instead of fragmenting. Must be set
# BEFORE `import torch` to take effect.
import os as _os
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── Global flash_attn availability workaround ────────────────────────────────
# Several trust_remote_code models in this repo (InternVL2.5, InternVideo2,
# Qwen2.5-VL, …) instantiate Qwen2ForCausalLM directly, triggering
# transformers' attention auto-detection. On shared hosts where flash_attn
# is half-installed (transitive ms-swift artefact), find_spec("flash_attn")
# raises ValueError: flash_attn.__spec__ is None instead of returning None.
# Patch find_spec to report flash_attn as missing; transformers then falls
# back to sdpa/eager cleanly. Must run BEFORE any model.from_pretrained.
import importlib.util as _il_util
if not getattr(_il_util, "_trk_flash_attn_patched", False):
    _ORIG_FIND_SPEC = _il_util.find_spec

    def _patched_find_spec(name, *args, **kwargs):
        if name == "flash_attn":
            return None
        return _ORIG_FIND_SPEC(name, *args, **kwargs)

    _il_util.find_spec = _patched_find_spec
    _il_util._trk_flash_attn_patched = True

# ── transformers compat shim ─────────────────────────────────────────────────
# OpenGVLab/InternVideo2-Stage2_6B (and other trust_remote_code models from
# the transformers 4.x era) import these three helpers from
# transformers.modeling_utils. In 5.x they were either relocated to
# transformers.pytorch_utils or removed outright. We restore them under the
# expected path so the custom code can `from transformers.modeling_utils
# import (apply_chunking_to_forward, find_pruneable_heads_and_indices,
# prune_linear_layer)` without raising ImportError.
def _install_transformers_compat_shim():
    import transformers.modeling_utils as _tmu
    import torch as _torch
    import torch.nn as _nn

    # 1) apply_chunking_to_forward — still in transformers.pytorch_utils on 5.x
    if not hasattr(_tmu, "apply_chunking_to_forward"):
        try:
            from transformers.pytorch_utils import apply_chunking_to_forward as _acf
            _tmu.apply_chunking_to_forward = _acf
        except ImportError:
            def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
                if chunk_size <= 0 or not input_tensors or not isinstance(input_tensors[0], _torch.Tensor):
                    return forward_fn(*input_tensors)
                num_chunks = (input_tensors[0].shape[chunk_dim] + chunk_size - 1) // chunk_size
                input_chunks = tuple(
                    t.chunk(num_chunks, dim=chunk_dim) if isinstance(t, _torch.Tensor) else (t,) * num_chunks
                    for t in input_tensors
                )
                output_chunks = [forward_fn(*chunk) for chunk in zip(*input_chunks)]
                return _torch.cat(output_chunks, dim=chunk_dim)
            _tmu.apply_chunking_to_forward = apply_chunking_to_forward

    # 2) prune_linear_layer — still in transformers.pytorch_utils on 5.x
    if not hasattr(_tmu, "prune_linear_layer"):
        try:
            from transformers.pytorch_utils import prune_linear_layer as _pll
            _tmu.prune_linear_layer = _pll
        except ImportError:
            def prune_linear_layer(layer, index, dim=0):
                index = index.to(layer.weight.device)
                W = layer.weight.index_select(dim, index).clone().detach()
                b = None
                if layer.bias is not None:
                    b = layer.bias.clone().detach() if dim == 1 else layer.bias[index].clone().detach()
                new_size = list(layer.weight.size())
                new_size[dim] = len(index)
                new_layer = _nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
                new_layer.weight.requires_grad = False
                new_layer.weight.copy_(W.contiguous())
                new_layer.weight.requires_grad = True
                if b is not None:
                    new_layer.bias.requires_grad = False
                    new_layer.bias.copy_(b.contiguous())
                    new_layer.bias.requires_grad = True
                return new_layer
            _tmu.prune_linear_layer = prune_linear_layer

    # 3) find_pruneable_heads_and_indices — removed in 5.x entirely.
    # Local re-implementation matching the original transformers behaviour.
    if not hasattr(_tmu, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            mask = _torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = _torch.arange(len(mask))[mask].long()
            return heads, index
        _tmu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    # 4) tokenization helpers removed from transformers.tokenization_utils in 5.x.
    # InternVideo2 modeling code does:
    #   from transformers.tokenization_utils import (PreTrainedTokenizer,
    #       _is_control, _is_punctuation, _is_whitespace)
    # PreTrainedTokenizer still lives there; the three underscore helpers don't.
    # Re-add canonical implementations (lifted from historical transformers).
    import transformers.tokenization_utils as _tok
    import unicodedata as _ud

    if not hasattr(_tok, "_is_control"):
        def _is_control(char):
            if char in ("\t", "\n", "\r"):
                return False
            return _ud.category(char).startswith("C")
        _tok._is_control = _is_control

    if not hasattr(_tok, "_is_punctuation"):
        def _is_punctuation(char):
            cp = ord(char)
            if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
                return True
            return _ud.category(char).startswith("P")
        _tok._is_punctuation = _is_punctuation

    if not hasattr(_tok, "_is_whitespace"):
        def _is_whitespace(char):
            if char in (" ", "\t", "\n", "\r"):
                return True
            return _ud.category(char) == "Zs"
        _tok._is_whitespace = _is_whitespace

try:
    _install_transformers_compat_shim()
except Exception as _e:
    # Don't take down the whole training script if the shim can't install.
    print(f"[shim] transformers compat shim install failed: {_e}")

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from models.cnn_temporal import CNNTemporal
from models.swin3d_finetune import VideoSwinFinetune
from models.vit_temporal import ViTTemporal
from models.videomae import VideoMAEClassifier
from models.motion_videomae import MotionVideoMAE
from models.frozen_videomae import FrozenVideoMAELarge
from models.qwen_vl_video import QwenVLVideo
from models.dynamic_videomae import DynamicVideoMAE
from models.frame_pair_net import FramePairNet
from models.dual_encoder import DualEncoder
from models.bidirectional_frame_pair import BidirectionalFramePairNet
from models.qwen_temporal_attn import QwenTemporalAttn
from models.videomae_lora import VideoMAELoRA
from models.videomaev2_giant import VideoMAEv2Giant
from models.timesformer_ssv2 import TimesFormerSSv2
from models.hiera_huge_wrapper import HieraHuge
from models.qwen_lora import QwenVLLoRA
from models.videomae_temporal_head import VideoMAETemporalHead
from models.videomae_domain_adapted import VideoMAEDomainAdapted
from models.videomae_pair_attn import VideoMAEPairAttn
from models.videomae_cross_attn_pairs import VideoMAECrossAttnPairs
from models.videomae_multiscale import VideoMAEMultiScale
from models.videomae_pairwise_ranking import VideoMAEPairwiseRanking
from models.dinov3_temporal import DINOv3Temporal
from models.videomae_large import VideoMAELarge
from models.vjepa2_head import VJEPA2Head
from models.internvl2_classifier import InternVL2Classifier
from models.internvit6b_temporal import InternViT6BTemporal
from models.internvit6b_pairwise import InternViT6BPairwise
from models.vjepa2_variants import VJEPA2Variant
from models.vjepa2_vitl_dup import VJEPA2ViTLDup
from models.internvideo2 import InternVideo2
from models.internvideo2_pairs import InternVideo2Pairs
from models.timesformer_head import TimesformerHead
from models.llava_mistral_video import LlavaMistralVideo
from models.vjepa2_llama3_vlm import VJEPA2Llama3VLM
from models.qwen25vl_3b_video import Qwen25VL3BVideo
from models.internvl25_4b_video import InternVL25_4BVideo
from models.videomae_multiscale_lora_temporal import VideoMAEMultiScaleLoRATemporal
# Non-transformer CNN baselines (added for ensemble diversity) — lazy to avoid
# pulling torchvision/pytorchvideo if not requested.
try:
    from models.R2Plus1D import R2Plus1D
except Exception as _e_r2:
    R2Plus1D = None
    print(f"[lazy-import] R2Plus1D unavailable ({_e_r2.__class__.__name__})")
try:
    from models.X3D import X3D
except Exception as _e_x3d:
    X3D = None
    print(f"[lazy-import] X3D unavailable ({_e_x3d.__class__.__name__})")
try:
    from models.mvit_v2_ssv2 import MViTV2SSv2
except Exception as _e_mvit:
    MViTV2SSv2 = None
    print(f"[lazy-import] MViTV2SSv2 unavailable ({_e_mvit.__class__.__name__})")
try:
    from models.xclip_video import XCLIPVideo
except Exception as _e_xclip:
    XCLIPVideo = None
    print(f"[lazy-import] XCLIPVideo unavailable ({_e_xclip.__class__.__name__})")
from utils import build_transforms, log_wandb_diagnostics, measure_grad_norm, set_seed, split_train_val
from gpu_wait import wait_for_gpu_memory, run_with_oom_retry, free_mb as gpu_free_mb

try:
    from tqdm.auto import tqdm
except ImportError:
    # Defensive fallback: tqdm is in transformers' deps so this should never trigger,
    # but if it does the loop still runs (silent, no bar).
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else iter([])

import os
import time
try:
    import wandb
    _WANDB_OK = True
except Exception as _wandb_err:
    print(f"[wandb] import failed ({_wandb_err}) — logging désactivé pour ce run")
    _WANDB_OK = False
    class _WandbStub:
        run = type("R", (), {"summary": {}})()
        def __getattr__(self, _name):
            return lambda *a, **kw: None
    wandb = _WandbStub()


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


# ── augmentations (same as train_trackA) ──────────────────────────────────────

@torch.no_grad()
def color_jitter_video(clips: torch.Tensor, p: float = 0.8, strength: float = 0.4) -> torch.Tensor:
    B, T, C, H, W = clips.shape
    device, dtype = clips.device, clips.dtype

    def _factor():
        return 1 + (torch.rand(B, 1, 1, 1, 1, device=device, dtype=dtype) * 2 - 1) * strength

    brightness, contrast, saturation = _factor(), _factor(), _factor()
    v = clips * brightness
    mean_per_video = v.mean(dim=(1, 3, 4), keepdim=True)
    v = (v - mean_per_video) * contrast + mean_per_video
    if C == 3:
        gray = (0.299 * v[:, :, 0] + 0.587 * v[:, :, 1] + 0.114 * v[:, :, 2]).unsqueeze(2)
        v = (v - gray) * saturation + gray
    v = v.clamp(0, 1)
    apply_mask = torch.rand(B, 1, 1, 1, 1, device=device) < p
    return torch.where(apply_mask, v, clips)


@torch.no_grad()
def gaussian_blur_video(
    clips: torch.Tensor, p: float = 0.3,
    sigma_range: Tuple[float, float] = (0.1, 1.5), kernel_size: int = 5,
) -> torch.Tensor:
    B, T, C, H, W = clips.shape
    device, dtype = clips.device, clips.dtype
    half = kernel_size // 2
    sigmas = torch.empty(B, device=device, dtype=dtype).uniform_(*sigma_range)
    x = torch.arange(kernel_size, device=device, dtype=dtype) - half
    g1 = torch.exp(-(x[None, :] ** 2) / (2 * sigmas[:, None] ** 2))
    g1 = g1 / g1.sum(dim=1, keepdim=True)
    kernel2d = g1[:, :, None] * g1[:, None, :]
    kernel = kernel2d[:, None].expand(B, C, kernel_size, kernel_size).reshape(B * C, 1, kernel_size, kernel_size)
    x_in = clips.permute(1, 0, 2, 3, 4).reshape(T, B * C, H, W)
    blurred = F.conv2d(x_in, kernel, padding=half, groups=B * C)
    blurred = blurred.reshape(T, B, C, H, W).permute(1, 0, 2, 3, 4).contiguous()
    apply_mask = torch.rand(B, 1, 1, 1, 1, device=device) < p
    return torch.where(apply_mask, blurred, clips)


@torch.no_grad()
def label_aware_horizontal_flip(
    clips: torch.Tensor, labels: torch.Tensor,
    forbidden_labels: torch.Tensor, p: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = clips.shape[0]
    device = clips.device
    is_forbidden = torch.isin(labels, forbidden_labels)
    should_flip = (torch.rand(B, device=device) < p) & (~is_forbidden)
    flipped = torch.flip(clips, dims=[-1])
    new_clips = torch.where(should_flip.view(B, 1, 1, 1, 1), flipped, clips)
    return new_clips, labels


@torch.no_grad()
def label_aware_temporal_reverse(
    clips: torch.Tensor, labels: torch.Tensor,
    reverse_lookup: torch.Tensor, p: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = clips.shape[0]
    device = clips.device
    mapped_labels = reverse_lookup[labels]
    has_mapping = mapped_labels >= 0
    should_rev = (torch.rand(B, device=device) < p) & has_mapping
    reversed_ = torch.flip(clips, dims=[1])
    new_clips = torch.where(should_rev.view(B, 1, 1, 1, 1), reversed_, clips)
    new_labels = torch.where(should_rev, mapped_labels, labels)
    return new_clips, new_labels


def _build_label_lookup(
    label_map: Dict[int, int], num_classes: int, device: torch.device,
) -> torch.Tensor:
    lookup = torch.full((num_classes,), -1, dtype=torch.long, device=device)
    for src, dst in label_map.items():
        lookup[src] = dst
    return lookup


@torch.no_grad()
def video_augment(
    clips: torch.Tensor, *,
    color_prob: float = 0.8, color_strength: float = 0.4,
    blur_prob: float = 0.1,
    input_normalized: bool = True, output_normalized: bool = True,
    mean: torch.Tensor | None = None, std: torch.Tensor | None = None,
) -> torch.Tensor:
    device, dtype = clips.device, clips.dtype
    m = (mean if mean is not None else _IMAGENET_MEAN).to(device, dtype)
    s = (std  if std  is not None else _IMAGENET_STD).to(device, dtype)
    if input_normalized:
        clips = clips * s + m
        clips = clips.clamp(0, 1)
    clips = color_jitter_video(clips, p=color_prob, strength=color_strength)
    if blur_prob > 0:
        clips = gaussian_blur_video(clips, p=blur_prob)
    if output_normalized:
        clips = (clips - m) / s
    return clips


# ── model factory ─────────────────────────────────────────────────────────────

def build_model(cfg: DictConfig) -> nn.Module:
    # Re-install the transformers compat shim *here*, not just at module load
    # time. One of the `from models.* import …` lines above silently re-loads
    # transformers.tokenization_utils (likely a tokenizer dependency in some
    # other model wrapper) which wipes our monkey-patched _is_control etc.
    # Running the shim again right before any from_pretrained call ensures the
    # patched attributes are present when trust_remote_code modeling files
    # like OpenGVLab/InternVideo2-Stage2_6B execute their imports.
    try:
        _install_transformers_compat_shim()
    except Exception as _e:
        print(f"[shim] re-install at build_model failed: {_e}")

    name = cfg.model.name
    if name == "cnn_temporal":
        return CNNTemporal(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "resnet34")),
            pretrained=bool(cfg.model.get("pretrained", True)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            head_dim=int(cfg.model.get("head_dim", 512)),
        )
    if name == "swin3d_finetune":
        return VideoSwinFinetune(
            num_classes=int(cfg.model.num_classes),
            pretrained=bool(cfg.model.get("pretrained", True)),
            dropout=float(cfg.model.get("dropout", 0.5)),
        )
    if name == "vit_temporal":
        return ViTTemporal(
            num_classes=int(cfg.model.num_classes),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 8)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            temporal_layers=int(cfg.model.get("temporal_layers", 2)),
            temporal_heads=int(cfg.model.get("temporal_heads", 8)),
        )
    if name == "videomae":
        return VideoMAEClassifier(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "MCG-NJU/videomae-base-finetuned-ssv2")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 8)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            frames_repeat=int(cfg.model.get("frames_repeat", 4)),
            use_linear_interp=bool(cfg.model.get("use_linear_interp", True)),
        )
    if name == "motion_videomae":
        return MotionVideoMAE(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "MCG-NJU/videomae-base-finetuned-ssv2")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 8)),
            dropout=float(cfg.model.get("dropout", 0.3)),
        )
    if name == "frozen_videomae":
        return FrozenVideoMAELarge(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "MCG-NJU/videomae-large-finetuned-kinetics")),
            dropout=float(cfg.model.get("dropout", 0.5)),
            num_unfrozen_blocks=int(cfg.model.get("num_unfrozen_blocks", 0)),
            head_hidden=int(cfg.model.get("head_hidden", 512)),
        )
    if name == "qwen_vl_video":
        return QwenVLVideo(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "Qwen/Qwen2-VL-2B-Instruct")),
            dropout=float(cfg.model.get("dropout", 0.5)),
            head_hidden=int(cfg.model.get("head_hidden", 512)),
        )
    if name == "qwen2_vl_7b":
        return QwenVLVideo(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "Qwen/Qwen2-VL-7B-Instruct")),
            dropout=float(cfg.model.get("dropout", 0.4)),
            head_hidden=int(cfg.model.get("head_hidden", 1024)),
        )
    if name == "qwen25_7b":
        return QwenVLVideo(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "Qwen/Qwen2.5-VL-7B-Instruct")),
            dropout=float(cfg.model.get("dropout", 0.4)),
            head_hidden=int(cfg.model.get("head_hidden", 768)),
        )
    if name == "dynamic_videomae":
        return DynamicVideoMAE(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "MCG-NJU/videomae-large-finetuned-kinetics")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 12)),
            dropout=float(cfg.model.get("dropout", 0.3)),
        )
    if name == "frame_pair_net":
        return FramePairNet(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "Qwen/Qwen2-VL-2B-Instruct")),
            n_cross_layers=int(cfg.model.get("n_cross_layers", 2)),
            n_heads=int(cfg.model.get("n_heads", 8)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            head_hidden=int(cfg.model.get("head_hidden", 512)),
        )
    if name == "dual_encoder":
        return DualEncoder(
            num_classes=int(cfg.model.num_classes),
            videomae_backbone=str(cfg.model.get("videomae_backbone", "MCG-NJU/videomae-large-finetuned-kinetics")),
            qwen_backbone=str(cfg.model.get("qwen_backbone", "Qwen/Qwen2-VL-7B-Instruct")),
            dropout=float(cfg.model.get("dropout", 0.4)),
            head_hidden=int(cfg.model.get("head_hidden", 1024)),
        )
    if name == "bidirectional_frame_pair":
        return BidirectionalFramePairNet(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "Qwen/Qwen2-VL-2B-Instruct")),
            n_cross_layers=int(cfg.model.get("n_cross_layers", 2)),
            n_heads=int(cfg.model.get("n_heads", 8)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            head_hidden=int(cfg.model.get("head_hidden", 512)),
        )
    if name == "qwen_temporal_attn":
        return QwenTemporalAttn(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "Qwen/Qwen2-VL-2B-Instruct")),
            n_attn_layers=int(cfg.model.get("n_attn_layers", 2)),
            n_heads=int(cfg.model.get("n_heads", 8)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            head_hidden=int(cfg.model.get("head_hidden", 512)),
        )
    if name == "hiera_huge":
        return HieraHuge(
            num_classes=int(cfg.model.num_classes),
            variant=str(cfg.model.get("variant", "frozen")),
            pretrained=bool(cfg.model.get("pretrained", True)),
            lora_rank=int(cfg.model.get("lora_rank", 16)),
            lora_alpha=float(cfg.model.get("lora_alpha", 32.0)),
            head_hidden=int(cfg.model.get("head_hidden", 0)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            num_frozen_stages=int(cfg.model.get("num_frozen_stages", 3)),
            model_size=str(cfg.model.get("model_size", "huge")),
        )
    if name == "timesformer_ssv2":
        return TimesFormerSSv2(
            num_classes=int(cfg.model.num_classes),
            variant=str(cfg.model.get("variant", "frozen")),
            backbone=str(cfg.model.get("backbone", "facebook/timesformer-base-finetuned-ssv2")),
            lora_rank=int(cfg.model.get("lora_rank", 16)),
            lora_alpha=float(cfg.model.get("lora_alpha", 32.0)),
            head_hidden=int(cfg.model.get("head_hidden", 0)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            num_frames=int(cfg.model.get("num_frames", 8)),
            image_size=int(cfg.model.get("image_size", 224)),
        )
    if name == "videomaev2_giant":
        return VideoMAEv2Giant(
            num_classes=int(cfg.model.num_classes),
            variant=str(cfg.model.get("variant", "frozen")),
            checkpoint_path=str(cfg.model.get("checkpoint_path", "/Data/What_Happens_Next_CSC_43M04_EP_2026/checkpoints/videomaev2_vitg_ssv2_ft.pth")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 32)),
            lora_rank=int(cfg.model.get("lora_rank", 16)),
            lora_alpha=float(cfg.model.get("lora_alpha", 32.0)),
            head_hidden=int(cfg.model.get("head_hidden", 0)),
            dropout=float(cfg.model.get("dropout", 0.3)),
        )
    if name == "videomae_lora":
        return VideoMAELoRA(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "MCG-NJU/videomae-large-finetuned-kinetics")),
            lora_rank=int(cfg.model.get("lora_rank", 8)),
            lora_alpha=float(cfg.model.get("lora_alpha", 16.0)),
            dropout=float(cfg.model.get("dropout", 0.4)),
            head_hidden=int(cfg.model.get("head_hidden", 512)),
        )
    if name == "qwen_lora":
        return QwenVLLoRA(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "Qwen/Qwen2-VL-7B-Instruct")),
            lora_rank=int(cfg.model.get("lora_rank", 16)),
            lora_alpha=float(cfg.model.get("lora_alpha", 32.0)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            head_hidden=int(cfg.model.get("head_hidden", 768)),
        )
    if name == "videomae_temporal_head":
        return VideoMAETemporalHead(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "MCG-NJU/videomae-base-finetuned-ssv2")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 12)),
            n_temporal_layers=int(cfg.model.get("n_temporal_layers", 3)),
            n_heads=int(cfg.model.get("n_heads", 8)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            head_hidden=int(cfg.model.get("head_hidden", 1024)),
            lora_rank=int(cfg.model.get("lora_rank", 0)),
            lora_alpha=float(cfg.model.get("lora_alpha", 16.0)),
        )
    if name == "videomae_domain_adapted":
        return VideoMAEDomainAdapted(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "MCG-NJU/videomae-base-finetuned-ssv2")),
            backbone_path=str(cfg.model.get("backbone_path", "")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 12)),
            n_temporal_layers=int(cfg.model.get("n_temporal_layers", 3)),
            n_heads=int(cfg.model.get("n_heads", 8)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            head_hidden=int(cfg.model.get("head_hidden", 1024)),
        )
    if name == "videomae_pair_attn":
        return VideoMAEPairAttn(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "MCG-NJU/videomae-base-finetuned-ssv2")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 6)),
            n_temporal_layers=int(cfg.model.get("n_temporal_layers", 3)),
            n_heads=int(cfg.model.get("n_heads", 8)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            head_hidden=int(cfg.model.get("head_hidden", 1024)),
            lora_rank=int(cfg.model.get("lora_rank", 16)),
            lora_alpha=float(cfg.model.get("lora_alpha", 32.0)),
        )
    if name == "videomae_cross_attn_pairs":
        return VideoMAECrossAttnPairs(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "MCG-NJU/videomae-base-finetuned-ssv2")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 2)),
            n_temporal_layers=int(cfg.model.get("n_temporal_layers", 2)),
            n_heads=int(cfg.model.get("n_heads", 8)),
            dropout=float(cfg.model.get("dropout", 0.25)),
            head_hidden=int(cfg.model.get("head_hidden", 1024)),
            lora_rank=int(cfg.model.get("lora_rank", 0)),
            lora_alpha=float(cfg.model.get("lora_alpha", 0.0)),
            interp_mode=str(cfg.model.get("interp_mode", "aligned")),
        )
    if name == "videomae_multiscale":
        return VideoMAEMultiScale(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "MCG-NJU/videomae-base-finetuned-ssv2")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 2)),
            n_temporal_layers=int(cfg.model.get("n_temporal_layers", 2)),
            n_heads=int(cfg.model.get("n_heads", 8)),
            dropout=float(cfg.model.get("dropout", 0.25)),
            head_hidden=int(cfg.model.get("head_hidden", 1024)),
        )
    if name == "videomae_pairwise_ranking":
        return VideoMAEPairwiseRanking(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "MCG-NJU/videomae-base-finetuned-ssv2")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 2)),
            dropout=float(cfg.model.get("dropout", 0.25)),
            head_hidden=int(cfg.model.get("head_hidden", 1024)),
            rank_hidden=int(cfg.model.get("rank_hidden", 64)),
        )
    if name == "dinov3_temporal":
        return DINOv3Temporal(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "facebook/dinov3-vith16plus-pretrain-lvd1689m")),
            proj_dim=int(cfg.model.get("proj_dim", 512)),
            n_temporal_layers=int(cfg.model.get("n_temporal_layers", 4)),
            n_heads=int(cfg.model.get("n_heads", 8)),
            dropout=float(cfg.model.get("dropout", 0.25)),
            head_hidden=int(cfg.model.get("head_hidden", 512)),
            backbone_dtype=str(cfg.model.get("backbone_dtype", "float32")),
        )
    if name == "videomae_large":
        return VideoMAELarge(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "MCG-NJU/videomae-large-finetuned-kinetics")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 4)),
            dropout=float(cfg.model.get("dropout", 0.25)),
            head_hidden=int(cfg.model.get("head_hidden", 2048)),
        )
    if name == "vjepa2_head":
        return VJEPA2Head(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "facebook/vjepa2-vitg-fpc64-384-ssv2")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 32)),
            dropout=float(cfg.model.get("dropout", 0.25)),
            head_hidden=int(cfg.model.get("head_hidden", 2048)),
            drop_path_rate=float(cfg.model.get("drop_path_rate", 0.0)),
            interp_mode=str(cfg.model.get("interp_mode", "aligned")),
        )
    if name == "internvideo2":
        return InternVideo2(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "OpenGVLab/InternVideo2-Stage2_6B")),
            dropout=float(cfg.model.get("dropout", 0.3)),
            head_hidden=int(cfg.model.get("head_hidden", 2048)),
            backbone_dtype=str(cfg.model.get("backbone_dtype", "bfloat16")),
        )
    if name == "timesformer_head":
        return TimesformerHead(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "facebook/timesformer-base-finetuned-ssv2")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 0)),
            dropout=float(cfg.model.get("dropout", 0.25)),
            head_hidden=int(cfg.model.get("head_hidden", 1024)),
        )
    if name == "llava_mistral_video":
        return LlavaMistralVideo(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "llava-hf/LLaVA-NeXT-Video-7B-hf")),
            lora_rank=int(cfg.model.get("lora_rank", 16)),
            lora_alpha=float(cfg.model.get("lora_alpha", 32.0)),
            lora_dropout=float(cfg.model.get("lora_dropout", 0.05)),
            target_modules=tuple(cfg.model.get("target_modules", ("q_proj", "v_proj"))),
            head_hidden=int(cfg.model.get("head_hidden", 2048)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            use_gradient_checkpointing=bool(cfg.model.get("use_gradient_checkpointing", True)),
        )
    if name == "internvideo2_pairs":
        return InternVideo2Pairs(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "OpenGVLab/InternVideo2-Stage2-1B-224p")),
            backbone_dtype=str(cfg.model.get("backbone_dtype", "bfloat16")),
            use_lora=bool(cfg.model.get("use_lora", True)),
            lora_rank=int(cfg.model.get("lora_rank", 16)),
            lora_alpha=float(cfg.model.get("lora_alpha", 32.0)),
            perceiver_heads=int(cfg.model.get("perceiver_heads", 8)),
            n_temporal_layers=int(cfg.model.get("n_temporal_layers", 3)),
            n_heads=int(cfg.model.get("n_heads", 8)),
            dropout=float(cfg.model.get("dropout", 0.25)),
            head_hidden=int(cfg.model.get("head_hidden", 2048)),
        )
    if name == "internvl2_classifier":
        return InternVL2Classifier(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "OpenGVLab/InternVL2-8B")),
            lora_rank=int(cfg.model.get("lora_rank", 16)),
            lora_alpha=float(cfg.model.get("lora_alpha", 32.0)),
            dropout=float(cfg.model.get("dropout", 0.25)),
            num_frozen_llm_layers=int(cfg.model.get("num_frozen_llm_layers", 24)),
        )
    if name == "internvit6b_temporal":
        return InternViT6BTemporal(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "OpenGVLab/InternViT-6B-448px-V2_5")),
            proj_dim=int(cfg.model.get("proj_dim", 512)),
            n_set_layers=int(cfg.model.get("n_set_layers", 3)),
            n_heads=int(cfg.model.get("n_heads", 8)),
            dropout=float(cfg.model.get("dropout", 0.25)),
        )
    if name == "vjepa2_variants":
        return VJEPA2Variant(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "facebook/vjepa2-vitg-fpc64-384-ssv2")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 32)),
            dropout=float(cfg.model.get("dropout", 0.25)),
            head_hidden=int(cfg.model.get("head_hidden", 2048)),
            variant=str(cfg.model.get("variant", "lora")),
            lora_rank=int(cfg.model.get("lora_rank", 16)),
            lora_alpha=float(cfg.model.get("lora_alpha", 32.0)),
        )
    if name == "internvit6b_pairwise":
        return InternViT6BPairwise(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "OpenGVLab/InternViT-6B-448px-V2_5")),
            proj_dim=int(cfg.model.get("proj_dim", 512)),
            n_heads=int(cfg.model.get("n_heads", 8)),
            dropout=float(cfg.model.get("dropout", 0.25)),
        )
    if name == "videomae_multiscale_lora_temporal":
        return VideoMAEMultiScaleLoRATemporal(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "MCG-NJU/videomae-base-finetuned-ssv2")),
            n_temporal_layers=int(cfg.model.get("n_temporal_layers", 3)),
            n_heads=int(cfg.model.get("n_heads", 8)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            head_hidden=int(cfg.model.get("head_hidden", 1024)),
            lora_rank=int(cfg.model.get("lora_rank", 8)),
            lora_alpha=float(cfg.model.get("lora_alpha", 16.0)),
            scale_indices=tuple(cfg.model.get("scale_indices", (6, 9, 12))),
        )
    if name == "vjepa2_predictor":
        from models.vjepa2_predictor import VJEPA2PredictorAnticipator
        return VJEPA2PredictorAnticipator(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "facebook/vjepa2-vitg-fpc64-384-ssv2")),
            cache_dir=cfg.model.get("cache_dir", None),
            n_future_queries=int(cfg.model.get("n_future_queries", 128)),
            head_hidden=int(cfg.model.get("head_hidden", 2048)),
            dropout=float(cfg.model.get("dropout", 0.25)),
            use_lora=bool(cfg.model.get("use_lora", True)),
            lora_rank=int(cfg.model.get("lora_rank", 16)),
            lora_alpha=float(cfg.model.get("lora_alpha", 32.0)),
            lora_dropout=float(cfg.model.get("lora_dropout", 0.05)),
            lora_targets=tuple(cfg.model.get("lora_targets", ("query", "key", "value", "dense"))),
            train_predictor_full=bool(cfg.model.get("train_predictor_full", False)),
        )
    if name == "vjepa2_llama3_vlm":
        return VJEPA2Llama3VLM(
            num_classes=int(cfg.model.num_classes),
            vjepa_backbone=str(cfg.model.get("vjepa_backbone", "facebook/vjepa2-vitg-fpc64-384-ssv2")),
            llama_backbone=str(cfg.model.get("llama_backbone", "meta-llama/Llama-3.1-8B-Instruct")),
            num_frozen_vjepa_blocks=int(cfg.model.get("num_frozen_vjepa_blocks", 40)),
            vision_token_pool=str(cfg.model.get("vision_token_pool", "spatial2x")),
            lora_rank=int(cfg.model.get("lora_rank", 16)),
            lora_alpha=float(cfg.model.get("lora_alpha", 32.0)),
            lora_dropout=float(cfg.model.get("lora_dropout", 0.05)),
            lora_targets=tuple(cfg.model.get("lora_targets", ("q_proj", "v_proj"))),
            head_hidden=int(cfg.model.get("head_hidden", 2048)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            use_gradient_checkpointing=bool(cfg.model.get("use_gradient_checkpointing", True)),
        )
    if name == "qwen25vl_3b_video":
        return Qwen25VL3BVideo(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "Qwen/Qwen2.5-VL-3B-Instruct")),
            cache_dir=str(cfg.model.get("cache_dir", "/Data/.hf_cache")),
            lora_rank=int(cfg.model.get("lora_rank", 16)),
            lora_alpha=float(cfg.model.get("lora_alpha", 32.0)),
            lora_dropout=float(cfg.model.get("lora_dropout", 0.05)),
            lora_targets=tuple(cfg.model.get("lora_targets", ("q_proj", "v_proj"))),
            head_hidden=int(cfg.model.get("head_hidden", 2048)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            use_gradient_checkpointing=bool(cfg.model.get("use_gradient_checkpointing", True)),
            target_frames=int(cfg.model.get("target_frames", 8)),
        )
    if name in ("vjepa2_vitl_dup", "vjepa2_vitg_dup"):
        _cd = cfg.model.get("cache_dir", None)
        return VJEPA2ViTLDup(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "facebook/vjepa2-vitl-fpc16-256-ssv2")),
            cache_dir=(str(_cd) if _cd else None),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 0)),
            crop_size=int(cfg.model.get("crop_size", 256)),
            dropout=float(cfg.model.get("dropout", 0.1)),
            head_hidden=int(cfg.model.get("head_hidden", 512)),
            pool_heads=int(cfg.model.get("pool_heads", 8)),
            gradient_checkpointing=bool(cfg.model.get("gradient_checkpointing", False)),
        )
    if name == "vjepa2_vitl_raft":
        from models.vjepa2_vitl_raft import VJEPA2RaftDup
        _cd = cfg.model.get("cache_dir", None)
        return VJEPA2RaftDup(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "facebook/vjepa2-vitl-fpc16-256-ssv2")),
            cache_dir=(str(_cd) if _cd else None),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 0)),
            crop_size=int(cfg.model.get("crop_size", 256)),
            dropout=float(cfg.model.get("dropout", 0.1)),
            head_hidden=int(cfg.model.get("head_hidden", 512)),
            pool_heads=int(cfg.model.get("pool_heads", 8)),
            gradient_checkpointing=bool(cfg.model.get("gradient_checkpointing", False)),
            raft_iters=int(cfg.model.get("raft_iters", 12)),
        )
    if name == "internvl25_4b_video":
        return InternVL25_4BVideo(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "OpenGVLab/InternVL2_5-4B")),
            cache_dir=str(cfg.model.get("cache_dir", "/Data/.hf_cache")),
            lora_rank=int(cfg.model.get("lora_rank", 16)),
            lora_alpha=float(cfg.model.get("lora_alpha", 32.0)),
            lora_dropout=float(cfg.model.get("lora_dropout", 0.05)),
            lora_targets=tuple(cfg.model.get("lora_targets", ("q_proj", "v_proj"))),
            head_hidden=int(cfg.model.get("head_hidden", 2048)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            use_gradient_checkpointing=bool(cfg.model.get("use_gradient_checkpointing", True)),
            pixel_shuffle_scale=float(cfg.model.get("pixel_shuffle_scale", 0.5)),
        )
    if name == "R2Plus1D":
        if R2Plus1D is None:
            raise RuntimeError("R2Plus1D requested but torchvision.models.video import failed")
        return R2Plus1D(
            num_classes=int(cfg.model.num_classes),
            dropout=float(cfg.model.get("dropout", 0.5)),
        )
    if name == "X3D":
        if X3D is None:
            raise RuntimeError("X3D requested but pytorchvideo import failed")
        return X3D(
            num_classes=int(cfg.model.num_classes),
            input_clip_length=int(cfg.dataset.num_frames),
            variant=str(cfg.model.get("variant", "xs")),
            dropout=float(cfg.model.get("dropout", 0.5)),
            use_frame_diff=bool(cfg.model.get("use_frame_diff", False)),
            drop_path_rate=float(cfg.model.get("stochastic_depth", 0.2)),
        )
    if name == "mvit_v2_ssv2":
        if MViTV2SSv2 is None:
            raise RuntimeError("mvit_v2_ssv2 requested but MViT model unavailable")
        return MViTV2SSv2(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "facebook/mvit-base-finetuned-ssv2")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 0)),
            dropout=float(cfg.model.get("dropout", 0.25)),
            head_hidden=int(cfg.model.get("head_hidden", 1024)),
            interp_mode=str(cfg.model.get("interp_mode", "aligned")),
        )
    if name == "xclip_video":
        if XCLIPVideo is None:
            raise RuntimeError("xclip_video requested but XCLIPModel unavailable")
        return XCLIPVideo(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "microsoft/xclip-base-patch32-16-frames")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 0)),
            dropout=float(cfg.model.get("dropout", 0.25)),
            head_hidden=int(cfg.model.get("head_hidden", 1024)),
            interp_mode=str(cfg.model.get("interp_mode", "aligned")),
        )
    raise ValueError(f"Unknown model.name for Track B: {name!r}")


# ── optimizer with differential LR ────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    base_lr = float(cfg.training.lr)
    backbone_scale = float(cfg.training.get("backbone_lr_scale", 0.1))
    weight_decay = float(cfg.training.get("weight_decay", 1e-4))
    llrd = float(cfg.training.get("llrd_factor", 0.0))  # 0 = disabled

    # Layer-wise LR decay: each transformer block gets a geometrically smaller LR
    # moving from the top of the network toward the bottom. This prevents early
    # (more general) layers from being over-trained while adapting later layers.
    if llrd > 0 and hasattr(model, "layerwise_lr_groups"):
        param_groups = model.layerwise_lr_groups(
            head_lr=base_lr,
            backbone_lr=base_lr * backbone_scale,
            llrd=llrd,
        )
    elif hasattr(model, "head_parameters"):
        # head_parameters() always present; backbone_parameters() may return [] (frozen model)
        backbone_params = model.backbone_parameters() if hasattr(model, "backbone_parameters") else []
        lora_params = model.lora_parameters() if hasattr(model, "lora_parameters") else []
        param_groups = [{"params": model.head_parameters(), "lr": base_lr}]
        if lora_params:
            # LoRA adapters train at backbone_lr_scale (gentler than head, faster than 0)
            param_groups.append({"params": lora_params, "lr": base_lr * backbone_scale})
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": base_lr * backbone_scale})
    else:
        param_groups = [{"params": model.parameters(), "lr": base_lr}]

    opt_name = str(cfg.training.get("optimizer", "adamw")).lower()
    if opt_name == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    if opt_name == "adam":
        return torch.optim.Adam(param_groups, weight_decay=weight_decay)
    if opt_name == "sgd":
        momentum = float(cfg.training.get("momentum", 0.9))
        return torch.optim.SGD(param_groups, momentum=momentum, weight_decay=weight_decay)
    if opt_name == "lion":
        from lion_pytorch import Lion
        return Lion(param_groups, weight_decay=weight_decay)
    if opt_name == "prodigy":
        from prodigyopt import Prodigy
        # Prodigy adapte le LR effectif automatiquement : on passe lr=1.0 partout
        # et on conserve uniquement la structure des param_groups (pas le ratio).
        for g in param_groups:
            g["lr"] = 1.0
        return Prodigy(
            param_groups, lr=1.0,
            weight_decay=weight_decay,
            decouple=True,
            safeguard_warmup=True,
        )
    raise ValueError(f"Unknown optimizer: {opt_name!r}")


# ── training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_augment: bool = True,
    aug_kwargs: Dict[str, Any] | None = None,
    flip_forbidden: torch.Tensor | None = None,
    flip_prob: float = 0.5,
    reverse_lookup: torch.Tensor | None = None,
    reverse_prob: float = 0.5,
    use_temporal_map: bool = False,
    clip_grad_norm: float = 1.0,
    grad_accum_steps: int = 1,
    ema_model: "torch.optim.swa_utils.AveragedModel | None" = None,
) -> Tuple[float, float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    grad_norm_sum = 0.0
    grad_norm_n = 0
    aug_kwargs = aug_kwargs or {}
    n_batches = len(data_loader)

    optimizer.zero_grad()
    pbar = tqdm(
        enumerate(data_loader), total=n_batches, desc="train", unit="batch",
        dynamic_ncols=True, mininterval=2.0,
    )
    skipped_oom = 0
    for step, (video_batch, labels) in pbar:
        video_batch = video_batch.to(device, non_blocking=True)
        labels      = labels.to(device, non_blocking=True)

        if flip_prob > 0 and flip_forbidden is not None:
            video_batch, labels = label_aware_horizontal_flip(
                video_batch, labels, flip_forbidden, p=flip_prob,
            )
        if use_temporal_map and reverse_lookup is not None:
            video_batch, labels = label_aware_temporal_reverse(
                video_batch, labels, reverse_lookup, p=reverse_prob,
            )
        if use_augment:
            video_batch = video_augment(video_batch, **aug_kwargs)

        # OOM-resilient forward+backward. On co-tenant memory spikes we'd rather
        # skip a batch and keep training than crash and lose the multi-GB ckpt.
        try:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(video_batch)
                loss = loss_fn(logits, labels) / grad_accum_steps
            loss.backward()
        except torch.cuda.OutOfMemoryError as _oom:
            skipped_oom += 1
            optimizer.zero_grad(set_to_none=True)
            del video_batch, labels
            if "logits" in locals():
                del logits
            if "loss" in locals():
                del loss
            torch.cuda.empty_cache()
            cur = gpu_free_mb()
            tqdm.write(f"  [oom-skip] step {step + 1}/{n_batches}: free={cur} MB, "
                       f"skipping batch (#{skipped_oom} OOM in this epoch). "
                       f"Waiting for ≥1024 MB before next batch...")
            try:
                wait_for_gpu_memory(min_free_mb=1024, poll_interval=20,
                                    max_wait=600, label="train-step")
            except TimeoutError:
                pass
            continue

        is_last_batch = (step + 1 == n_batches)
        if (step + 1) % grad_accum_steps == 0 or is_last_batch:
            batch_grad_norm = float(nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=clip_grad_norm if clip_grad_norm > 0 else float("inf"),
            ).item())
            grad_norm_sum += batch_grad_norm
            grad_norm_n += 1
            try:
                optimizer.step()
                if ema_model is not None:
                    ema_model.update_parameters(model)
            except torch.cuda.OutOfMemoryError:
                # Adam step needs ~optim_state-size of fresh memory; if it OOMs the
                # gradients on this micro-batch are discarded but we keep training.
                skipped_oom += 1
                torch.cuda.empty_cache()
                tqdm.write(f"  [oom-skip] step {step + 1}: optimizer.step() OOM, "
                           f"discarded grads, waiting for memory…")
                try:
                    wait_for_gpu_memory(min_free_mb=1024, poll_interval=20,
                                        max_wait=600, label="opt-step")
                except TimeoutError:
                    pass
            optimizer.zero_grad()

        running_loss += float(loss.item()) * grad_accum_steps * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)
        # Live running averages on the bar (avoids a postfix re-format on every step)
        if total > 0 and (step + 1) % max(1, n_batches // 20) == 0:
            pbar.set_postfix(loss=f"{running_loss / total:.3f}",
                             acc=f"{correct / total:.3f}", refresh=False)

    if skipped_oom > 0:
        print(f"  [oom-skip] epoch summary: {skipped_oom} batch(es) skipped due to GPU contention")

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    grad_norm = grad_norm_sum / max(grad_norm_n, 1)
    print(f"  Train loss: {avg_loss:.4f}, accuracy: {acc:.4f}, grad_norm: {grad_norm:.3f}")
    return avg_loss, acc, grad_norm


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module, data_loader: DataLoader,
    loss_fn: nn.Module, device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    skipped_oom = 0
    pbar = tqdm(data_loader, total=len(data_loader), desc="val",
                unit="batch", dynamic_ncols=True, mininterval=2.0)
    for video_batch, labels in pbar:
        video_batch = video_batch.to(device, non_blocking=True)
        labels      = labels.to(device, non_blocking=True)
        try:
            logits = model(video_batch)
            loss = loss_fn(logits, labels)
        except torch.cuda.OutOfMemoryError:
            skipped_oom += 1
            del video_batch, labels
            if "logits" in locals():
                del logits
            torch.cuda.empty_cache()
            tqdm.write(f"  [oom-skip val] free={gpu_free_mb()} MB, skipping batch "
                       f"(#{skipped_oom}). Waiting for ≥1024 MB…")
            try:
                wait_for_gpu_memory(min_free_mb=1024, poll_interval=20,
                                    max_wait=600, label="val-step")
            except TimeoutError:
                pass
            continue
        running_loss += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)
    if skipped_oom > 0:
        print(f"  [oom-skip val] {skipped_oom} batch(es) skipped (val acc on "
              f"{total}/{len(data_loader.dataset)} samples)")
    return running_loss / max(total, 1), correct / max(total, 1)


# ── main ──────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "What_Happens_Next_CSC_43M04_EP_2026"),
        name=os.environ.get("WANDB_RUN_NAME"),
        config=OmegaConf.to_container(cfg, resolve=True),
        resume="allow",
    )
    _wandb_watch = bool(cfg.training.get("wandb_watch", True))
    _wandb_watch_freq = int(cfg.training.get("wandb_watch_freq", 500))

    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    train_dir = Path(cfg.dataset.train_dir).resolve()
    all_samples = collect_video_samples(train_dir)
    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        all_samples = all_samples[: int(max_samples)]

    train_samples, val_samples = split_train_val(
        all_samples, val_ratio=float(cfg.dataset.val_ratio), seed=int(cfg.dataset.seed),
    )

    use_rrc = bool(cfg.training.get("use_random_resized_crop", True))
    rrc_scale_min = float(cfg.training.get("rrc_scale_min", 0.6))
    rrc_scale_max = float(cfg.training.get("rrc_scale_max", 1.0))
    train_transform = build_transforms(
        is_training=True, use_imagenet_norm=True, use_random_resized_crop=use_rrc,
        rrc_scale_min=rrc_scale_min, rrc_scale_max=rrc_scale_max,
    )
    eval_transform = build_transforms(is_training=False, use_imagenet_norm=True)

    train_dataset = VideoFrameDataset(
        root_dir=train_dir, num_frames=int(cfg.dataset.num_frames),
        transform=train_transform, sample_list=train_samples,
        temporal_jitter=int(cfg.training.get("temporal_jitter", 0)),
    )
    val_dataset = VideoFrameDataset(
        root_dir=train_dir, num_frames=int(cfg.dataset.num_frames),
        transform=eval_transform, sample_list=val_samples,
    )

    # Optional class-balanced sampling : remplace shuffle=True par un
    # WeightedRandomSampler qui équilibre la fréquence d'apparition de chaque classe.
    # Effet : ~uniforme par classe (chaque classe a la même proba à chaque step),
    # défait l'imbalance train interne ET attaque le distribution shift train->val.
    _train_labels = [int(s[1]) for s in train_samples]
    _use_balanced = bool(cfg.training.get("class_balanced_sampling", False))
    if _use_balanced:
        _per_class = {}
        for _l in _train_labels:
            _per_class[_l] = _per_class.get(_l, 0) + 1
        # Poids par sample = 1/count_classe (puis normalisé)
        _sample_w = [1.0 / _per_class[_l] for _l in _train_labels]
        _sampler = WeightedRandomSampler(_sample_w, num_samples=len(_sample_w),
                                          replacement=True)
        train_loader = DataLoader(
            train_dataset, batch_size=int(cfg.training.batch_size), sampler=_sampler,
            num_workers=int(cfg.training.num_workers), pin_memory=(device.type == "cuda"),
            persistent_workers=True,
        )
        print(f"  Class-balanced sampling activé : {len(_per_class)} classes, "
              f"max/min count = {max(_per_class.values())}/{min(_per_class.values())}")
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=int(cfg.training.batch_size), shuffle=True,
            num_workers=int(cfg.training.num_workers), pin_memory=(device.type == "cuda"),
            persistent_workers=True,
        )
    val_loader = DataLoader(
        val_dataset, batch_size=int(cfg.training.batch_size), shuffle=False,
        num_workers=int(cfg.training.num_workers), pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    # Heavy backbones (DINOv3-7B, InternVideo2-6B, LLaVA-7B…) sometimes OOM at
    # model.to(device) when a co-tenant briefly spikes its GPU usage. Wait until
    # we have a reasonable margin (2 GB) before allocating, then retry on OOM —
    # crashing here would lose the multi-GB checkpoint progress.
    if device.type == "cuda":
        wait_for_gpu_memory(min_free_mb=2048, poll_interval=30, label="model-init")
    model = run_with_oom_retry(
        lambda: build_model(cfg).to(device),
        min_free_mb=2048, poll_interval=30, label="model-init",
    )
    # wandb.watch(log="all") snapshote TOUS les params (frozen inclus) → satura le
    # buffer sur backbones 1B+ et faisait disparaître les wandb.log() d'epoch.
    # log="gradients" ne hook que les params trainables (frozen → pas de grad →
    # pas de hook), donc safe même avec un backbone 6.7B frozen + head 55M trainable.
    # On ne garde "all" que pour les modèles vraiment petits.
    if _wandb_watch:
        _n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _n_total = sum(p.numel() for p in model.parameters())
        if _n_train <= 100_000_000 and _n_total <= 200_000_000:
            _mode, _freq = "all", _wandb_watch_freq
        else:
            _mode = "gradients"
            _freq = max(_wandb_watch_freq, 2000) if _n_train > 500_000_000 else _wandb_watch_freq
        wandb.watch(model, log=_mode, log_freq=_freq)
        print(f"[wandb] watch: {_n_train/1e6:.0f}M trainable / {_n_total/1e6:.0f}M total, "
              f"log='{_mode}', log_freq={_freq}")

    # Optional per-class weighting. Two forms accepted:
    #   - list of `num_classes` floats (explicit weights)
    #   - the string "auto_confusion" → boosts classes most often *under*-recalled
    #     by best_combo_v2 (from observed confusion matrix: 9→14, 11→14, 25→29,
    #     16→22, 2→22, 18→19 — see eval logs of series 8)
    _class_weights_cfg = cfg.training.get("class_weights", None)
    _loss_weight: torch.Tensor | None = None
    if _class_weights_cfg is not None:
        _num_classes = int(cfg.model.num_classes)
        if isinstance(_class_weights_cfg, str) and _class_weights_cfg == "auto_confusion":
            _cw = torch.ones(_num_classes, dtype=torch.float32, device=device)
            for _c, _boost in [(9, 1.4), (11, 1.4), (25, 1.4),
                               (16, 1.25), (2, 1.25), (18, 1.2), (3, 1.2)]:
                if _c < _num_classes:
                    _cw[_c] = _boost
            _loss_weight = _cw
            print(f"  Class weights: auto_confusion → boosted {(_cw > 1.0).sum().item()} classes")
        elif isinstance(_class_weights_cfg, str) and _class_weights_cfg == "inverse_freq":
            # Inverse frequency from train labels — rééquilibre l'apprentissage
            # vers les classes rares dans train.
            _per_class = {}
            for _l in (int(s[1]) for s in train_samples):
                _per_class[_l] = _per_class.get(_l, 0) + 1
            _cw = torch.ones(_num_classes, dtype=torch.float32, device=device)
            for _c in range(_num_classes):
                _cw[_c] = 1.0 / max(_per_class.get(_c, 1), 1)
            # Normalise pour que la moyenne soit 1 (évite que la LR effective change)
            _cw *= _num_classes / _cw.sum()
            print(f"  Class weights: inverse_freq → max/min = {_cw.max():.2f}/{_cw.min():.2f}")
            _loss_weight = _cw
        elif isinstance(_class_weights_cfg, str) and _class_weights_cfg == "val_aware":
            # Aligne la distribution training vers la distribution val_dir.
            # Ratios calculés depuis frames_new (deeplearning) :
            #   ratio[c] = freq_val[c] / freq_train[c]
            # Classes UNDER-rep en val (ratio < 0.7) → poids réduit (modèle prédit
            # moins ces classes). OVER-rep (ratio > 1.5) → poids augmenté.
            _ratios = {
                0:1.42, 1:1.02, 2:1.32, 3:1.96, 4:0.90, 5:0.90, 6:1.34, 7:1.57,
                8:0.76, 9:0.76, 10:1.77, 11:1.35, 12:2.12, 13:1.68, 14:0.98,
                15:1.19, 16:0.43, 17:0.34, 18:0.73, 19:0.53, 20:0.70, 21:1.08,
                22:0.89, 23:0.67, 24:0.58, 25:2.46, 26:2.47, 28:0.94, 29:0.54,
                30:1.27, 31:0.86, 32:1.71,
            }
            _cw = torch.ones(_num_classes, dtype=torch.float32, device=device)
            for _c, _r in _ratios.items():
                if _c < _num_classes:
                    _cw[_c] = _r
            _cw *= _num_classes / _cw.sum()
            print(f"  Class weights: val_aware → max/min = {_cw.max():.2f}/{_cw.min():.2f}")
            _loss_weight = _cw
        else:
            _loss_weight = torch.tensor(list(_class_weights_cfg),
                                        dtype=torch.float32, device=device)
            print(f"  Class weights: explicit (max={_loss_weight.max():.2f}, "
                  f"min={_loss_weight.min():.2f})")

    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=float(cfg.training.get("label_smoothing", 0.1)),
        weight=_loss_weight,
    )
    optimizer = build_optimizer(model, cfg)

    # Optional EMA of model weights (evaluated at val time instead of model).
    # Standard improvement of +0.3 to +1.0pp for stable training schedules.
    ema_model: "torch.optim.swa_utils.AveragedModel | None" = None
    if bool(cfg.training.get("use_ema", False)):
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        _ema_decay = float(cfg.training.get("ema_decay", 0.999))
        ema_model = AveragedModel(
            model, multi_avg_fn=get_ema_multi_avg_fn(_ema_decay), use_buffers=True,
        )
        print(f"  EMA enabled: decay={_ema_decay}")

    # Print LR groups
    for i, pg in enumerate(optimizer.param_groups):
        n = sum(p.numel() for p in pg["params"])
        print(f"  Param group {i}: {n/1e6:.1f}M params, lr={pg['lr']:.2e}")

    total_epochs = int(cfg.training.epochs)
    warmup_epochs = int(cfg.training.get("warmup_epochs", 10))
    min_lr = float(cfg.training.get("min_lr", 1e-6))
    warmup_sched = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_sched = CosineAnnealingLR(
        optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=min_lr
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
    )

    use_augment = bool(cfg.training.get("use_augment", True))
    aug_kwargs: Dict[str, Any] = {
        "color_prob":       float(cfg.training.get("color_prob",     0.8)),
        "color_strength":   float(cfg.training.get("color_strength", 0.3)),
        "blur_prob":        float(cfg.training.get("blur_prob",      0.1)),
        "input_normalized": True,
        "output_normalized": True,
    }

    forbidden = list(cfg.dataset.get("flip_forbidden_labels", []))
    flip_forbidden = torch.tensor([int(x) for x in forbidden], dtype=torch.long, device=device)
    flip_prob = float(cfg.training.get("flip_prob", 0.5))

    num_classes = int(cfg.model.num_classes)
    reverse_lookup = None
    if cfg.dataset.get("temporal_reverse_map"):
        rev_map = {int(k): int(v) for k, v in cfg.dataset.temporal_reverse_map.items()}
        reverse_lookup = _build_label_lookup(rev_map, num_classes, device)
        print(f"  Temporal reverse equivariance: {len(rev_map)} class pairs")
    reverse_prob = float(cfg.training.get("equivariant_reverse_prob", 0.5))
    use_temporal_map = bool(cfg.training.get("use_temporal_map", True))

    clip_grad_norm = float(cfg.training.get("clip_grad_norm", 1.0))
    grad_accum_steps = int(cfg.training.get("grad_accum_steps", 1))
    if grad_accum_steps > 1:
        print(f"  Gradient accumulation: {grad_accum_steps} steps "
              f"(effective batch size = {int(cfg.training.batch_size) * grad_accum_steps})")

    # Gradual unfreezing schedule: list of [epoch, n_frozen_blocks] pairs
    unfreeze_sched: List[Tuple[int, int]] = []
    if cfg.training.get("unfreeze_schedule"):
        unfreeze_sched = sorted(
            [(int(p[0]), int(p[1])) for p in cfg.training.unfreeze_schedule],
            key=lambda x: x[0],
        )

    best_val_accuracy = 0.0
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    recent_save_accs: list = []
    epochs_since_save = 0
    start_epoch = 0

    if checkpoint_path.exists():
        print(f"  Checkpoint trouvé : {checkpoint_path} — reprise de l'entraînement...")
        _ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(_ckpt["model_state_dict"])
        if "optimizer_state_dict" in _ckpt:
            optimizer.load_state_dict(_ckpt["optimizer_state_dict"])
        if _ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(_ckpt["scheduler_state_dict"])
        start_epoch       = _ckpt.get("epoch", 0)
        best_val_accuracy = _ckpt.get("best_val_accuracy", _ckpt.get("val_accuracy", 0.0))
        recent_save_accs  = _ckpt.get("recent_save_accs", [])
        epochs_since_save = _ckpt.get("epochs_since_save", 0)
        print(f"  Reprise depuis epoch {start_epoch}/{total_epochs} (best val acc={best_val_accuracy:.4f})")

    for epoch in range(start_epoch, total_epochs):
        # Apply gradual unfreezing at scheduled epochs
        for sched_epoch, n_frozen in unfreeze_sched:
            if epoch == sched_epoch and hasattr(model, "unfreeze_to"):
                model.unfreeze_to(n_frozen)
                optimizer = build_optimizer(model, cfg)
                remaining = max(1, total_epochs - epoch)
                scheduler = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=min_lr)
                print(f"  Epoch {epoch+1}: encoder thawed to {n_frozen} frozen blocks, optimizer rebuilt")
                break

        current_lr = optimizer.param_groups[-1]["lr"]  # head LR
        epoch_t0 = time.perf_counter()
        train_loss, train_acc, grad_norm = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            use_augment=use_augment, aug_kwargs=aug_kwargs,
            flip_forbidden=flip_forbidden, flip_prob=flip_prob,
            reverse_lookup=reverse_lookup, reverse_prob=reverse_prob,
            use_temporal_map=use_temporal_map,
            clip_grad_norm=clip_grad_norm,
            grad_accum_steps=grad_accum_steps,
            ema_model=ema_model,
        )
        train_time = time.perf_counter() - epoch_t0
        val_t0 = time.perf_counter()
        # If EMA is enabled, evaluate on the averaged model (slightly slower since
        # the EMA wrapper adds a thin Python forward overhead, but usually more
        # accurate). Fall back to the live model if EMA is off.
        _eval_model = ema_model if ema_model is not None else model
        val_loss, val_acc = evaluate_epoch(_eval_model, val_loader, loss_fn, device)
        val_time = time.perf_counter() - val_t0
        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{total_epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"head_lr {current_lr:.2e} | grad {grad_norm:.2f} | "
            f"t {train_time:.0f}+{val_time:.0f}s"
        )
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "head_lr": current_lr,
            "grad_norm": grad_norm,
            "epoch_time_train_s": train_time,
            "epoch_time_val_s": val_time,
            "epoch_time_total_s": train_time + val_time,
        }, step=epoch + 1)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            recent_save_accs.append(val_acc)
            epochs_since_save = 0
            # When EMA is on, the eval'd val_acc came from the EMA model — save
            # those as `model_state_dict` so downstream consumers (evaluate.py,
            # ensemble_strategies.py) use the same weights that produced the score.
            # Live weights remain accessible under `live_state_dict` if needed.
            _ckpt_state = (ema_model.module.state_dict()
                           if ema_model is not None else model.state_dict())
            _save_blob = {
                "model_state_dict":     _ckpt_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch":                epoch + 1,
                "best_val_accuracy":    best_val_accuracy,
                "recent_save_accs":     recent_save_accs,
                "epochs_since_save":    0,
                "model_name":           cfg.model.name,
                "num_classes":          num_classes,
                "pretrained":           bool(cfg.model.get("pretrained", True)),
                "use_imagenet_norm":    True,
                "num_frames":           int(cfg.dataset.num_frames),
                "val_accuracy":         val_acc,
                "config":               OmegaConf.to_container(cfg, resolve=True),
            }
            if ema_model is not None:
                _save_blob["live_state_dict"] = model.state_dict()
            torch.save(_save_blob, checkpoint_path)
            print(f"  Saved best model to {checkpoint_path} (epoch={epoch + 1}, val acc={val_acc:.4f})")
            wandb.run.summary["best_val_acc"] = val_acc
            wandb.run.summary["best_epoch"] = epoch + 1
        else:
            epochs_since_save += 1

        stop = False
        _es_patience = int(cfg.training.get("early_stop_patience", 6))
        _es_min_gain = float(cfg.training.get("early_stop_min_gain", 0.01))
        _es_disabled = _es_patience <= 0
        if not _es_disabled and epochs_since_save >= _es_patience:
            print(f"  Early stop: no model saved in the last {_es_patience} epochs.")
            stop = True
        if (not _es_disabled
                and len(recent_save_accs) >= 3
                and recent_save_accs[-1] - recent_save_accs[-3] < _es_min_gain):
            print(f"  Early stop: val accuracy gain over last 3 saves < {_es_min_gain:.0%}.")
            stop = True
        if stop:
            break

    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}")
    if checkpoint_path.exists():
        try:
            _best = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(_best["model_state_dict"])
            log_wandb_diagnostics(model, val_loader, device, num_classes)
        except Exception as exc:
            print(f"[wandb] diagnostics skipped ({exc})")
    wandb.finish()


if __name__ == "__main__":
    main()
