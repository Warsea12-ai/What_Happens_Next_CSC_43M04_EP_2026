"""
Convertit ton dataset (dossiers de classes contenant des dossiers vidéos de 4 frames)
au format JSONL attendu par ms-swift pour fine-tuner Qwen2.5-VL.

Structure attendue en entrée :
    train_dir/
    ├── ClassName1/
    │   ├── video_id_001/
    │   │   ├── frame_0001.jpg
    │   │   ├── frame_0002.jpg
    │   │   ├── frame_0003.jpg
    │   │   └── frame_0004.jpg
    │   └── video_id_002/...
    ├── ClassName2/...

Sortie :
    /Data/qwen_data/
    ├── train.jsonl
    ├── val.jsonl
    └── class_mapping.json   # pour le decoding à l'inférence
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List


# ============================================================
# 1. Configuration
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True,
                   help="Racine du dataset (contient les sous-dossiers de classes)")
    p.add_argument("--out_dir", required=True,
                   help="Où écrire train.jsonl / val.jsonl")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frame_ext", default=".jpg",
                   help="Extension des frames (jpg, png, ...)")
    return p.parse_args()


# ============================================================
# 2. Construction du prompt
# ============================================================
def build_user_prompt(class_names: List[str]) -> str:
    """Prompt utilisateur : on liste les classes et on demande au modèle de choisir."""
    options = "\n".join(f"{i}. {c.replace('_', ' ')}" for i, c in enumerate(class_names))
    return (
        "<video>You are watching the beginning of a video showing a human-object "
        "interaction. Based on what you can see in the first frames, predict which "
        "of the following actions will be performed next.\n\n"
        "Choose ONE action from the list below:\n"
        f"{options}\n\n"
        "Answer with only the number and the action name, exactly as written. "
        "Example: \"5. Picking something up\""
    )


def build_assistant_response(class_idx: int, class_names: List[str]) -> str:
    """Réponse cible attendue du modèle (utilisée pour la loss)."""
    name = class_names[class_idx].replace("_", " ")
    return f"{class_idx}. {name}"


# ============================================================
# 3. Collecte des vidéos
# ============================================================
def collect_videos(train_dir: Path, frame_ext: str):
    """Retourne (class_names, [(class_idx, video_folder_path), ...])."""
    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]
    print(f"Found {len(class_names)} classes")

    samples = []
    for class_idx, class_dir in enumerate(class_dirs):
        videos = sorted([d for d in class_dir.iterdir() if d.is_dir()])
        print(f"  [{class_idx:2d}] {class_dir.name}: {len(videos)} videos")
        for video_dir in videos:
            frames = sorted(video_dir.glob(f"*{frame_ext}"))
            if len(frames) < 1:
                print(f"  ⚠️  Skipping {video_dir} (no {frame_ext} frames)")
                continue
            samples.append((class_idx, video_dir, frames))

    return class_names, samples


# ============================================================
# 4. Génération du JSONL
# ============================================================
def make_record(class_idx: int, frames: List[Path], class_names: List[str]) -> dict:
    """
    Un exemple au format ms-swift / Qwen2.5-VL.

    On passe la liste des frames comme une "vidéo" via la convention des chemins
    séparés par des virgules. Qwen2.5-VL traitera la liste comme une séquence
    temporelle de frames.
    """
    frame_paths = ",".join(str(f.resolve()) for f in frames)

    return {
        "videos": [frame_paths],
        "messages": [
            {"role": "user", "content": build_user_prompt(class_names)},
            {"role": "assistant",
             "content": build_assistant_response(class_idx, class_names)},
        ],
    }


def write_jsonl(samples, class_names, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for class_idx, _video_dir, frames in samples:
            rec = make_record(class_idx, frames, class_names)
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(samples)} samples to {out_path}")


# ============================================================
# 5. Main
# ============================================================
def main() -> None:
    args = parse_args()
    train_dir = Path(args.train_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    random.seed(args.seed)

    class_names, samples = collect_videos(train_dir, args.frame_ext)
    print(f"\nTotal videos: {len(samples)}")

    # --- Split train / val stratifié simple ---
    random.shuffle(samples)
    n_val = int(len(samples) * args.val_ratio)
    val = samples[:n_val]
    train = samples[n_val:]
    print(f"Train: {len(train)} | Val: {len(val)}")

    # --- Écriture des JSONL ---
    write_jsonl(train, class_names, out_dir / "train.jsonl")
    write_jsonl(val,   class_names, out_dir / "val.jsonl")

    # --- Mapping classes pour le post-processing ---
    mapping = {i: name for i, name in enumerate(class_names)}
    (out_dir / "class_mapping.json").write_text(json.dumps(mapping, indent=2))
    print(f"Wrote class mapping to {out_dir / 'class_mapping.json'}")


if __name__ == "__main__":
    main()