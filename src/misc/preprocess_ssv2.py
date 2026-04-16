#!/usr/bin/env python3
"""
Preprocess Something-Something V2 (or similar) video data for image-sequence classification.

Reads JSON lists of { "id", "template", ... } entries, filters by a user-defined list of
class names, and extracts uniformly sampled RGB frames from the first X%% of each video.

Two split modes:

* **random** — one ``--annotations`` file, then an 80/20 (configurable) stratified
  train/val split (useful for quick experiments).
* **official** — separate ``--train-json`` and ``--val-json`` (e.g. SSv2 ``train.json``
  and ``validation.json``); no random re-splitting.

* **Test** — Optional ``--test-json`` (ids only). If you also pass private ``--test-answers``
  (``video_id;plain_label``), test clips are **filtered** to your selected classes and
  written under ``test/<class>/video_<id>/``; a filtered ``test-answers.csv`` is written to
  the output folder for instructor evaluation only. Without ``--test-answers``, unlabeled
  extraction uses ``test/video_<id>/``.

Dependencies: Python 3.8+, OpenCV (cv2), standard library only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def _import_cv2():
    """Load OpenCV lazily so --help works without cv2 installed."""
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        print(
            "Error: OpenCV is required. Install with: pip install opencv-python-headless",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    return cv2


# ---------------------------------------------------------------------------
# Data loading and filtering
# ---------------------------------------------------------------------------


def load_annotations(path: Path, class_field: str = "template") -> Dict[str, str]:
    """
    Load annotations from a JSON file into a mapping video_id -> class label string.

    Expected format (Something-Something V2 official JSON): a list of objects, each with
    at least "id" and a class field. By default we use "template" so labels match
    labels.json (e.g. "Moving [something] up"). Use class_field="label" only if your
    JSON already stores the exact strings you list in --selected-classes.

    Args:
        path: Path to the JSON file (e.g. train.json or validation.json).
        class_field: Which key holds the class name ("template" or "label").

    Returns:
        Dictionary mapping video id (string) to class name (string).
    """
    with path.open("r", encoding="utf-8") as f:
        data: Any = json.load(f)

    if isinstance(data, dict):
        # Allow a direct id -> label mapping: { "12345": "Moving [something] up", ... }
        out: Dict[str, str] = {str(k): str(v) for k, v in data.items()}
        return out

    if not isinstance(data, list):
        raise ValueError(f"Unsupported JSON root type: {type(data).__name__}")

    annotations: Dict[str, str] = {}
    skipped = 0
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            skipped += 1
            continue
        vid = item.get("id")
        if vid is None:
            skipped += 1
            continue
        label = item.get(class_field)
        if label is None:
            skipped += 1
            continue
        annotations[str(vid)] = str(label)

    if skipped:
        print(
            f"Warning: skipped {skipped} entries that were missing id or '{class_field}'.",
            file=sys.stderr,
        )

    return annotations


def load_test_ids(path: Path) -> List[str]:
    """
    Load video ids from Something-Something ``test.json`` (ids only, no labels).

    Expected format: ``[ {"id": "1420"}, ... ]``
    """
    with path.open("r", encoding="utf-8") as f:
        data: Any = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"test.json must be a JSON list, got {type(data).__name__}")
    ids: List[str] = []
    for item in data:
        if isinstance(item, dict) and item.get("id") is not None:
            ids.append(str(item["id"]))
    return ids


def load_test_answers_csv(path: Path) -> List[Tuple[str, str]]:
    """
    Load ``test-answers.csv``: one row per test video, ``video_id;plain_label``.

    The label is the dataset's plain-English class line (no ``[placeholders]``), matching
    ``labels.json`` key style.
    """
    rows: List[Tuple[str, str]] = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("id;"):
            continue
        if ";" not in line:
            continue
        vid, label = line.split(";", 1)
        rows.append((vid.strip(), label.strip()))
    return rows


def build_plain_norm_to_template_map(selected_classes: Sequence[str]) -> Dict[str, str]:
    """
    Map normalized *plain* label (brackets stripped from templates) → canonical template
    string from the selected-classes file (first line wins for each key).
    Used to align ``test-answers.csv`` plain labels with train/val ``template`` strings.
    """
    plain_norm_to_template: Dict[str, str] = {}
    for c in selected_classes:
        c = c.strip()
        if not c:
            continue
        plain = strip_bracket_placeholders(c)
        key = normalize_class_name_for_matching(plain)
        plain_norm_to_template.setdefault(key, c)
    return plain_norm_to_template


def filter_test_rows_by_selected_classes(
    rows: List[Tuple[str, str]],
    plain_norm_to_template: Dict[str, str],
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Split test-answers rows into (kept, dropped) by whether the plain label matches a
    selected class.

    Returns:
        kept: list of (video_id, template_string) using the template form from
        ``classes.txt`` for consistency with train/val.
        dropped: list of (video_id, plain_label) for rows not in the selection.
    """
    kept: List[Tuple[str, str]] = []
    dropped: List[Tuple[str, str]] = []
    for vid, plain_label in rows:
        key = normalize_class_name_for_matching(plain_label)
        tmpl = plain_norm_to_template.get(key)
        if tmpl is not None:
            kept.append((vid, tmpl))
        else:
            dropped.append((vid, plain_label))
    return kept, dropped


def strip_bracket_placeholders(name: str) -> str:
    """
    Convert template-style text to the plain phrasing used in ``test-answers.csv``
    and ``labels.json`` keys, e.g. ``Moving [something] up`` → ``Moving something up``.
    """
    return re.sub(r"\[([^\]]*)\]", r"\1", name.strip())


def normalize_class_name_for_matching(name: str) -> str:
    """
    Normalize a class template string so small formatting differences still match.

    Handles:
    - Case differences, including ``[Something]`` vs ``[something]`` (full string is
      case-folded).
    - Unicode compatibility (NFKC), e.g. odd spaces or punctuation variants.
    - Comma variants: fullwidth comma (U+FF0C) → ASCII ``,``; spaces around commas
      normalized to a single comma followed by one space (``, ``).
    - Repeated / odd whitespace collapsed to a single space.

    This is only used for *matching*; the original annotation string from the JSON
    is kept when building examples (so filenames and ``class_to_idx`` stay consistent
    with the dataset).
    """
    s = unicodedata.normalize("NFKC", name.strip())
    # Fullwidth / compatibility commas → ASCII (common copy-paste issue).
    for ch in ("\uff0c", "\ufe50", "\ufe10"):
        s = s.replace(ch, ",")
    s = s.casefold()
    s = re.sub(r"\s+", " ", s)
    # "foo , bar" / "foo,bar" → "foo, bar"
    s = re.sub(r"\s*,\s*", ", ", s)
    return s.strip()


def filter_classes(
    annotations: Dict[str, str],
    selected_classes: Sequence[str],
) -> List[Tuple[str, str]]:
    """
    Keep only videos whose label matches one of the selected class names.

    Matching uses :func:`normalize_class_name_for_matching`, so differences such as
    ``[Something]`` vs ``[something]``, extra spaces, or comma spacing do not prevent a
    match. Each kept sample still uses the **original** class string from the
    annotations (not the line from ``classes.txt``).

    Args:
        annotations: video_id -> class name.
        selected_classes: Iterable of allowed class names (e.g. from classes.txt).

    Returns:
        List of (video_id, class_name) pairs, in arbitrary order.
    """
    # Map normalized form → first occurrence in the file (for duplicate-key warnings).
    allowed_norm_to_line: Dict[str, str] = {}
    for c in selected_classes:
        c = c.strip()
        if not c:
            continue
        key = normalize_class_name_for_matching(c)
        if key in allowed_norm_to_line and allowed_norm_to_line[key] != c:
            print(
                f"Warning: selected classes normalize to the same key; using first line "
                f"for reference: {allowed_norm_to_line[key]!r} vs {c!r}",
                file=sys.stderr,
            )
        allowed_norm_to_line.setdefault(key, c)

    if not allowed_norm_to_line:
        raise ValueError("No non-empty class names in selected_classes.")

    allowed_keys = set(allowed_norm_to_line.keys())

    pairs: List[Tuple[str, str]] = []
    dropped = 0
    for vid, cls in annotations.items():
        if normalize_class_name_for_matching(cls) in allowed_keys:
            pairs.append((vid, cls))
        else:
            dropped += 1

    # Selected lines that never matched any annotation (often a wording mismatch).
    matched_norms = {normalize_class_name_for_matching(cls) for _, cls in pairs}
    unused = [
        line for k, line in allowed_norm_to_line.items() if k not in matched_norms
    ]
    if unused:
        print(
            f"Warning: {len(unused)} selected class line(s) had no matching videos "
            f"(check spelling vs JSON templates). Example: {unused[0]!r}",
            file=sys.stderr,
        )

    print(
        f"Filtered to {len(pairs)} videos in {len(allowed_keys)} classes "
        f"(dropped {dropped} videos not in selected set).",
        file=sys.stderr,
    )
    return pairs


# ---------------------------------------------------------------------------
# Train / validation split (stratified when each class has enough samples)
# ---------------------------------------------------------------------------


def split_dataset(
    pairs: List[Tuple[str, str]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Split (video_id, class_name) pairs into train and validation sets.

    Stratified: for each class, roughly val_ratio of its videos go to validation.
    Classes with only one video are assigned entirely to train so validation never
    ends up empty for that class.

    Args:
        pairs: All labeled samples after filtering.
        val_ratio: Fraction of samples per class for validation (e.g. 0.2).
        seed: RNG seed for reproducibility.

    Returns:
        (train_pairs, val_pairs), each a list of (video_id, class_name).
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1 (exclusive).")

    rng = random.Random(seed)
    by_class: Dict[str, List[str]] = defaultdict(list)
    for vid, cls in pairs:
        by_class[cls].append(vid)

    train_out: List[Tuple[str, str]] = []
    val_out: List[Tuple[str, str]] = []

    for cls, vids in sorted(by_class.items()):
        vids = vids.copy()
        rng.shuffle(vids)
        n = len(vids)
        # Number of validation clips for this class (at least 0, at most n-1 when n>1).
        n_val = int(round(n * val_ratio))
        if n <= 1:
            n_val = 0
        else:
            n_val = min(max(0, n_val), n - 1)

        val_ids = set(vids[:n_val])
        for vid in vids:
            if vid in val_ids:
                val_out.append((vid, cls))
            else:
                train_out.append((vid, cls))

    rng.shuffle(train_out)
    rng.shuffle(val_out)
    return train_out, val_out


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def _safe_subdir_name(class_name: str, class_idx: int) -> str:
    """
    Build a filesystem-safe folder name: zero-padded index + short slug from class name.
    Example: 012_Moving_something_up
    """
    slug = re.sub(r"[^\w\-.]+", "_", class_name)
    slug = slug.strip("_")[:60]
    return f"{class_idx:03d}_{slug}" if slug else f"{class_idx:03d}_class"


def extract_frames(
    video_path: Path,
    out_dir: Path,
    num_frames: int,
    first_percent: float,
    resize_wh: Tuple[int, int] = (224, 224),
    jpeg_quality: int = 90,
) -> bool:
    """
    Decode a video, sample frames uniformly from the first `first_percent`%% of its
    length, resize to resize_wh, and save as frame_000.jpg, frame_001.jpg, ...

    Args:
        video_path: Path to the video file (e.g. .webm).
        out_dir: Directory to create and write frames into (created if missing).
        num_frames: How many frames to save (uniform indices in time).
        first_percent: Use only the first this percentage of the video (0-100).
        resize_wh: Output (width, height).
        jpeg_quality: JPEG quality 0-100 for cv2.imwrite.

    Returns:
        True if frames were written, False on failure (caller may log a warning).
    """
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1.")
    if not 0.0 < first_percent <= 100.0:
        raise ValueError("first_percent must be in (0, 100].")

    cv2 = _import_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0

        if frame_count <= 0:
            # Some containers report 0 frames; try to count by reading.
            frame_count = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                frame_count += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if frame_count <= 0:
            return False

        # Last frame index (inclusive) within the "first X%%" of the video.
        last_idx = max(0, int(math.floor((frame_count - 1) * (first_percent / 100.0))))
        usable = last_idx + 1
        if usable < 1:
            return False

        # Uniform indices in [0, last_idx].
        if num_frames == 1:
            indices = [last_idx // 2]
        else:
            indices = [
                int(round(i * (usable - 1) / (num_frames - 1)))
                for i in range(num_frames)
            ]

        out_dir.mkdir(parents=True, exist_ok=True)
        w, h = resize_wh

        for i, fi in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                # Fallback: seek again; some codecs are picky.
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
                ok, frame = cap.read()
            if not ok or frame is None:
                return False

            # BGR -> RGB is not needed for saving with cv2.imwrite (expects BGR).
            resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            out_file = out_dir / f"frame_{i:03d}.jpg"
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
            if not cv2.imwrite(str(out_file), resized, encode_params):
                return False

        return True
    finally:
        cap.release()


def find_video_file(video_dir: Path, video_id: str) -> Path | None:
    """
    Locate a video file for a given id. Tries common extensions.
    """
    for ext in (".webm", ".mp4", ".mkv", ".avi", ".mov"):
        p = video_dir / f"{video_id}{ext}"
        if p.is_file():
            return p
    return None


def load_selected_classes(path: Path) -> List[str]:
    """Load class names from a text file (one per line) or a JSON list of strings."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON selected-classes file must be a list of strings.")
        return [str(x).strip() for x in data]
    return [line.strip() for line in text.splitlines() if line.strip()]


def build_class_mapping(sorted_class_names: Sequence[str]) -> Dict[str, int]:
    """Map each class name to a contiguous integer id (order is alphabetical)."""
    return {name: i for i, name in enumerate(sorted_class_names)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess SSv2-style videos for classification (frames + splits).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--video-dir",
        type=Path,
        required=True,
        help="Directory containing video files named <id>.webm (or .mp4, ...).",
    )
    p.add_argument(
        "--split-mode",
        choices=("random", "official"),
        default="random",
        help=(
            "random: use --annotations then stratified train/val split. "
            "official: use --train-json and --val-json (dataset train/val; no re-split)."
        ),
    )
    p.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Single JSON with labels (required for --split-mode random).",
    )
    p.add_argument(
        "--train-json",
        type=Path,
        default=None,
        help="Official train split JSON (required for --split-mode official).",
    )
    p.add_argument(
        "--val-json",
        type=Path,
        default=None,
        help="Official validation split JSON (required for --split-mode official).",
    )
    p.add_argument(
        "--test-json",
        type=Path,
        default=None,
        help=(
            "Optional official test id list (ids only). With --test-answers, only ids "
            "present in both files are used. Without --test-answers, frames go to "
            "test/video_<id>/ (no labels, no class folders)."
        ),
    )
    p.add_argument(
        "--test-answers",
        type=Path,
        default=None,
        help=(
            "Private/instructor file: test-answers.csv (video_id;plain_label). "
            "Filters the test set to selected classes (matching plain labels to "
            "classes.txt templates). Writes a filtered copy to OUTPUT/test-answers.csv "
            "and extracts frames under test/<class>/video_<id>/."
        ),
    )
    p.add_argument(
        "--selected-classes",
        type=Path,
        required=True,
        help="Text file: one class name per line, OR a JSON list of strings.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed_data"),
        help="Root output directory (train/ and val/ will be created here).",
    )
    p.add_argument(
        "--class-field",
        choices=("template", "label"),
        default="template",
        help='JSON field for the class string. Use "template" for official SSv2 / labels.json.',
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation fraction (stratified). Only for --split-mode random.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting. Only for --split-mode random.",
    )
    p.add_argument(
        "--num-frames",
        type=int,
        default=4,
        help="Number of frames to sample per video (uniform in first X%%).",
    )
    p.add_argument(
        "--first-percent",
        type=float,
        default=40.0,
        help="Only use the first this percentage of each video's timeline.",
    )
    p.add_argument(
        "--resize",
        type=int,
        default=224,
        help="Square resize: resize x resize pixels.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="If a video folder already has frame_000.jpg, skip re-extraction.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    video_dir = args.video_dir.expanduser().resolve()
    out_root = args.output_dir.expanduser().resolve()
    resize_wh = (args.resize, args.resize)

    if args.split_mode == "random":
        if args.annotations is None:
            raise SystemExit(
                "Error: --annotations is required for --split-mode random."
            )
    else:
        if args.train_json is None or args.val_json is None:
            raise SystemExit(
                "Error: --train-json and --val-json are required for --split-mode official."
            )

    selected = load_selected_classes(args.selected_classes)
    plain_norm_to_template = build_plain_norm_to_template_map(selected)
    class_field = args.class_field

    if args.split_mode == "random":
        ann = load_annotations(
            args.annotations.expanduser().resolve(), class_field=class_field
        )
        pairs = filter_classes(ann, selected)
        train_pairs, val_pairs = split_dataset(
            pairs, val_ratio=args.val_ratio, seed=args.seed
        )
    else:
        train_ann = load_annotations(
            args.train_json.expanduser().resolve(), class_field=class_field
        )
        val_ann = load_annotations(
            args.val_json.expanduser().resolve(), class_field=class_field
        )
        train_pairs = filter_classes(train_ann, selected)
        val_pairs = filter_classes(val_ann, selected)

    test_pairs_labeled: List[Tuple[str, str]] = []
    test_answer_rows_out: List[Tuple[str, str]] = []
    if args.test_answers is not None:
        answers_path = args.test_answers.expanduser().resolve()
        all_test_rows = load_test_answers_csv(answers_path)
        kept, dropped_ans = filter_test_rows_by_selected_classes(
            all_test_rows, plain_norm_to_template
        )
        if args.test_json is not None:
            official_test_ids = set(
                load_test_ids(args.test_json.expanduser().resolve())
            )
            before = len(kept)
            kept = [(vid, cls) for vid, cls in kept if vid in official_test_ids]
            print(
                f"Test: {before} clips matched selected classes; "
                f"{before - len(kept)} removed (id not in --test-json).",
                file=sys.stderr,
            )
        test_pairs_labeled = kept
        vid_to_plain = {v: pl for v, pl in all_test_rows}
        for vid, tmpl in kept:
            plain = vid_to_plain.get(vid, strip_bracket_placeholders(tmpl))
            test_answer_rows_out.append((vid, plain))
        print(
            f"Test (from test-answers): {len(test_pairs_labeled)} clips in selected classes; "
            f"{len(dropped_ans)} test videos skipped (label not in selection).",
            file=sys.stderr,
        )

    # Class indices: train + val + labeled test (so every extracted clip has an id).
    unique_classes = sorted(
        {c for _, c in train_pairs}
        | {c for _, c in val_pairs}
        | {c for _, c in test_pairs_labeled}
    )
    class_to_idx = build_class_mapping(unique_classes)

    out_root.mkdir(parents=True, exist_ok=True)
    mapping_path = out_root / "class_to_idx.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2, ensure_ascii=False)
    print(f"Wrote class mapping ({len(class_to_idx)} classes) to {mapping_path}")

    if test_answer_rows_out:
        out_answers = out_root / "test-answers.csv"
        with out_answers.open("w", encoding="utf-8", newline="") as f:
            for vid, plain in test_answer_rows_out:
                f.write(f"{vid};{plain}\n")
        print(
            f"Wrote filtered instructor labels ({len(test_answer_rows_out)} rows) to {out_answers}",
            file=sys.stderr,
        )

    n_selected_lines = len([ln for ln in selected if ln.strip()])
    n_classes_with_video = len(unique_classes)
    n_train = len(train_pairs)
    n_val = len(val_pairs)
    if test_pairs_labeled:
        n_test = len(test_pairs_labeled)
    elif args.test_json is not None:
        n_test = len(load_test_ids(args.test_json.expanduser().resolve()))
    else:
        n_test = 0

    norms_with_any_video = (
        {normalize_class_name_for_matching(c) for _, c in train_pairs}
        | {normalize_class_name_for_matching(c) for _, c in val_pairs}
        | {normalize_class_name_for_matching(c) for _, c in test_pairs_labeled}
    )
    selected_classes_with_no_videos: List[str] = []
    for line in selected:
        line = line.strip()
        if not line:
            continue
        if normalize_class_name_for_matching(line) not in norms_with_any_video:
            selected_classes_with_no_videos.append(line)

    summary_extra = ""
    if selected_classes_with_no_videos:
        summary_extra = (
            "\n  Selected classes with NO videos (check wording vs JSON / test-answers):\n"
            + "".join(f"    - {name}\n" for name in selected_classes_with_no_videos)
        )
    else:
        summary_extra = "\n  Selected classes with NO videos: (none)\n"

    print(
        "\n=== Summary (before video frame extraction) ===\n"
        f"  Non-empty lines in selected-classes file: {n_selected_lines}\n"
        f"  Distinct classes with at least one matching video: {n_classes_with_video}\n"
        f"  Train videos: {n_train}\n"
        f"  Val videos:   {n_val}\n"
        f"  Test videos:  {n_test}"
        + (
            " (filtered by --test-answers)"
            if test_pairs_labeled
            else (" (all ids from --test-json)" if args.test_json is not None else " (test split not configured)")
        )
        + summary_extra
        + "==============================================\n",
        file=sys.stderr,
    )

    splits: List[Tuple[str, List[Tuple[str, str]]]] = [
        ("train", train_pairs),
        ("val", val_pairs),
    ]
    stats_ok = 0
    stats_skip = 0
    stats_bad = 0

    for split_name, split_pairs in splits:
        split_dir = out_root / split_name
        for rank, (vid, cls) in enumerate(split_pairs):
            idx = class_to_idx[cls]
            class_dir_name = _safe_subdir_name(cls, idx)
            vid_dir = split_dir / class_dir_name / f"video_{vid}"
            if args.skip_existing and (vid_dir / "frame_000.jpg").is_file():
                stats_skip += 1
                continue

            vpath = find_video_file(video_dir, vid)
            if vpath is None:
                print(
                    f"Warning: no video file for id={vid} in {video_dir}",
                    file=sys.stderr,
                )
                stats_bad += 1
                continue

            if vid_dir.exists():
                # Remove partial / old frames for a clean re-run.
                for old in vid_dir.glob("frame_*.jpg"):
                    old.unlink()

            ok = extract_frames(
                vpath,
                vid_dir,
                num_frames=args.num_frames,
                first_percent=args.first_percent,
                resize_wh=resize_wh,
            )
            if ok:
                stats_ok += 1
            else:
                print(
                    f"Warning: failed to extract frames from {vpath} (corrupt or unsupported).",
                    file=sys.stderr,
                )
                stats_bad += 1

            if (rank + 1) % 100 == 0:
                print(f"  [{split_name}] processed {rank + 1} / {len(split_pairs)} ...")

    # Test split: either filtered by private test-answers.csv (class folders) or raw ids from test.json.
    if test_pairs_labeled:
        split_name = "test"
        split_dir = out_root / split_name
        print(
            f"Extracting {len(test_pairs_labeled)} filtered test clips under {split_dir}/<class>/video_<id>/ ...",
            file=sys.stderr,
        )
        for rank, (vid, cls) in enumerate(test_pairs_labeled):
            idx = class_to_idx[cls]
            class_dir_name = _safe_subdir_name(cls, idx)
            vid_dir = split_dir / class_dir_name / f"video_{vid}"
            if args.skip_existing and (vid_dir / "frame_000.jpg").is_file():
                stats_skip += 1
                continue

            vpath = find_video_file(video_dir, vid)
            if vpath is None:
                print(
                    f"Warning: no video file for id={vid} in {video_dir}",
                    file=sys.stderr,
                )
                stats_bad += 1
                continue

            if vid_dir.exists():
                for old in vid_dir.glob("frame_*.jpg"):
                    old.unlink()

            ok = extract_frames(
                vpath,
                vid_dir,
                num_frames=args.num_frames,
                first_percent=args.first_percent,
                resize_wh=resize_wh,
            )
            if ok:
                stats_ok += 1
            else:
                print(
                    f"Warning: failed to extract frames from {vpath} (corrupt or unsupported).",
                    file=sys.stderr,
                )
                stats_bad += 1

            if (rank + 1) % 100 == 0:
                print(
                    f"  [{split_name}] processed {rank + 1} / {len(test_pairs_labeled)} ..."
                )

    elif args.test_json is not None:
        test_ids = load_test_ids(args.test_json.expanduser().resolve())
        split_name = "test"
        split_dir = out_root / split_name
        print(
            f"Extracting {len(test_ids)} test clips (no labels) under {split_dir}/video_<id>/ ...",
            file=sys.stderr,
        )
        for rank, vid in enumerate(test_ids):
            vid_dir = split_dir / f"video_{vid}"
            if args.skip_existing and (vid_dir / "frame_000.jpg").is_file():
                stats_skip += 1
                continue

            vpath = find_video_file(video_dir, vid)
            if vpath is None:
                print(
                    f"Warning: no video file for id={vid} in {video_dir}",
                    file=sys.stderr,
                )
                stats_bad += 1
                continue

            if vid_dir.exists():
                for old in vid_dir.glob("frame_*.jpg"):
                    old.unlink()

            ok = extract_frames(
                vpath,
                vid_dir,
                num_frames=args.num_frames,
                first_percent=args.first_percent,
                resize_wh=resize_wh,
            )
            if ok:
                stats_ok += 1
            else:
                print(
                    f"Warning: failed to extract frames from {vpath} (corrupt or unsupported).",
                    file=sys.stderr,
                )
                stats_bad += 1

            if (rank + 1) % 500 == 0:
                print(f"  [{split_name}] processed {rank + 1} / {len(test_ids)} ...")

    print(
        f"Done. Extracted OK: {stats_ok}, skipped (existing): {stats_skip}, "
        f"missing/failed: {stats_bad}. Output: {out_root}"
    )


if __name__ == "__main__":
    main()
