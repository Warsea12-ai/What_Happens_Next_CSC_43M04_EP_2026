# Video classification challenge

This repository contains the codebase for the CSC_43M04_EP - Modal d'informatique - Deep Learning in Computer Vision Challenge "What_Happens_Next?" 

The codebase allows to train a **video classifier** on folders of extracted frames. Each video is a directory of JPG frames; the model sees a fixed number of frames per clip and predicts one class among 33 action categories.

## Getting the data

1. Download the prepared dataset from Google Drive: **[add your link here]**.
2. Unzip it so that you have a `processed_data` folder at the **root of this project** (next to `src/`), with at least:
   - `processed_data/train/` — class subfolders, each containing `video_<id>/frame_*.jpg`
   - `processed_data/val/` — same layout for validation
   - `processed_data/test/` — video subfolders, used for submission 
   The exact subfolder names under `processed_data` must match what you set in the Hydra config (see below). The default configuration expects paths like `processed_data/train`, `processed_data/val`, and `processed_data/test` (see `src/configs/data/default.yaml`).

If your zip uses a different folder name than `val2`, either rename it or override the paths on the command line (examples below).

## Environment

Use Python 3.10+ and `uv`:

```bash
uv sync
```

Run training and evaluation from the **`src/`** directory so Hydra finds `configs/`:

```bash
cd src
```

Or from the repo root:

```bash
python src/train.py experiment=cnn_lstm
```

## How the code is organized

| Piece | Role |
|--------|------|
| `dataset/video_dataset.py` | Loads `T` frames per video folder, applies image transforms, returns tensors `(batch, time, channels, height, width)` and integer labels. |
| `models/` | Neural networks: each model maps a batch of shape `(B, T, C, H, W)` to logits `(B, num_classes)`. |
| `utils.py` | Image transforms, train/val split helper, seeds. |
| `train.py` | Training loop; saves the best checkpoint by validation accuracy (full Hydra config + weights). |
| `evaluate.py` | Rebuilds the model from the checkpoint config and reports **top-1** and **top-5** on the **full** validation directory (`dataset.val_dir`). |
| `create_submission.py` | Loads a checkpoint, runs inference on the test split, writes `video_name,predicted_class`. |
| `configs/` | [Hydra](https://hydra.cc/) YAML: **`experiment/`** (choose a preset), **`model/`**, **`data/`**, **`train/`**. |

The main composition file is `configs/config.yaml`. Global values such as `num_classes: 33` apply across model configs.

### Experiments (recommended way to train)

An **experiment** selects which model and other settings (learning rate, optimizer, data augmentation) to use without editing Python. Defaults live in `configs/experiment/`:

- `baseline_from_scratch` — ResNet18 backbone, average pooling over time (Track 1 - Closed World)
- `baseline_pretrained` — pretrained ResNet18 backbone, average pooling over time (Track 2 - Open World)

Run:

```bash
python src/train.py experiment=baseline_from_scratch
```

This sets the active `model` group (via Hydra `override /model: ...`). You can still override any field:

```bash
python train.py experiment=baseline_from_scratch model.pretrained=false dataset.train_dir=/path/to/train
python train.py training.epochs=10 training.batch_size=16 training.lr=0.0001
```

The best checkpoint is written to **`training.checkpoint_path`** (see printed path). It always stores the **full merged Hydra config**, so evaluation and submission reload the same architecture automatically.

Hydra may create an `outputs/` folder with logs for each run.

## Evaluation

Evaluation uses the **entire** validation set under **`dataset.val_dir`** (no random split). The checkpoint must have been produced by the current `train.py` (it needs the saved `config` inside the `.pt` file).

```bash
python evaluate.py training.checkpoint_path=best_model.pt
```

```bash
python evaluate.py training.checkpoint_path=/path/to/ckpt.pt dataset.val_dir=/path/to/val
```

## Creating a submission file

Reads test frames from **`dataset.test_dir`**, clip order from **`dataset.test_manifest`**, writes **`dataset.submission_output`**.

```bash
python create_submission.py training.checkpoint_path=best_model.pt
```

```bash
python create_submission.py \
  training.checkpoint_path=best_model.pt \
  dataset.submission_output=../my_submission.csv
```

CSV format:

```text
video_name,predicted_class
video_12345,7
```

## Adding a new model

1. **Implement** `torch.nn.Module` in `src/models/your_model.py`.  
   - **Input:** `(B, T, C, H, W)`  
   - **Output:** logits `(B, num_classes)`.

2. **Register once** in `train.py` inside `build_model()`: add a branch for `cfg.model.name == "your_model_name"` and return your module.

3. **Add** `src/configs/model/your_model.yaml`:

   ```yaml
   # @package _global_
   model:
     name: your_model_name
     pretrained: true
     num_classes: ${num_classes}
     # your hyperparameters
   ```

4. **Add an experiment** `src/configs/experiment/your_experiment.yaml` that points Hydra at that model:

   ```yaml
   # @package _global_
   defaults:
     - override /model: your_model
   ```

   Optionally add more `defaults` lines or same-level keys to override data or training for that experiment only.

5. **Train**:

   ```bash
   python train.py experiment=your_experiment
   ```

`evaluate.py` and `create_submission.py` do **not** need edits: they call `build_model` with the config saved in your checkpoint.

## Tips

- Set `training.device=cuda` when a GPU is available; use `cpu` otherwise.
- Keep `num_classes` in `configs/config.yaml` aligned with the dataset (default **33**).
- `dataset.seed` controls the internal train/val split during training only.

If something fails, check that `processed_data` paths in `configs/data/default.yaml` (or your CLI overrides) match the folder you downloaded.
