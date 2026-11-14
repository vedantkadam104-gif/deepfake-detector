"""Evaluate a trained checkpoint with Single-Crop and 4x Test-Time Augmentation (TTA)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import DeepfakeFolderDataset, discover_splits
from src.model import load_checkpoint
from src.transforms import IMG_SIZE, get_eval_transforms, get_tta_transforms


@torch.no_grad()
def evaluate_tta(
    model: torch.nn.Module,
    samples: List[Tuple[Path, int]],
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    tta_transforms = get_tta_transforms(IMG_SIZE)
    all_probs = np.zeros(len(samples), dtype=np.float32)

    for tfm in tta_transforms:
        ds = DeepfakeFolderDataset(samples, transform=tfm)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
        view_probs = []
        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(imgs)
                probs = torch.softmax(logits.float(), dim=1)[:, 1]
            view_probs.extend(probs.cpu().numpy().tolist())
        all_probs += np.array(view_probs, dtype=np.float32)

    all_probs /= len(tta_transforms)
    labels = np.array([lbl for _, lbl in samples], dtype=int)
    return labels, all_probs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DeepfakeEfficientNet with and without TTA.")
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset root folder.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    model, ckpt = load_checkpoint(ckpt_path, device)
    thr_path = ckpt_path.parent / "threshold.json"
    threshold = 0.5
    if thr_path.exists():
        threshold = float(json.loads(thr_path.read_text(encoding="utf-8")).get("fake_threshold", 0.5))
    print(f"Loaded checkpoint '{args.checkpoint}' (arch={ckpt.get('arch')}, threshold={threshold:.3f}) on {device}")

    splits = discover_splits(Path(args.data_dir), max_samples_per_class=args.max_samples_per_class)
    samples = splits[args.split] or splits["valid"]
    print(f"Evaluating on {len(samples)} images from split '{args.split}'...")

    # Single-crop evaluation
    eval_ds = DeepfakeFolderDataset(samples, transform=get_eval_transforms(IMG_SIZE))
    eval_loader = DataLoader(eval_ds, batch_size=64, shuffle=False, num_workers=2)
    single_probs, single_labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(eval_loader, desc="Single-Crop Eval"):
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                probs = torch.softmax(model(imgs).float(), dim=1)[:, 1]
            single_probs.extend(probs.cpu().numpy().tolist())
            single_labels.extend(lbls.numpy().tolist())

    y_true = np.array(single_labels)
    y_prob_single = np.array(single_probs)
    y_pred_single = (y_prob_single >= threshold).astype(int)

    print(f"\n=== Single-Crop Results (threshold={threshold:.3f}) ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred_single) * 100:.2f}% | ROC-AUC: {roc_auc_score(y_true, y_prob_single):.4f}")
    print(classification_report(y_true, y_pred_single, target_names=["Real (0)", "Fake (1)"]))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred_single))

    # 4x TTA evaluation
    y_true_tta, y_prob_tta = evaluate_tta(model, samples, device)
    y_pred_tta = (y_prob_tta >= threshold).astype(int)
    print(f"\n=== 4x Test-Time Augmentation (TTA) Results (threshold={threshold:.3f}) ===")
    print(f"Accuracy: {accuracy_score(y_true_tta, y_pred_tta) * 100:.2f}% | ROC-AUC: {roc_auc_score(y_true_tta, y_prob_tta):.4f}")
    print(classification_report(y_true_tta, y_pred_tta, target_names=["Real (0)", "Fake (1)"]))
    print("Confusion Matrix:\n", confusion_matrix(y_true_tta, y_pred_tta))

    # Surface top failure cases for inspection
    mistakes = [
        (samples[i][0], int(y_true_tta[i]), float(y_prob_tta[i]))
        for i in range(len(samples))
        if y_pred_tta[i] != y_true_tta[i]
    ]
    if mistakes:
        print(f"\nSample Misclassified Files ({len(mistakes)} total, showing up to 5):")
        for path, true_lbl, prob in mistakes[:5]:
            lbl_str = "Fake" if true_lbl == 1 else "Real"
            print(f"  - {path} (True: {lbl_str}, Pred P(Fake): {prob:.3f})")


if __name__ == "__main__":
    main()
