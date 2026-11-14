"""Pick the best fake/real decision threshold on the VALIDATION split, then report it on TEST.

The threshold is chosen only from validation data, so the test accuracy stays an honest estimate.
Saves the chosen threshold into checkpoints/threshold.json for the web app to use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix

from evaluate import evaluate_tta
from src.dataset import discover_splits
from src.model import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(Path(args.checkpoint), device)
    splits = discover_splits(Path(args.data_dir))

    y_val, p_val = evaluate_tta(model, splits["valid"], device)
    thresholds = np.linspace(0.05, 0.95, 181)
    accs = [accuracy_score(y_val, (p_val >= t).astype(int)) for t in thresholds]
    best_t = float(thresholds[int(np.argmax(accs))])
    print(f"Validation: acc @0.50 = {accuracy_score(y_val, (p_val >= 0.5).astype(int)) * 100:.2f}% | "
          f"best threshold = {best_t:.3f} -> {max(accs) * 100:.2f}%")

    y_test, p_test = evaluate_tta(model, splits["test"], device)
    for name, t in (("default 0.50", 0.5), (f"tuned {best_t:.3f}", best_t)):
        pred = (p_test >= t).astype(int)
        print(f"\nTEST ({name}): accuracy = {accuracy_score(y_test, pred) * 100:.2f}%")
        print("Confusion matrix [rows=true Real,Fake | cols=pred Real,Fake]:\n", confusion_matrix(y_test, pred))

    out = Path(args.checkpoint).parent / "threshold.json"
    out.write_text(json.dumps({"fake_threshold": best_t}, indent=2), encoding="utf-8")
    print(f"\nSaved threshold to {out}")


if __name__ == "__main__":
    main()
