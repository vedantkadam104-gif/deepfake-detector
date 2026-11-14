"""Diagnostic script to inspect your 1.4 lakh dataset structure, balance, and image integrity."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from PIL import Image

from src.dataset import discover_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and validate a deepfake image dataset.")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to your 140k dataset root folder.")
    parser.add_argument("--sample-check", type=int, default=200, help="Number of random images to open & verify.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    info = discover_splits(data_dir)
    print(f"\nEffective Dataset Root: {info['effective_root']}")
    print("Label Mapping (Locked): Real -> 0 | Fake -> 1\n")

    for split in ("train", "valid", "test"):
        samples = info[split]
        counts = Counter(lbl for _, lbl in samples)
        print(f"  [{split.upper():5s}] Total: {len(samples):7d} | Real (0): {counts[0]:6d} | Fake (1): {counts[1]:6d}")

    # Integrity & resolution check on random subset
    all_imgs = info["train"] + info["valid"]
    if all_imgs:
        sample = random.sample(all_imgs, min(args.sample_check, len(all_imgs)))
        sizes = Counter()
        modes = Counter()
        corrupt = 0
        for p, _ in sample:
            try:
                with Image.open(p) as im:
                    sizes[im.size] += 1
                    modes[im.mode] += 1
            except Exception:
                corrupt += 1
        print(f"\nSampled {len(sample)} images -> Corrupt: {corrupt}")
        print(f"  Top resolutions: {sizes.most_common(5)}")
        print(f"  Color modes:     {modes.most_common(5)}")

    print("\nReady to train on RTX 3050! Run:")
    print(f'  .\\.venv\\Scripts\\python.exe train.py --data-dir "{args.data_dir}" --epochs 12\n')


if __name__ == "__main__":
    main()
