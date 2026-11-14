"""Flexible dataset discovery and strict label mapping for 140k+ deepfake datasets."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .transforms import CLASS_TO_IDX

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
REAL_KEYWORDS = {"real", "original", "pristine", "authentic"}
FAKE_KEYWORDS = {"fake", "df", "deepfake", "manipulated", "synthetic", "ai"}


def infer_label_from_folder(folder_name: str) -> Optional[int]:
    """Maps folder name to strict label: real=0, fake=1."""
    low = folder_name.strip().lower()
    if low in REAL_KEYWORDS or any(k == low or low.startswith(k + "_") or low.endswith("_" + k) for k in REAL_KEYWORDS):
        return CLASS_TO_IDX["real"]
    if low in FAKE_KEYWORDS or any(k == low or low.startswith(k + "_") or low.endswith("_" + k) for k in FAKE_KEYWORDS):
        return CLASS_TO_IDX["fake"]
    return None


def _collect_class_dir_images(split_dir: Path) -> List[Tuple[Path, int]]:
    samples: List[Tuple[Path, int]] = []
    if not split_dir.is_dir():
        return samples
    for sub in sorted(split_dir.iterdir()):
        if not sub.is_dir():
            continue
        label = infer_label_from_folder(sub.name)
        if label is None:
            continue
        for p in sub.rglob("*"):
            if p.is_file() and p.suffix.lower() in VALID_EXTS:
                samples.append((p, label))
    return samples


def _find_effective_root(data_dir: Path) -> Path:
    """Handles Kaggle 140k nested paths like real_vs_fake/real-vs-fake/train."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    # Direct train/valid/test or real/fake check
    candidates = [data_dir]
    for sub in data_dir.iterdir():
        if sub.is_dir():
            candidates.append(sub)
            for sub2 in sub.iterdir():
                if sub2.is_dir():
                    candidates.append(sub2)

    for cand in candidates:
        child_names = {c.name.lower() for c in cand.iterdir() if c.is_dir()}
        if "train" in child_names and ("valid" in child_names or "val" in child_names or "test" in child_names):
            return cand
        has_real = any(infer_label_from_folder(n) == 0 for n in child_names)
        has_fake = any(infer_label_from_folder(n) == 1 for n in child_names)
        if has_real and has_fake:
            return cand

    raise ValueError(
        f"Could not find 'real'/'fake' or 'train'/'valid' folders inside '{data_dir}'. "
        "Expected either train/(real,fake) + valid/(real,fake) or real/ + fake/."
    )


def discover_splits(
    data_dir: Path,
    max_samples_per_class: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, List[Tuple[Path, int]]]:
    """Discovers train/valid/test splits or performs a stratified 80/10/10 split."""
    root = _find_effective_root(data_dir)
    child_map = {c.name.lower(): c for c in root.iterdir() if c.is_dir()}

    train_samples: List[Tuple[Path, int]] = []
    val_samples: List[Tuple[Path, int]] = []
    test_samples: List[Tuple[Path, int]] = []

    if "train" in child_map:
        train_samples = _collect_class_dir_images(child_map["train"])
        val_dir = child_map.get("valid") or child_map.get("val")
        if val_dir:
            val_samples = _collect_class_dir_images(val_dir)
        if "test" in child_map:
            test_samples = _collect_class_dir_images(child_map["test"])
        if not val_samples and test_samples:
            val_samples = test_samples
        elif not val_samples and train_samples:
            paths = [s[0] for s in train_samples]
            labels = [s[1] for s in train_samples]
            tr_p, va_p, tr_l, va_l = train_test_split(
                paths, labels, test_size=0.1, stratify=labels, random_state=seed
            )
            train_samples = list(zip(tr_p, tr_l))
            val_samples = list(zip(va_p, va_l))
    else:
        all_samples = _collect_class_dir_images(root)
        if not all_samples:
            raise ValueError(f"No valid images found under '{root}'.")
        paths = [s[0] for s in all_samples]
        labels = [s[1] for s in all_samples]
        tr_p, tmp_p, tr_l, tmp_l = train_test_split(
            paths, labels, test_size=0.2, stratify=labels, random_state=seed
        )
        va_p, te_p, va_l, te_l = train_test_split(
            tmp_p, tmp_l, test_size=0.5, stratify=tmp_l, random_state=seed
        )
        train_samples = list(zip(tr_p, tr_l))
        val_samples = list(zip(va_p, va_l))
        test_samples = list(zip(te_p, te_l))

    if max_samples_per_class is not None:
        train_samples = _subsample_balanced(train_samples, max_samples_per_class, seed)
        val_samples = _subsample_balanced(val_samples, max(100, max_samples_per_class // 5), seed)
        if test_samples:
            test_samples = _subsample_balanced(test_samples, max(100, max_samples_per_class // 5), seed)

    return {"train": train_samples, "valid": val_samples, "test": test_samples, "effective_root": root}


def _subsample_balanced(
    samples: List[Tuple[Path, int]],
    max_per_class: int,
    seed: int,
) -> List[Tuple[Path, int]]:
    rng = random.Random(seed)
    by_cls: Dict[int, List[Tuple[Path, int]]] = {0: [], 1: []}
    for item in samples:
        by_cls[item[1]].append(item)
    out: List[Tuple[Path, int]] = []
    for cls in (0, 1):
        items = by_cls[cls]
        rng.shuffle(items)
        out.extend(items[:max_per_class])
    rng.shuffle(out)
    return out


class DeepfakeFolderDataset(Dataset):
    """PyTorch Dataset enforcing RGB conversion and strict label indices (Real=0, Fake=1)."""

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        transform: Optional[Callable] = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            rgb = img.convert("RGB")
        if self.transform is not None:
            tensor = self.transform(rgb)
        else:
            tensor = rgb
        return tensor, label
