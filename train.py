"""Two-Phase RTX 3050 Training Pipeline for 95%+ Accuracy on 140k Real & Fake Faces."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import DeepfakeFolderDataset, discover_splits
from src.model import DeepfakeEfficientNet, save_checkpoint
from src.transforms import IMG_SIZE, get_eval_transforms, get_train_transforms


class ModelEMA:
    """Maintains an Exponential Moving Average of model weights for smoother generalization."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * self.decay + msd[k].detach() * (1.0 - self.decay))
            else:
                v.copy_(msd[k])


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits.float(), dim=1)[:, 1]
        total_loss += float(loss.item()) * images.size(0)
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    return {
        "val_loss": total_loss / max(len(y_true), 1),
        "val_acc": float(accuracy_score(y_true, y_pred)),
        "val_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "val_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "val_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "val_auc": auc,
    }


def train_one_epoch(
    model: nn.Module,
    ema: ModelEMA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    device: torch.device,
    grad_accum_steps: int,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)

    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels) / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)

        batch_size = images.size(0)
        running_loss += float(loss.item()) * grad_accum_steps * batch_size
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += batch_size
        pbar.set_postfix(loss=f"{running_loss / total:.4f}", acc=f"{correct / total * 100:.2f}%")

    return running_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeepfakeEfficientNet on 140k+ images (RTX 3050 optimized).")
    parser.add_argument("--data-dir", type=str, required=True, help="Root folder containing train/valid or real/fake.")
    parser.add_argument("--arch", type=str, default="efficientnet_b0", help="timm backbone (e.g., efficientnet_b0, tf_efficientnetv2_s.in21k_ft_in1k).")
    parser.add_argument("--epochs", type=int, default=12, help="Total training epochs (2 warmup + N-2 fine-tune).")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Phase 1 frozen-backbone warmup epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per step (32 fits comfortably in 6GB VRAM).")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps (effective batch = batch_size * grad_accum).")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes.")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to save model checkpoints.")
    parser.add_argument("--max-samples-per-class", type=int, default=None, help="Optional cap per class for quick test runs.")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained backbone initialization.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA unavailable, training on CPU.")

    splits = discover_splits(Path(args.data_dir), max_samples_per_class=args.max_samples_per_class)
    train_ds = DeepfakeFolderDataset(splits["train"], transform=get_train_transforms(IMG_SIZE))
    val_ds = DeepfakeFolderDataset(splits["valid"], transform=get_eval_transforms(IMG_SIZE))
    print(f"Loaded {len(train_ds)} training images, {len(val_ds)} validation images (Real=0, Fake=1).")

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        persistent_workers=args.num_workers > 0,
    )

    model = DeepfakeEfficientNet(
        arch=args.arch,
        pretrained=not args.no_pretrained,
        num_classes=2,
        img_size=IMG_SIZE,
    ).to(device)
    ema = ModelEMA(model, decay=0.999)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_acc = 0.0

    # Phase 1: Head Warmup (frozen backbone)
    warmup_epochs = min(args.warmup_epochs, args.epochs)
    if warmup_epochs > 0:
        print(f"\n--- Phase 1: Head Warmup ({warmup_epochs} epochs, lr=1e-3) ---")
        model.freeze_backbone()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3,
            weight_decay=1e-4,
        )
        for epoch in range(1, warmup_epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(
                model, ema, train_loader, optimizer, scaler, criterion, device, args.grad_accum, epoch, args.epochs
            )
            metrics = evaluate_loader(ema.ema, val_loader, criterion, device)
            elapsed = time.time() - t0
            print(
                f"[Epoch {epoch:02d}/{args.epochs:02d}] "
                f"Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | "
                f"Val Loss: {metrics['val_loss']:.4f} Acc: {metrics['val_acc']*100:.2f}% "
                f"F1: {metrics['val_f1']:.4f} AUC: {metrics['val_auc']:.4f} ({elapsed:.1f}s)"
            )
            record = {"epoch": epoch, "phase": "warmup", "train_loss": tr_loss, "train_acc": tr_acc, **metrics}
            history.append(record)
            save_checkpoint(output_dir / "last_model.pt", ema.ema, epoch, metrics)
            if metrics["val_acc"] >= best_val_acc:
                best_val_acc = metrics["val_acc"]
                save_checkpoint(output_dir / "best_model.pt", ema.ema, epoch, metrics)

    # Phase 2: Full Fine-Tuning with Discriminative LRs & Cosine Annealing
    remaining = args.epochs - warmup_epochs
    if remaining > 0:
        print(f"\n--- Phase 2: Full Fine-Tuning ({remaining} epochs, Cosine Annealing) ---")
        model.unfreeze_backbone()
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": 1e-4},
                {"params": model.head.parameters(), "lr": 3e-4},
            ],
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining, eta_min=1e-6
        )
        for epoch in range(warmup_epochs + 1, args.epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(
                model, ema, train_loader, optimizer, scaler, criterion, device, args.grad_accum, epoch, args.epochs
            )
            scheduler.step()
            metrics = evaluate_loader(ema.ema, val_loader, criterion, device)
            elapsed = time.time() - t0
            print(
                f"[Epoch {epoch:02d}/{args.epochs:02d}] "
                f"Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | "
                f"Val Loss: {metrics['val_loss']:.4f} Acc: {metrics['val_acc']*100:.2f}% "
                f"F1: {metrics['val_f1']:.4f} AUC: {metrics['val_auc']:.4f} ({elapsed:.1f}s)"
            )
            record = {"epoch": epoch, "phase": "finetune", "train_loss": tr_loss, "train_acc": tr_acc, **metrics}
            history.append(record)
            save_checkpoint(output_dir / "last_model.pt", ema.ema, epoch, metrics)
            if metrics["val_acc"] >= best_val_acc:
                best_val_acc = metrics["val_acc"]
                save_checkpoint(output_dir / "best_model.pt", ema.ema, epoch, metrics)

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete! Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Best model saved to: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
