import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a date fruit classifier with PyTorch + timm."
    )
    parser.add_argument("--train-dir", type=str, default="train")
    parser.add_argument("--val-dir", type=str, default="test")
    parser.add_argument("--model-name", type=str, default="efficientnetv2_s")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--save-name", type=str, default="best_model.pt")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_transforms(image_size: int):
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03
            ),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_tfms, eval_tfms


def build_weighted_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    targets = torch.tensor(dataset.targets)
    class_counts = torch.bincount(targets)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[targets]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
    scaler=None,
):
    is_train = optimizer is not None
    model.train(is_train)
    epoch_loss = 0.0
    epoch_acc = 0.0
    n_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bs = images.size(0)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                with torch.autocast(
                    device_type=device.type, enabled=device.type == "cuda"
                ):
                    logits = model(images)
                    loss = criterion(logits, targets)

        epoch_loss += loss.item() * bs
        epoch_acc += top1_accuracy(logits, targets) * bs
        n_samples += bs

    return epoch_loss / n_samples, epoch_acc / n_samples


def main():
    args = parse_args()
    seed_everything(args.seed)

    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    train_tfms, eval_tfms = build_transforms(args.image_size)
    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tfms)

    if train_ds.class_to_idx != val_ds.class_to_idx:
        raise ValueError(
            "Class mapping mismatch between train and val directories. "
            "Ensure both contain the same class folder names."
        )

    sampler = build_weighted_sampler(train_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(train_ds.classes)
    model = timm.create_model(
        args.model_name,
        pretrained=True,
        num_classes=num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    total_steps = args.epochs
    warmup_steps = max(1, args.warmup_epochs)

    def lr_lambda(epoch_idx: int) -> float:
        if epoch_idx < warmup_steps:
            return float(epoch_idx + 1) / float(warmup_steps)
        progress = (epoch_idx - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_val_acc = 0.0
    best_epoch = -1
    epochs_no_improve = 0
    best_path = output_dir / args.save_name

    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    print(f"Device: {device}")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(f"Classes ({num_classes}): {train_ds.classes}")
    print(f"Model: {args.model_name}")

    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
        )
        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
        )
        scheduler.step()
        elapsed = time.time() - start

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} | "
            f"lr {lr:.2e} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(
                {
                    "model_name": args.model_name,
                    "image_size": args.image_size,
                    "num_classes": num_classes,
                    "class_to_idx": train_ds.class_to_idx,
                    "idx_to_class": idx_to_class,
                    "state_dict": model.state_dict(),
                    "val_acc": best_val_acc,
                    "epoch": best_epoch,
                    "args": vars(args),
                },
                best_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(
                    f"Early stopping: no improvement for {args.patience} epochs."
                )
                break

    metrics_path = output_dir / "training_summary.json"
    metrics_path.write_text(
        json.dumps(
            {
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "checkpoint": str(best_path),
                "model_name": args.model_name,
                "image_size": args.image_size,
            },
            indent=2,
        )
    )
    print(f"Best val acc: {best_val_acc:.4f} @ epoch {best_epoch}")
    print(f"Saved checkpoint: {best_path}")
    print(f"Saved summary: {metrics_path}")


if __name__ == "__main__":
    main()
