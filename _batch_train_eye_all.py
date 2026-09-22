"""Train MobileNetV2 eye-state classifier on all per-student eye_labels.csv files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.eye_state import INPUT_SIZE, LABEL_MAP, build_eye_model
from data.eye_dataset import (
    EyeStateDataset,
    balance_binary_samples,
    collect_labeled_samples,
    exclude_students,
    split_by_student,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-train eye open/closed MobileNetV2")
    parser.add_argument("--input", required=True, help="班级文件夹（内含学生小文件夹）")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 2)
    parser.add_argument("--pretrained", default="", help="可选 ImageNet MobileNetV2 权重")
    parser.add_argument(
        "--exclude-student",
        action="append",
        default=[],
        help="排除学生，可重复。例如 --exclude-student 142",
    )
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        help="训练集欠采样多数类，使 open/closed 为 1:1；验证集保持原始比例",
    )
    parser.add_argument(
        "--human-only",
        action="store_true",
        help="只用 source=human 的标签，忽略伪标签",
    )
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "models"))
    parser.add_argument("--curve-path", default=os.path.join(ROOT, "output", "eye_train_curve.png"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(samples) -> torch.Tensor:
    counts = Counter(sample.label for sample in samples)
    weights = []
    total = float(len(samples))
    n_classes = len(LABEL_MAP)
    for name in ("closed", "open"):
        count = max(counts.get(name, 0), 1)
        weights.append(total / (n_classes * count))
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        iterator = tqdm(loader, leave=False, desc="train" if train else "val")
        for images, targets, _keys in iterator:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            batch = targets.size(0)
            total_loss += loss.item() * batch
            pred = logits.argmax(dim=1)
            correct += int((pred == targets).sum().item())
            total += batch
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / float(total)


def dump_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def plot_curves(history: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    samples = collect_labeled_samples(
        args.input,
        sources=("human",) if args.human_only else None,
    )
    if args.exclude_student:
        samples = exclude_students(samples, args.exclude_student)
    if not samples:
        raise SystemExit("no labeled samples under {}".format(args.input))
    counts = Counter(sample.label for sample in samples)
    print("labeled samples: {}  open={}  closed={}  students={}".format(
        len(samples),
        counts.get("open", 0),
        counts.get("closed", 0),
        len({s.student_id for s in samples}),
    ))

    split = split_by_student(samples, val_ratio=args.val_ratio, seed=args.seed)
    train_samples = split["train_samples"]
    val_samples = split["val_samples"]
    if args.balance_classes:
        train_samples = balance_binary_samples(train_samples, seed=args.seed)
        split["train_samples"] = train_samples
        split["train_keys"] = [s.key for s in train_samples]
        split["train_count"] = len(train_samples)
        split["balanced"] = True
    else:
        split["balanced"] = False
    split["excluded_students"] = args.exclude_student
    split["human_only"] = bool(args.human_only)
    if not train_samples:
        raise SystemExit("train split is empty")
    train_counts = Counter(sample.label for sample in train_samples)
    val_counts = Counter(sample.label for sample in val_samples)
    print("train open={} closed={} | val open={} closed={}".format(
        train_counts.get("open", 0),
        train_counts.get("closed", 0),
        val_counts.get("open", 0),
        val_counts.get("closed", 0),
    ))

    os.makedirs(args.output_dir, exist_ok=True)
    split_path = os.path.join(args.output_dir, "eye_state_split.json")
    split_info = {key: value for key, value in split.items()
                  if key not in ("train_samples", "val_samples")}
    dump_json(split_path, split_info)
    print("split mode={} train={} val={} -> {}".format(
        split["mode"], split["train_count"], split["val_count"], split_path
    ))

    train_loader = DataLoader(
        EyeStateDataset(train_samples, train=True, input_size=INPUT_SIZE),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = None
    if val_samples:
        val_loader = DataLoader(
            EyeStateDataset(val_samples, train=False, input_size=INPUT_SIZE),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    pretrained = args.pretrained.strip() or None
    model = build_eye_model(num_classes=2, pretrained=pretrained)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_samples).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = -1.0
    best_epoch = 0
    stale = 0
    weights_path = os.path.join(args.output_dir, "eye_state_best.pth")
    meta_path = os.path.join(args.output_dir, "eye_state_best_meta.json")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        if val_loader is not None:
            val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        else:
            val_loss, val_acc = train_loss, train_acc
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            "epoch {:03d}/{}  train loss={:.4f} acc={:.4f}  val loss={:.4f} acc={:.4f}".format(
                epoch, args.epochs, train_loss, train_acc, val_loss, val_acc
            )
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "label_map": LABEL_MAP,
                    "input_size": INPUT_SIZE,
                    "val_accuracy": best_acc,
                    "epoch": epoch,
                },
                weights_path,
            )
            dump_json(
                meta_path,
                {
                    "label_map": LABEL_MAP,
                    "input_size": INPUT_SIZE,
                    "val_accuracy": round(best_acc, 4),
                    "samples": len(samples),
                    "train_samples": len(train_samples),
                    "val_samples": len(val_samples),
                    "split_seed": args.seed,
                    "val_ratio": args.val_ratio,
                    "split_mode": split["mode"],
                    "excluded_students": args.exclude_student,
                    "human_only": bool(args.human_only),
                    "balanced": bool(args.balance_classes),
                    "best_epoch": best_epoch,
                    "weights": os.path.relpath(weights_path, ROOT),
                },
            )
            print("  saved best acc={:.4f} -> {}".format(best_acc, weights_path))
        else:
            stale += 1
            if stale >= args.patience:
                print("early stop at epoch {} (patience={})".format(epoch, args.patience))
                break

    plot_curves(history, args.curve_path)
    print("best val acc={:.4f} @ epoch {}".format(best_acc, best_epoch))
    print("curve:", args.curve_path)
    print("meta:", meta_path)


if __name__ == "__main__":
    main()
