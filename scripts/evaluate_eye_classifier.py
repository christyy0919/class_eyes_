"""Evaluate the trained eye-state MobileNetV2 on the saved val split or all labels."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.eye_state import INDEX_TO_LABEL, INPUT_SIZE, LABEL_MAP, load_eye_model
from data.eye_dataset import EyeStateDataset, collect_labeled_samples, filter_samples_by_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate eye open/closed classifier")
    parser.add_argument("--input", required=True, help="班级文件夹（内含学生小文件夹）")
    parser.add_argument(
        "--weights",
        default=os.path.join(ROOT, "models", "eye_state_best.pth"),
        help="训练得到的 pth 权重",
    )
    parser.add_argument(
        "--split",
        default=os.path.join(ROOT, "models", "eye_state_split.json"),
        help="训练时保存的划分文件",
    )
    parser.add_argument("--full", action="store_true", help="在全部已标注样本上测试")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 2)
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT, "output", "eye_eval.json"),
        help="指标 JSON 输出路径",
    )
    return parser.parse_args()


def select_samples(all_samples, args):
    if args.full:
        return all_samples, "full"
    if not os.path.isfile(args.split):
        raise SystemExit(
            "split file not found: {}\nTrain first, or pass --full to test all labeled images.".format(
                args.split
            )
        )
    with open(args.split, "r", encoding="utf-8") as handle:
        split = json.load(handle)
    keys = split.get("val_keys") or []
    if not keys:
        raise SystemExit("split file has no val_keys: {}".format(args.split))
    selected = filter_samples_by_keys(all_samples, keys)
    if not selected:
        raise SystemExit("no val samples matched the split under {}".format(args.input))
    return selected, "val"


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    all_samples = collect_labeled_samples(args.input)
    if not all_samples:
        raise SystemExit("no labeled samples under {}".format(args.input))
    samples, split_name = select_samples(all_samples, args)
    counts = Counter(sample.label for sample in samples)
    print("eval split={}  samples={}  open={}  closed={}".format(
        split_name, len(samples), counts.get("open", 0), counts.get("closed", 0)
    ))

    model, ckpt_meta = load_eye_model(args.weights, device=device)
    loader = DataLoader(
        EyeStateDataset(samples, train=False, input_size=INPUT_SIZE),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    criterion = nn.CrossEntropyLoss()

    y_true = []
    y_pred = []
    total_loss = 0.0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, targets, _keys in tqdm(loader, desc="eval"):
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            pred = logits.argmax(dim=1)
            batch = targets.size(0)
            total_loss += loss.item() * batch
            total += batch
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

    accuracy = (sum(int(a == b) for a, b in zip(y_true, y_pred)) / float(total)) if total else 0.0
    avg_loss = total_loss / total if total else 0.0
    labels = [0, 1]
    target_names = [INDEX_TO_LABEL[i] for i in labels]
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, zero_division=0
    )
    print("accuracy={:.4f}  loss={:.4f}".format(accuracy, avg_loss))
    print(report)
    print("confusion_matrix [rows=true closed,open; cols=pred closed,open]:")
    print(matrix)

    try:
        model_ref = os.path.relpath(args.weights, ROOT)
    except ValueError:
        model_ref = args.weights
    payload = {
        "model": model_ref,
        "data": os.path.abspath(args.input),
        "split": split_name,
        "split_file": None if args.full else os.path.abspath(args.split),
        "samples": total,
        "accuracy": round(accuracy, 4),
        "loss": round(avg_loss, 4),
        "label_map": LABEL_MAP,
        "input_size": INPUT_SIZE,
        "per_class": {
            INDEX_TO_LABEL[i]: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i in labels
        },
        "confusion_matrix": matrix,
        "checkpoint_meta": {
            key: ckpt_meta.get(key)
            for key in ("val_accuracy", "epoch", "input_size", "label_map")
            if isinstance(ckpt_meta, dict) and key in ckpt_meta
        },
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
