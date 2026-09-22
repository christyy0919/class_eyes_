"""Method 2: sharpness classifier + clear/blur eye models."""

from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.eye_dataset import SHARPNESS_MAP, collect_labeled_samples, filter_by_sharpness
from core.train_loop import dump_json, load_or_create_student_split, set_seed, train_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dual clear/blur eye-state models")
    parser.add_argument("--input", required=True, help="班级根目录，例如 e:\\课堂图片")
    parser.add_argument(
        "--pretrained-eye",
        default=os.path.join(ROOT, "models", "init70", "eye_state_init70.pth"),
    )
    parser.add_argument("--imagenet", default="", help="清晰模糊分类器可选 ImageNet 骨干")
    parser.add_argument("--split", default=os.path.join(ROOT, "models", "sharpness_split.json"))
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "models", "method2"))
    parser.add_argument("--results", default=os.path.join(ROOT, "output", "method2_results.json"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sharp-lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if not os.path.isfile(args.pretrained_eye):
        raise SystemExit("init70 weights not found: {}".format(args.pretrained_eye))

    samples = collect_labeled_samples(args.input, sources=("human",), require_sharpness=True)
    if not samples:
        raise SystemExit(
            "no human labels with sharpness under {}\n"
            "Run: python scripts\\annotate_sharpness.py --input \"{}\"".format(args.input, args.input)
        )
    split = load_or_create_student_split(samples, args.split, val_ratio=args.val_ratio, seed=args.seed)
    train_samples = split["train_samples"]
    val_samples = split["val_samples"]
    train_clear = filter_by_sharpness(train_samples, "clear")
    train_blur = filter_by_sharpness(train_samples, "blur")
    val_clear = filter_by_sharpness(val_samples, "clear")
    val_blur = filter_by_sharpness(val_samples, "blur")
    print(
        "train clear={} blur={} | val clear={} blur={}".format(
            len(train_clear), len(train_blur), len(val_clear), len(val_blur)
        )
    )
    if not train_clear or not train_blur:
        raise SystemExit("need both clear and blur training samples")

    os.makedirs(args.output_dir, exist_ok=True)
    sharp_path = os.path.join(args.output_dir, "sharpness_cls.pth")
    clear_path = os.path.join(args.output_dir, "eye_clear.pth")
    blur_path = os.path.join(args.output_dir, "eye_blur.pth")

    print("=== sharpness classifier ===")
    sharp = train_classifier(
        train_samples,
        val_samples,
        sharp_path,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.sharp_lr,
        patience=args.patience,
        pretrained=args.imagenet,
        target_attr="sharpness",
        label_map=SHARPNESS_MAP,
        class_names=("blur", "clear"),
        extra_meta={"role": "sharpness", "split": args.split},
        curve_path=os.path.join(ROOT, "output", "method2_sharpness_curve.png"),
        num_workers=args.num_workers,
    )

    print("=== clear eye model from init70 ===")
    clear = train_classifier(
        train_clear,
        val_clear or val_samples,
        clear_path,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        checkpoint=args.pretrained_eye,
        extra_meta={"role": "eye_clear", "init70": args.pretrained_eye, "split": args.split},
        curve_path=os.path.join(ROOT, "output", "method2_eye_clear_curve.png"),
        num_workers=args.num_workers,
    )

    print("=== blur eye model from init70 ===")
    blur = train_classifier(
        train_blur,
        val_blur or val_samples,
        blur_path,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        checkpoint=args.pretrained_eye,
        extra_meta={"role": "eye_blur", "init70": args.pretrained_eye, "split": args.split},
        curve_path=os.path.join(ROOT, "output", "method2_eye_blur_curve.png"),
        num_workers=args.num_workers,
    )

    payload = {
        "method": "dual_sharpness_eye",
        "init70": args.pretrained_eye,
        "split": args.split,
        "models": {
            "sharpness": {"weights": sharp_path, "val_accuracy": sharp["best_acc"]},
            "eye_clear": {"weights": clear_path, "val_accuracy": clear["best_acc"]},
            "eye_blur": {"weights": blur_path, "val_accuracy": blur["best_acc"]},
        },
    }
    dump_json(args.results, payload)
    print("wrote", args.results)


if __name__ == "__main__":
    main()
