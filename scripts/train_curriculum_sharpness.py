"""Method 1: finetune init70 on clear images, then mix 10/20/30% blur."""

from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.train_loop import dump_json, load_or_create_student_split, set_seed, subset_by_ratio, train_classifier
from data.eye_dataset import collect_labeled_samples, filter_by_sharpness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curriculum clear-then-blur eye-state training")
    parser.add_argument("--input", required=True, help="班级根目录，例如 e:\\课堂图片")
    parser.add_argument(
        "--pretrained",
        default=os.path.join(ROOT, "models", "init70", "eye_state_init70.pth"),
        help="init70 备份权重",
    )
    parser.add_argument("--split", default=os.path.join(ROOT, "models", "sharpness_split.json"))
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "models", "method1"))
    parser.add_argument("--results", default=os.path.join(ROOT, "output", "method1_results.json"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 2)
    return parser.parse_args()


def require_init70(path: str) -> str:
    if not os.path.isfile(path):
        raise SystemExit("init70 weights not found: {}".format(path))
    return path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    init70 = require_init70(args.pretrained)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

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
    print(
        "train clear={} blur={} | val clear={} blur={}".format(
            len(train_clear),
            len(train_blur),
            len(filter_by_sharpness(val_samples, "clear")),
            len(filter_by_sharpness(val_samples, "blur")),
        )
    )
    if not train_clear:
        raise SystemExit("no clear training samples")

    os.makedirs(args.output_dir, exist_ok=True)
    stages = []
    previous_blur_keys = []
    ratios = [("stage_clear", 0.0), ("stage_blur10", 0.10), ("stage_blur20", 0.20), ("stage_blur30", 0.30)]
    checkpoint = init70
    for name, ratio in ratios:
        blur_part = subset_by_ratio(train_blur, ratio, seed=args.seed, previous=previous_blur_keys)
        previous_blur_keys = [s.key for s in blur_part]
        mix = list(train_clear) + blur_part
        weights_path = os.path.join(args.output_dir, name + ".pth")
        print("=== {}  mix clear={} blur={}  from {} ===".format(
            name, len(train_clear), len(blur_part), checkpoint
        ))
        result = train_classifier(
            mix,
            val_samples,
            weights_path,
            device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            checkpoint=checkpoint,
            extra_meta={
                "stage": name,
                "blur_ratio": ratio,
                "clear_train": len(train_clear),
                "blur_train": len(blur_part),
                "init70": init70,
                "split": args.split,
            },
            curve_path=os.path.join(ROOT, "output", "method1_{}_curve.png".format(name)),
            num_workers=args.num_workers,
        )
        stages.append(
            {
                "stage": name,
                "blur_ratio": ratio,
                "clear_train": len(train_clear),
                "blur_train": len(blur_part),
                "val_accuracy": result["best_acc"],
                "best_epoch": result["best_epoch"],
                "weights": weights_path,
            }
        )
        checkpoint = weights_path

    payload = {
        "method": "curriculum_clear_then_blur",
        "init70": init70,
        "split": args.split,
        "val_students": split.get("val_students"),
        "train_students": split.get("train_students"),
        "stages": stages,
    }
    dump_json(args.results, payload)
    print("wrote", args.results)


if __name__ == "__main__":
    main()
