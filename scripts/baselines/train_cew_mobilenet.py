"""Experiment 3: CEW / classroom cross-domain MobileNetV2."""

from __future__ import annotations

import argparse
import os
import random
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.eye_state import build_eye_model
from core.train_loop import dump_json, load_json, set_seed, train_classifier
from data.eye_dataset import EyeSample, list_images
from scripts.baselines.common import (
    CEW_B_FACE,
    CLASSROOM,
    MNV2_CLASSROOM,
    load_classroom_split,
    load_method1_train_pool,
)
from scripts.evaluate_dual_sharpness import evaluate_single_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CEW / classroom cross-domain MobileNetV2")
    parser.add_argument("--cew", default=CEW_B_FACE)
    parser.add_argument("--input", default=CLASSROOM, help="classroom root for frozen val")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--weights", default=os.path.join(ROOT, "models", "baselines", "mnv2_cew.pth"))
    parser.add_argument("--mix-weights", default=os.path.join(ROOT, "models", "baselines", "mnv2_mix.pth"))
    parser.add_argument("--output", default=os.path.join(ROOT, "output", "baseline_exp3.json"))
    parser.add_argument("--skip-train", action="store_true", help="alias of --skip-cew-train")
    parser.add_argument("--skip-cew-train", action="store_true")
    parser.add_argument("--skip-mix-train", action="store_true")
    return parser.parse_args()


def collect_cew_faces(root: str):
    if not os.path.isdir(root):
        raise SystemExit("CEW Dataset B faces not found: {}".format(root))
    samples = []
    for folder, label in (("ClosedFace", "closed"), ("OpenFace", "open")):
        dirpath = os.path.join(root, folder)
        if not os.path.isdir(dirpath):
            raise SystemExit("missing {} under {}".format(folder, root))
        for name in list_images(dirpath):
            samples.append(
                EyeSample(
                    student_id="cew_{}_{}".format(label, os.path.splitext(name)[0]),
                    student_dir=dirpath,
                    image_name=name,
                    label=label,
                    rel_path="{}/{}".format(folder, name),
                    source="cew",
                    sharpness="",
                )
            )
    if not samples:
        raise SystemExit("no CEW face images under {}".format(root))
    return samples


def image_split(samples, val_ratio: float, seed: int):
    ordered = list(samples)
    random.Random(seed).shuffle(ordered)
    n_val = max(1, int(round(len(ordered) * val_ratio)))
    return ordered[n_val:], ordered[:n_val]


def maybe_train(skip: bool, weights_path: str, train_samples, val_samples, device, args, extra_ckpt, extra_meta, curve_name):
    train_info = {}
    if skip and os.path.isfile(weights_path):
        print("skip train, use", weights_path)
        meta_path = os.path.splitext(weights_path)[0] + "_meta.json"
        if os.path.isfile(meta_path):
            meta = load_json(meta_path)
            train_info = {
                "best_epoch": meta.get("best_epoch"),
                "best_acc": meta.get("val_accuracy"),
            }
        return train_info
    os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)
    model = build_eye_model(num_classes=2, pretrained=None)
    return train_classifier(
        train_samples,
        val_samples,
        weights_path,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        extra_ckpt=extra_ckpt,
        extra_meta=extra_meta,
        curve_path=os.path.join(ROOT, "output", curve_name),
        num_workers=args.num_workers,
        model=model,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    skip_cew = args.skip_cew_train or args.skip_train

    cew_samples = collect_cew_faces(args.cew)
    cew_train, cew_val = image_split(cew_samples, args.val_ratio, args.seed)
    print("CEW faces total={} train={} self-val={}".format(len(cew_samples), len(cew_train), len(cew_val)))
    classroom_pool, classroom_val = load_method1_train_pool(args.input)
    _human_train, _, split = load_classroom_split(args.input)
    print("classroom pool={} val={}".format(len(classroom_pool), len(classroom_val)))

    print("=== CEW-only MobileNetV2 ===")
    cew_train_info = maybe_train(
        skip_cew,
        args.weights,
        cew_train,
        cew_val,
        device,
        args,
        {"domain": "cew_dataset_b_faces"},
        {
            "domain": "cew_dataset_b_faces",
            "cew_root": args.cew,
            "from_scratch": True,
            "init70_loaded": False,
        },
        "baseline_cew_curve.png",
    )

    print("=== CEW + classroom mix MobileNetV2 ===")
    mix_train = list(cew_samples) + list(classroom_pool)
    print("mix train={} (cew={} classroom={})".format(len(mix_train), len(cew_samples), len(classroom_pool)))
    mix_train_info = maybe_train(
        args.skip_mix_train,
        args.mix_weights,
        mix_train,
        classroom_val,
        device,
        args,
        {"domain": "cew_plus_classroom"},
        {
            "domain": "cew_plus_classroom",
            "cew_root": args.cew,
            "cew_samples": len(cew_samples),
            "classroom_pool": len(classroom_pool),
            "from_scratch": True,
            "init70_loaded": False,
        },
        "baseline_mix_curve.png",
    )

    cew_self = evaluate_single_model(args.weights, cew_val, device, args.batch_size, 0)
    cew_self["name"] = "mnv2_cew_selfval"
    classroom_from_cew = evaluate_single_model(args.weights, classroom_val, device, args.batch_size, 0)
    classroom_from_cew["name"] = "mnv2_cew_on_classroom"

    classroom_trained = evaluate_single_model(MNV2_CLASSROOM, classroom_val, device, args.batch_size, 0)
    classroom_trained["name"] = "mnv2_classroom_on_classroom"
    classroom_on_cew = evaluate_single_model(MNV2_CLASSROOM, cew_val, device, args.batch_size, 0)
    classroom_on_cew["name"] = "mnv2_classroom_on_cew"

    mix_on_classroom = evaluate_single_model(args.mix_weights, classroom_val, device, args.batch_size, 0)
    mix_on_classroom["name"] = "mnv2_mix_on_classroom"

    payload = {
        "experiment": "cew_cross_domain",
        "cew_root": args.cew,
        "cew_total": len(cew_samples),
        "cew_train": len(cew_train),
        "cew_self_val": len(cew_val),
        "classroom_pool": len(classroom_pool),
        "classroom_val": len(classroom_val),
        "mix_train": len(mix_train),
        "val_students": split.get("val_students"),
        "train_best_epoch": cew_train_info.get("best_epoch"),
        "train_cew_self_acc": cew_train_info.get("best_acc"),
        "mix_best_epoch": mix_train_info.get("best_epoch"),
        "mix_val_acc": mix_train_info.get("best_acc"),
        "weights": args.weights,
        "mix_weights": args.mix_weights,
        "results": {
            "mnv2_cew_selfval": cew_self,
            "mnv2_cew_on_classroom": classroom_from_cew,
            "mnv2_classroom_on_classroom": classroom_trained,
            "mnv2_classroom_on_cew": classroom_on_cew,
            "mnv2_mix_on_classroom": mix_on_classroom,
        },
    }
    dump_json(args.output, payload)
    print("wrote", args.output)
    for name, item in payload["results"].items():
        print("  {} acc={}".format(name, item.get("accuracy")))


if __name__ == "__main__":
    main()
