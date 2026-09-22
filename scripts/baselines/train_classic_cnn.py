"""Experiment 2: train LeNet or AlexNet on the method-1 classroom pool, eval frozen val."""

from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.train_loop import dump_json, load_json, set_seed, train_classifier
from scripts.baselines.classic_cnns import build_classic
from scripts.baselines.common import CLASSROOM, MNV2_CLASSROOM, load_method1_train_pool
from scripts.evaluate_dual_sharpness import evaluate_single_model, predict_logits, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classic CNN baselines on classroom pool")
    parser.add_argument("--arch", required=True, choices=["lenet", "alexnet"])
    parser.add_argument("--input", default=CLASSROOM)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "models", "baselines"))
    parser.add_argument("--results", default=os.path.join(ROOT, "output", "baseline_exp2.json"))
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def eval_classic(arch: str, weights: str, samples, device, batch_size: int) -> dict:
    model = build_classic(arch)
    try:
        raw = torch.load(weights, map_location=device, weights_only=False)
    except TypeError:
        raw = torch.load(weights, map_location=device)
    state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    model.load_state_dict(state)
    model.to(device)
    keys, logits, targets = predict_logits(model, samples, device, batch_size, 0)
    preds = logits.argmax(dim=1).tolist()
    keyed = {k: sample for k, sample in zip(keys, samples)}
    ordered = [keyed[k] for k in keys]
    result = summarize(targets, preds, ordered)
    result["model"] = weights
    result["name"] = arch
    return result


def merge_result(path: str, arch: str, payload: dict) -> dict:
    existing = load_json(path) if os.path.isfile(path) else {
        "experiment": "arch_baselines",
        "models": {},
    }
    existing["experiment"] = "arch_baselines"
    existing.setdefault("models", {})
    existing["models"][arch] = payload
    existing["train_samples"] = payload.get("train_samples")
    existing["val_samples"] = payload.get("val_samples")
    dump_json(path, existing)
    return existing


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "arch:", args.arch)
    train_samples, val_samples = load_method1_train_pool(args.input)
    print("train pool={} val={}".format(len(train_samples), len(val_samples)))
    os.makedirs(args.output_dir, exist_ok=True)
    weights_path = os.path.join(args.output_dir, args.arch + ".pth")

    train_info = {}
    if args.skip_train and os.path.isfile(weights_path):
        print("skip train, use", weights_path)
    else:
        model = build_classic(args.arch)
        train_info = train_classifier(
            train_samples,
            val_samples,
            weights_path,
            device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            extra_ckpt={"arch": args.arch},
            extra_meta={"arch": args.arch, "from_scratch": True, "imagenet_pretrained": False},
            curve_path=os.path.join(ROOT, "output", "baseline_{}_curve.png".format(args.arch)),
            num_workers=args.num_workers,
            model=model,
        )

    eval_result = eval_classic(args.arch, weights_path, val_samples, device, args.batch_size)
    eval_result["train_samples"] = len(train_samples)
    eval_result["val_samples"] = len(val_samples)
    eval_result["best_epoch"] = train_info.get("best_epoch")
    eval_result["train_val_acc"] = train_info.get("best_acc")
    payload = eval_result

    existing = merge_result(args.results, args.arch, payload)
    if "mobilenetv2_selftrain" not in existing.get("models", {}) and os.path.isfile(MNV2_CLASSROOM):
        mnv2 = evaluate_single_model(MNV2_CLASSROOM, val_samples, device, args.batch_size, 0)
        mnv2["name"] = "mobilenetv2_selftrain"
        existing["models"]["mobilenetv2_selftrain"] = mnv2
        dump_json(args.results, existing)

    print("wrote", args.results)
    for name, item in existing.get("models", {}).items():
        print("  {} acc={}".format(name, item.get("accuracy")))


if __name__ == "__main__":
    main()
