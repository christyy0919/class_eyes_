"""Evaluate dual sharpness/eye models: softmax mix and hard routing."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.eye_state import INPUT_SIZE, LABEL_MAP, load_eye_model
from core.train_loop import accuracy_of, confusion_from_preds, dump_json, load_json
from data.eye_dataset import (
    EyeSample,
    EyeStateDataset,
    SHARPNESS_MAP,
    collect_labeled_samples,
    filter_samples_by_keys,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dual clear/blur eye routing")
    parser.add_argument("--input", required=True)
    parser.add_argument("--split", default=os.path.join(ROOT, "models", "sharpness_split.json"))
    parser.add_argument("--sharp-weights", default=os.path.join(ROOT, "models", "method2", "sharpness_cls.pth"))
    parser.add_argument("--clear-weights", default=os.path.join(ROOT, "models", "method2", "eye_clear.pth"))
    parser.add_argument("--blur-weights", default=os.path.join(ROOT, "models", "method2", "eye_blur.pth"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 2)
    parser.add_argument("--output", default=os.path.join(ROOT, "output", "method2_eval.json"))
    return parser.parse_args()


def load_val_samples(root: str, split_path: str) -> List[EyeSample]:
    samples = collect_labeled_samples(root, sources=("human",), require_sharpness=True)
    if not samples:
        raise SystemExit("no human+sharpness samples under {}".format(root))
    if not os.path.isfile(split_path):
        raise SystemExit("split not found: {}".format(split_path))
    split = load_json(split_path)
    keys = split.get("val_keys") or []
    selected = filter_samples_by_keys(samples, keys)
    if not selected:
        val_students = set(split.get("val_students") or [])
        selected = [s for s in samples if s.student_id in val_students]
    if not selected:
        raise SystemExit("no val samples matched {}".format(split_path))
    return selected


def predict_logits(model, samples: Sequence[EyeSample], device, batch_size: int, num_workers: int, target_attr="label", label_map=None):
    loader = DataLoader(
        EyeStateDataset(
            samples,
            train=False,
            input_size=INPUT_SIZE,
            target_attr=target_attr,
            label_map=label_map or LABEL_MAP,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    keys = []
    logits_out = []
    targets = []
    model.eval()
    with torch.no_grad():
        for images, batch_targets, batch_keys in tqdm(loader, leave=False, desc="infer"):
            images = images.to(device)
            logits = model(images)
            logits_out.append(logits.cpu())
            targets.extend(batch_targets.tolist())
            keys.extend(batch_keys)
    return keys, torch.cat(logits_out, dim=0), targets


def summarize(y_true: List[int], y_pred: List[int], samples: Sequence[EyeSample]) -> dict:
    overall = accuracy_of(y_true, y_pred)
    by_sharp = defaultdict(lambda: {"true": [], "pred": []})
    for sample, t, p in zip(samples, y_true, y_pred):
        by_sharp[sample.sharpness]["true"].append(t)
        by_sharp[sample.sharpness]["pred"].append(p)
    return {
        "samples": len(y_true),
        "accuracy": round(overall, 4),
        "clear_accuracy": round(accuracy_of(by_sharp["clear"]["true"], by_sharp["clear"]["pred"]), 4),
        "blur_accuracy": round(accuracy_of(by_sharp["blur"]["true"], by_sharp["blur"]["pred"]), 4),
        "clear_samples": len(by_sharp["clear"]["true"]),
        "blur_samples": len(by_sharp["blur"]["true"]),
        "confusion_matrix": confusion_from_preds(y_true, y_pred),
    }


def evaluate_single_model(weights: str, samples: Sequence[EyeSample], device, batch_size: int, num_workers: int) -> dict:
    model, _meta = load_eye_model(weights, device=device)
    keys, logits, targets = predict_logits(model, samples, device, batch_size, num_workers)
    preds = logits.argmax(dim=1).tolist()
    keyed = {k: sample for k, sample in zip(keys, samples)}
    ordered = [keyed[k] for k in keys]
    result = summarize(targets, preds, ordered)
    result["model"] = weights
    return result


def evaluate_dual(
    samples: Sequence[EyeSample],
    sharp_weights: str,
    clear_weights: str,
    blur_weights: str,
    device,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Dict[str, dict]:
    sharp_net, _ = load_eye_model(sharp_weights, device=device)
    clear_net, _ = load_eye_model(clear_weights, device=device)
    blur_net, _ = load_eye_model(blur_weights, device=device)
    keys, sharp_logits, _sharp_targets = predict_logits(
        sharp_net, samples, device, batch_size, num_workers, target_attr="sharpness", label_map=SHARPNESS_MAP
    )
    _, clear_logits, targets = predict_logits(clear_net, samples, device, batch_size, num_workers)
    _, blur_logits, _ = predict_logits(blur_net, samples, device, batch_size, num_workers)
    sharp_prob = torch.softmax(sharp_logits, dim=1)
    clear_prob = torch.softmax(clear_logits, dim=1)
    blur_prob = torch.softmax(blur_logits, dim=1)
    s = sharp_prob[:, SHARPNESS_MAP["clear"]].unsqueeze(1)
    mixed = s * clear_prob + (1.0 - s) * blur_prob
    mix_pred = mixed.argmax(dim=1).tolist()
    hard_pred = []
    for i in range(len(keys)):
        if s[i, 0] > 0.5:
            hard_pred.append(int(clear_logits[i].argmax().item()))
        else:
            hard_pred.append(int(blur_logits[i].argmax().item()))
    sample_by_key = {sample.key: sample for sample in samples}
    ordered = [sample_by_key[k] for k in keys]
    return {
        "weighted": summarize(targets, mix_pred, ordered),
        "hard_route": summarize(targets, hard_pred, ordered),
        "sharpness_acc": round(accuracy_of(_sharp_targets, sharp_logits.argmax(dim=1).tolist()), 4),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_val_samples(args.input, args.split)
    result = evaluate_dual(
        samples,
        args.sharp_weights,
        args.clear_weights,
        args.blur_weights,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dump_json(args.output, result)
    print("weighted acc={}  hard acc={}  sharpness acc={}".format(
        result["weighted"]["accuracy"],
        result["hard_route"]["accuracy"],
        result["sharpness_acc"],
    ))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
