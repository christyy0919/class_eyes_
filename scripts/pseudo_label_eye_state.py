"""Pre-annotate unlabeled head crops with MobileNetV2 pseudo labels."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.eye_state import INDEX_TO_LABEL, INPUT_SIZE, load_eye_model
from data.eye_dataset import (
    HUMAN_SOURCE,
    PSEUDO_SOURCE,
    LabelRecord,
    _student_token_match,
    build_transforms,
    discover_student_dirs,
    list_images,
    load_student_label_records,
    save_student_label_records,
)


class UnlabeledCropDataset(Dataset):
    def __init__(self, items, input_size=INPUT_SIZE):
        self.items = items
        self.transform = build_transforms(train=False, input_size=input_size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        image = Image.open(item["abs_path"]).convert("RGB")
        return self.transform(image), index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pseudo-label remaining eye images")
    parser.add_argument("--input", required=True, help="班级文件夹或学生文件夹")
    parser.add_argument(
        "--weights",
        default=os.path.join(ROOT, "models", "eye_state_best.pth"),
        help="已训练的眼状态权重",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.55,
        help="接受伪标签的最低置信度（默认 0.55，低于人工复核常用阈值）",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 2)
    parser.add_argument(
        "--exclude-student",
        action="append",
        default=[],
        help="排除学生，可重复，例如 --exclude-student 142",
    )
    parser.add_argument(
        "--overwrite-pseudo",
        action="store_true",
        help="覆盖已有伪标签；人工标注始终不改",
    )
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写 CSV")
    parser.add_argument(
        "--report",
        default=os.path.join(ROOT, "output", "eye_pseudo_label_report.json"),
    )
    return parser.parse_args()


def collect_unlabeled_items(root, exclude_tokens, overwrite_pseudo):
    items = []
    skipped_human = 0
    skipped_pseudo = 0
    for student_dir in discover_student_dirs(root):
        student_id = os.path.basename(student_dir.rstrip("\\/"))
        if any(_student_token_match(student_id, token) for token in exclude_tokens):
            continue
        records = load_student_label_records(student_dir)
        for image_name in list_images(student_dir):
            rec = records.get(image_name)
            if rec is None:
                items.append(
                    {
                        "student_dir": student_dir,
                        "student_id": student_id,
                        "image_name": image_name,
                        "abs_path": os.path.join(student_dir, image_name),
                    }
                )
                continue
            if rec.source == HUMAN_SOURCE:
                skipped_human += 1
                continue
            if rec.source == PSEUDO_SOURCE and not overwrite_pseudo:
                skipped_pseudo += 1
                continue
            items.append(
                {
                    "student_dir": student_dir,
                    "student_id": student_id,
                    "image_name": image_name,
                    "abs_path": os.path.join(student_dir, image_name),
                }
            )
    return items, skipped_human, skipped_pseudo


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.conf <= 1.0:
        raise SystemExit("--conf must be between 0 and 1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("weights:", args.weights)
    print("confidence threshold:", args.conf)

    items, skipped_human, skipped_pseudo = collect_unlabeled_items(
        os.path.abspath(args.input),
        args.exclude_student,
        args.overwrite_pseudo,
    )
    print(
        "to_predict={}  keep_human={}  keep_existing_pseudo={}".format(
            len(items), skipped_human, skipped_pseudo
        )
    )
    if not items:
        print("no unlabeled images")
        return

    model, _meta = load_eye_model(args.weights, device=device)
    loader = DataLoader(
        UnlabeledCropDataset(items, input_size=INPUT_SIZE),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    accepted = []
    rejected = 0
    conf_sum = 0.0
    model.eval()
    with torch.no_grad():
        for images, indices in tqdm(loader, desc="pseudo"):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
            for row, conf, pred in zip(indices.tolist(), confs.tolist(), preds.tolist()):
                conf_sum += float(conf)
                if conf < args.conf:
                    rejected += 1
                    continue
                item = items[row]
                accepted.append(
                    {
                        "student_dir": item["student_dir"],
                        "student_id": item["student_id"],
                        "image_name": item["image_name"],
                        "label": INDEX_TO_LABEL[int(pred)],
                        "confidence": float(conf),
                    }
                )

    counts = Counter(row["label"] for row in accepted)
    mean_conf = (sum(row["confidence"] for row in accepted) / len(accepted)) if accepted else 0.0
    print(
        "accepted={}  rejected_low_conf={}  open={}  closed={}  mean_conf={:.4f}".format(
            len(accepted),
            rejected,
            counts.get("open", 0),
            counts.get("closed", 0),
            mean_conf,
        )
    )

    written_students = 0
    if not args.dry_run:
        grouped = defaultdict(list)
        for row in accepted:
            grouped[row["student_dir"]].append(row)
        for student_dir, rows in grouped.items():
            records = load_student_label_records(student_dir)
            for row in rows:
                prev = records.get(row["image_name"])
                if prev is not None and prev.source == HUMAN_SOURCE:
                    continue
                records[row["image_name"]] = LabelRecord(
                    image_name=row["image_name"],
                    label=row["label"],
                    source=PSEUDO_SOURCE,
                    confidence=row["confidence"],
                )
            save_student_label_records(student_dir, records)
            written_students += 1
        print("updated CSV in {} student folders".format(written_students))
    else:
        print("dry-run: no CSV written")

    payload = {
        "input": os.path.abspath(args.input),
        "weights": args.weights,
        "conf": args.conf,
        "dry_run": bool(args.dry_run),
        "to_predict": len(items),
        "accepted": len(accepted),
        "rejected_low_conf": rejected,
        "keep_human": skipped_human,
        "keep_existing_pseudo": skipped_pseudo,
        "open": counts.get("open", 0),
        "closed": counts.get("closed", 0),
        "mean_confidence": round(mean_conf, 4),
        "mean_confidence_all_predictions": round(conf_sum / len(items), 4) if items else 0.0,
        "students_updated": written_students,
        "excluded_students": args.exclude_student,
    }
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print("report:", args.report)


if __name__ == "__main__":
    main()
