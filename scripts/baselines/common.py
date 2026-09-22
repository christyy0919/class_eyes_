"""Shared helpers for classroom baseline experiments."""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from core.eye_state import LABEL_MAP
from core.train_loop import apply_student_split, load_json
from data.eye_dataset import EyeSample, HUMAN_SOURCE, PSEUDO_SOURCE, collect_labeled_samples
from scripts.evaluate_dual_sharpness import summarize


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLASSROOM = r"e:\课堂图片"
SPLIT_PATH = os.path.join(ROOT, "models", "sharpness_split.json")
MNV2_CLASSROOM = os.path.join(ROOT, "models", "method1", "stage_blur30.pth")
CEW_B_FACE = os.path.join(r"c:\Users\panda\Desktop\CEW", "dataset_B_FacialImages")


def load_classroom_split(root: str = CLASSROOM, split_path: str = SPLIT_PATH):
    human = collect_labeled_samples(root, sources=(HUMAN_SOURCE,), require_sharpness=True)
    if not os.path.isfile(split_path):
        raise SystemExit("split not found: {}".format(split_path))
    split = apply_student_split(human, load_json(split_path))
    return split["train_samples"], split["val_samples"], split


def load_method1_train_pool(root: str = CLASSROOM, split_path: str = SPLIT_PATH) -> Tuple[List[EyeSample], List[EyeSample]]:
    _train_human, val_samples, split = load_classroom_split(root, split_path)
    val_students = set(split.get("val_students") or [])
    human = collect_labeled_samples(root, sources=(HUMAN_SOURCE,), require_sharpness=True)
    pseudo = collect_labeled_samples(root, sources=(PSEUDO_SOURCE,), require_sharpness=True)
    train = [s for s in human + pseudo if s.student_id not in val_students]
    return train, val_samples


def _imdecode(path: str, flags: int):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def load_bgr(path: str):
    image = _imdecode(path, cv2.IMREAD_COLOR)
    if image is None:
        from PIL import Image

        pil = Image.open(path).convert("RGB")
        image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    if image is None:
        raise OSError("cannot read {}".format(path))
    return image


def load_gray(path: str):
    image = _imdecode(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        bgr = load_bgr(path)
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return image


def labels_of(samples: Sequence[EyeSample]) -> List[int]:
    return [LABEL_MAP[s.label] for s in samples]


def pack_preds(name: str, samples: Sequence[EyeSample], preds: Sequence[int], extra: dict = None) -> dict:
    y_true = labels_of(samples)
    payload = summarize(y_true, list(preds), samples)
    payload["name"] = name
    if extra:
        payload.update(extra)
    return payload
