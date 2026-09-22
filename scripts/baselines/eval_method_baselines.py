"""Experiment 1: EAR+threshold, HOG+SVM, LBP+SVM vs classroom MobileNetV2."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from sklearn.svm import LinearSVC
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.train_loop import dump_json
from scripts.baselines.common import (
    CLASSROOM,
    MNV2_CLASSROOM,
    load_bgr,
    load_classroom_split,
    load_gray,
    labels_of,
    pack_preds,
)
from scripts.evaluate_dual_sharpness import evaluate_single_model


# MediaPipe Face Mesh eye corners / lids (468 landmarks).
RIGHT = dict(outer=33, inner=133, up1=159, up2=158, low1=145, low2=153)
LEFT = dict(outer=263, inner=362, up1=386, up2=385, low1=374, low2=380)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Method baselines on frozen classroom val")
    parser.add_argument("--input", default=CLASSROOM)
    parser.add_argument("--weights", default=MNV2_CLASSROOM)
    parser.add_argument("--output", default=os.path.join(ROOT, "output", "baseline_exp1.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-mnv2", action="store_true")
    return parser.parse_args()


def _pt(landmarks, idx, width, height):
    lm = landmarks[idx]
    return np.array([lm.x * width, lm.y * height], dtype=np.float32)


def ear_six(landmarks, spec, width, height) -> float:
    p1 = _pt(landmarks, spec["outer"], width, height)
    p2 = _pt(landmarks, spec["up1"], width, height)
    p3 = _pt(landmarks, spec["up2"], width, height)
    p4 = _pt(landmarks, spec["inner"], width, height)
    p5 = _pt(landmarks, spec["low2"], width, height)
    p6 = _pt(landmarks, spec["low1"], width, height)
    vert = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horiz = np.linalg.norm(p1 - p4)
    return float(vert / (2.0 * horiz + 1e-6))


LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
LANDMARKER_PATH = os.path.join(ROOT, "models", "baselines", "face_landmarker.task")


def ensure_face_landmarker(path: str = LANDMARKER_PATH) -> str:
    if os.path.isfile(path) and os.path.getsize(path) > 1000:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("download FaceLandmarker ->", path)
    import urllib.request

    urllib.request.urlretrieve(LANDMARKER_URL, path)
    return path


class EarSolver:
    def __init__(self):
        self.mesh = None
        self.landmarker = None
        self.mp_image_cls = None
        self.mp_format = None
        self.detector = "haar"
        try:
            import mediapipe as mp

            if hasattr(mp, "solutions"):
                self.mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=False,
                    min_detection_confidence=0.3,
                )
                self.detector = "mediapipe+haar"
            else:
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision

                model_path = ensure_face_landmarker()
                options = vision.FaceLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=model_path),
                    num_faces=1,
                    min_face_detection_confidence=0.3,
                    min_face_presence_confidence=0.3,
                    running_mode=vision.RunningMode.IMAGE,
                )
                self.landmarker = vision.FaceLandmarker.create_from_options(options)
                self.mp_image_cls = mp.Image
                self.mp_format = mp.ImageFormat.SRGB
                self.detector = "mediapipe+haar"
        except Exception as exc:
            print("MediaPipe unavailable, Haar only:", exc)
        cascade_dir = getattr(cv2.data, "haarcascades", "")
        face_xml = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
        eye_xml = os.path.join(cascade_dir, "haarcascade_eye.xml")
        self.face_det = cv2.CascadeClassifier(face_xml) if os.path.isfile(face_xml) else None
        self.eye_det = cv2.CascadeClassifier(eye_xml) if os.path.isfile(eye_xml) else None

    def close(self):
        if self.mesh is not None:
            self.mesh.close()
            self.mesh = None
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def mediapipe_ear(self, bgr) -> Optional[float]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = bgr.shape[:2]
        lms = None
        if self.mesh is not None:
            result = self.mesh.process(rgb)
            if result.multi_face_landmarks:
                lms = result.multi_face_landmarks[0].landmark
        elif self.landmarker is not None:
            mp_image = self.mp_image_cls(image_format=self.mp_format, data=np.ascontiguousarray(rgb))
            result = self.landmarker.detect(mp_image)
            if result.face_landmarks:
                lms = result.face_landmarks[0]
        if lms is None:
            return None
        return 0.5 * (ear_six(lms, LEFT, w, h) + ear_six(lms, RIGHT, w, h))

    def haar_openness(self, bgr) -> Optional[float]:
        if self.eye_det is None:
            return None
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        roi = gray
        if self.face_det is not None:
            faces = self.face_det.detectMultiScale(gray, 1.05, 3, minSize=(24, 24))
            if len(faces):
                x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                roi = gray[y : y + fh, x : x + fw]
        eyes = self.eye_det.detectMultiScale(roi, 1.05, 3, minSize=(8, 8))
        if len(eyes) == 0:
            return None
        ratios = [float(eh) / float(max(ew, 1)) for _x, _y, ew, eh in eyes]
        return float(np.mean(ratios))

    def score(self, path: str) -> Tuple[Optional[float], str]:
        bgr = load_bgr(path)
        value = self.mediapipe_ear(bgr)
        if value is not None:
            return value, "mediapipe"
        value = self.haar_openness(bgr)
        if value is not None:
            return value, "haar"
        return None, "fail"


def search_threshold(scores: Sequence[Optional[float]], y_true: Sequence[int]) -> float:
    valid = [(s, t) for s, t in zip(scores, y_true) if s is not None]
    if not valid:
        return 0.2
    vals = sorted({round(s, 4) for s, _t in valid})
    best_t, best_acc = vals[0], -1.0
    for t in vals:
        correct = 0
        for s, t_true in valid:
            pred = 1 if s >= t else 0
            correct += int(pred == t_true)
        acc = correct / float(len(valid))
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return float(best_t)


def ear_predict(samples, threshold: float, solver: EarSolver):
    scores = []
    sources = []
    preds = []
    y_true = labels_of(samples)
    for sample, true in tqdm(zip(samples, y_true), total=len(samples), desc="ear"):
        score, src = solver.score(sample.abs_path)
        scores.append(score)
        sources.append(src)
        if score is None:
            preds.append(1 - true)
        else:
            preds.append(1 if score >= threshold else 0)
    detected = sum(1 for s in scores if s is not None)
    return preds, {
        "threshold": round(threshold, 4),
        "detect_rate": round(detected / float(len(samples)), 4) if samples else 0.0,
        "detected": detected,
        "failed": len(samples) - detected,
        "detector": solver.detector,
        "detector_counts": {
            "mediapipe": sources.count("mediapipe"),
            "haar": sources.count("haar"),
            "fail": sources.count("fail"),
        },
        "fail_counted_as_error": True,
    }


HOG = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)


def hog_feature(gray: np.ndarray) -> np.ndarray:
    img = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    return HOG.compute(img).ravel().astype(np.float32)


def lbp_histogram(gray: np.ndarray) -> np.ndarray:
    img = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    h, w = img.shape
    center = img[1:-1, 1:-1]
    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dy, dx) in enumerate(offsets):
        neigh = img[1 + dy : h - 1 + dy, 1 + dx : w - 1 + dx]
        lbp |= ((neigh >= center).astype(np.uint8) << bit)
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)
    return hist.astype(np.float32)


def extract_features(samples, kind: str) -> np.ndarray:
    feats = []
    fn = hog_feature if kind == "hog" else lbp_histogram
    for sample in tqdm(samples, desc=kind):
        feats.append(fn(load_gray(sample.abs_path)))
    return np.stack(feats, axis=0)


def train_svm(x_train, y_train) -> LinearSVC:
    clf = LinearSVC(class_weight="balanced", max_iter=8000, dual=False, C=1.0)
    clf.fit(x_train, y_train)
    return clf


def main() -> None:
    args = parse_args()
    train_samples, val_samples, split = load_classroom_split(args.input)
    print("train={} val={}".format(len(train_samples), len(val_samples)))
    y_train = labels_of(train_samples)
    methods = {}

    print("=== EAR + threshold ===")
    solver = EarSolver()
    train_scores = []
    for sample in tqdm(train_samples, desc="ear-train"):
        score, _src = solver.score(sample.abs_path)
        train_scores.append(score)
    threshold = search_threshold(train_scores, y_train)
    print("EAR threshold", threshold, "detector", solver.detector)
    ear_preds, ear_extra = ear_predict(val_samples, threshold, solver)
    solver.close()
    methods["ear_threshold"] = pack_preds("ear_threshold", val_samples, ear_preds, ear_extra)

    print("=== HOG + SVM ===")
    x_train = extract_features(train_samples, "hog")
    x_val = extract_features(val_samples, "hog")
    hog_clf = train_svm(x_train, y_train)
    hog_preds = hog_clf.predict(x_val).tolist()
    methods["hog_svm"] = pack_preds("hog_svm", val_samples, hog_preds)

    print("=== LBP + SVM ===")
    x_train = extract_features(train_samples, "lbp")
    x_val = extract_features(val_samples, "lbp")
    lbp_clf = train_svm(x_train, y_train)
    lbp_preds = lbp_clf.predict(x_val).tolist()
    methods["lbp_svm"] = pack_preds("lbp_svm", val_samples, lbp_preds)

    if not args.skip_mnv2:
        print("=== MobileNetV2 stage_blur30 ===")
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mnv2 = evaluate_single_model(args.weights, val_samples, device, args.batch_size, 0)
        mnv2["name"] = "mobilenetv2_selftrain"
        methods["mobilenetv2_selftrain"] = mnv2

    payload = {
        "experiment": "method_baselines",
        "split": os.path.abspath(os.path.join(ROOT, "models", "sharpness_split.json")),
        "val_students": split.get("val_students"),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "methods": methods,
    }
    dump_json(args.output, payload)
    print("wrote", args.output)
    for name, item in methods.items():
        print("  {} acc={} clear={} blur={}".format(
            name, item.get("accuracy"), item.get("clear_accuracy"), item.get("blur_accuracy")
        ))


if __name__ == "__main__":
    main()
