"""Interactive eye-state annotation: 1=open, 2=closed, 0=skip, b=back, q/Esc=quit."""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.eye_dataset import (  # noqa: E402
    discover_student_dirs,
    list_images,
    load_student_labels,
    save_student_labels,
)

WINDOW_NAME = "eye_state_annotate"
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate open/closed eye labels")
    parser.add_argument("--input", required=True, help="班级文件夹或单个学生文件夹")
    parser.add_argument("--display-size", type=int, default=480, help="显示最短边像素")
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_for_display(image: Image.Image, display_size: int) -> Image.Image:
    width, height = image.size
    short = max(min(width, height), 1)
    scale = max(display_size / float(short), 1.0)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    if new_size == image.size:
        return image
    resample = Image.NEAREST
    return image.resize(new_size, resample)


def draw_banner(image: Image.Image, lines: List[str]) -> Image.Image:
    font = load_font(22)
    small = load_font(18)
    padding = 10
    line_h = 28
    banner_h = padding * 2 + line_h * len(lines)
    canvas = Image.new("RGB", (image.width, image.height + banner_h), (24, 24, 24))
    canvas.paste(image, (0, banner_h))
    draw = ImageDraw.Draw(canvas)
    y = padding
    for index, line in enumerate(lines):
        draw.text((padding, y), line, fill=(240, 240, 240) if index == 0 else (200, 200, 200),
                  font=font if index == 0 else small)
        y += line_h
    return canvas


def to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def collect_items(root: str) -> List[Tuple[str, str, str]]:
    items = []
    for student_dir in discover_student_dirs(root):
        student_id = os.path.basename(student_dir.rstrip("\\/"))
        for image_name in list_images(student_dir):
            items.append((student_dir, student_id, image_name))
    return items


def count_labels(label_book: Dict[str, Dict[str, str]]) -> Counter:
    counter = Counter()
    for labels in label_book.values():
        counter.update(labels.values())
    return counter


def first_unlabeled_index(items, label_book) -> int:
    for index, (student_dir, _, image_name) in enumerate(items):
        if image_name not in label_book.get(student_dir, {}):
            return index
    return max(len(items) - 1, 0)


def next_unlabeled_index(items, label_book, start: int) -> Optional[int]:
    for index in range(start + 1, len(items)):
        student_dir, _, image_name = items[index]
        if image_name not in label_book.get(student_dir, {}):
            return index
    return None


def persist(student_dir: str, labels: Dict[str, str]) -> None:
    save_student_labels(student_dir, labels)


def decode_key(code: int) -> str:
    if code in (27,):
        return "esc"
    if code in (ord("q"), ord("Q")):
        return "q"
    if code in (ord("b"), ord("B")):
        return "b"
    if code in (ord("1"),):
        return "1"
    if code in (ord("2"),):
        return "2"
    if code in (ord("0"),):
        return "0"
    return ""


def annotate(root: str, display_size: int) -> None:
    items = collect_items(root)
    if not items:
        raise SystemExit("no images found under {}".format(root))

    label_book = {student_dir: load_student_labels(student_dir)
                  for student_dir, _, _ in items}
    unlabeled = sum(
        1 for student_dir, _, image_name in items
        if image_name not in label_book.get(student_dir, {})
    )
    index = first_unlabeled_index(items, label_book)
    if unlabeled == 0:
        print("全部已标注，进入复查（可按 b 回看，1/2 改标，q 退出）")
        index = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    print("keys: 1=open  2=closed  0=skip  b=back  q/Esc=save+quit")
    print("students={}, images={}, unlabeled={}".format(
        len({item[0] for item in items}), len(items), unlabeled
    ))

    while True:
        student_dir, student_id, image_name = items[index]
        labels = label_book.setdefault(student_dir, {})
        current = labels.get(image_name, "unlabeled")
        counts = count_labels(label_book)
        abs_path = os.path.join(student_dir, image_name)
        try:
            preview = resize_for_display(load_rgb(abs_path), display_size)
        except OSError as exc:
            print("cannot read {}: {}".format(abs_path, exc))
            nxt = next_unlabeled_index(items, label_book, index)
            if nxt is None:
                print("no more readable images")
                break
            index = nxt
            continue

        banner = draw_banner(
            preview,
            [
                "[{}/{}] {} / {}".format(index + 1, len(items), student_id, image_name),
                "current={}   open={}  closed={}".format(
                    current, counts.get("open", 0), counts.get("closed", 0)
                ),
                "1 open | 2 closed | 0 skip | b back | q/Esc save+quit",
            ],
        )
        cv2.imshow(WINDOW_NAME, to_bgr(banner))
        key = decode_key(cv2.waitKey(0) & 0xFF)

        if key in ("q", "esc"):
            persist(student_dir, labels)
            print("saved and exit")
            break
        if key == "b":
            persist(student_dir, labels)
            if index > 0:
                index -= 1
            continue
        if key == "1":
            labels[image_name] = "open"
            persist(student_dir, labels)
        elif key == "2":
            labels[image_name] = "closed"
            persist(student_dir, labels)
        elif key == "0":
            if image_name in labels:
                del labels[image_name]
            persist(student_dir, labels)
        else:
            continue

        nxt = next_unlabeled_index(items, label_book, index)
        if nxt is not None:
            index = nxt
            continue
        if index + 1 < len(items):
            index += 1
            continue
        persist(student_dir, labels)
        print("标注完成: open={} closed={}".format(
            count_labels(label_book).get("open", 0),
            count_labels(label_book).get("closed", 0),
        ))
        break

    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    annotate(os.path.abspath(args.input), args.display_size)


if __name__ == "__main__":
    main()
