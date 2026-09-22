"""Interactive clear/blur annotation: 1=clear, 2=blur, 0=skip, b=back, q/Esc=quit.

Only walks source=human eye labels that do not yet have a sharpness value.
"""

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
    HUMAN_SOURCE,
    VALID_SHARPNESS,
    discover_student_dirs,
    load_student_label_records,
    update_record_sharpness,
)

WINDOW_NAME = "sharpness_annotate"
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]

Item = Tuple[str, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate clear/blur on human eye labels")
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
    return image.resize(new_size, Image.NEAREST)


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
        draw.text(
            (padding, y),
            line,
            fill=(240, 240, 240) if index == 0 else (200, 200, 200),
            font=font if index == 0 else small,
        )
        y += line_h
    return canvas


def to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def collect_items(root: str) -> List[Item]:
    items = []
    for student_dir in discover_student_dirs(root):
        student_id = os.path.basename(student_dir.rstrip("\\/"))
        records = load_student_label_records(student_dir)
        for image_name, rec in sorted(records.items()):
            if rec.source != HUMAN_SOURCE:
                continue
            items.append((student_dir, student_id, image_name))
    return items


def load_book(items: List[Item]) -> Dict[str, Dict[str, str]]:
    book = {}
    for student_dir, _, _ in items:
        if student_dir in book:
            continue
        records = load_student_label_records(student_dir)
        book[student_dir] = {
            name: rec.sharpness
            for name, rec in records.items()
            if rec.sharpness in VALID_SHARPNESS
        }
    return book


def load_eye_labels(items: List[Item]) -> Dict[str, Dict[str, str]]:
    book = {}
    for student_dir, _, _ in items:
        if student_dir in book:
            continue
        records = load_student_label_records(student_dir)
        book[student_dir] = {name: rec.label for name, rec in records.items()}
    return book


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


def persist(student_dir: str, image_name: str, sharpness: str) -> None:
    update_record_sharpness(student_dir, image_name, sharpness)


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
        raise SystemExit("no human-labeled eye images under {}".format(root))

    label_book = load_book(items)
    eye_book = load_eye_labels(items)
    unlabeled = sum(
        1
        for student_dir, _, image_name in items
        if image_name not in label_book.get(student_dir, {})
    )
    index = first_unlabeled_index(items, label_book)
    if unlabeled == 0:
        print("全部已标清晰度，进入复查（可按 b 回看，1/2 改标，q 退出）")
        index = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    print("keys: 1=clear  2=blur  0=skip  b=back  q/Esc=save+quit")
    print("students={}, human images={}, unlabeled sharpness={}".format(
        len({item[0] for item in items}), len(items), unlabeled
    ))

    while True:
        student_dir, student_id, image_name = items[index]
        labels = label_book.setdefault(student_dir, {})
        current = labels.get(image_name, "unlabeled")
        eye_label = eye_book.get(student_dir, {}).get(image_name, "?")
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
                "eye={}  sharpness={}   clear={}  blur={}".format(
                    eye_label, current, counts.get("clear", 0), counts.get("blur", 0)
                ),
                "1 clear | 2 blur | 0 skip | b back | q/Esc save+quit",
            ],
        )
        cv2.imshow(WINDOW_NAME, to_bgr(banner))
        key = decode_key(cv2.waitKey(0) & 0xFF)

        if key in ("q", "esc"):
            print("saved and exit")
            break
        if key == "b":
            if index > 0:
                index -= 1
            continue
        if key == "1":
            labels[image_name] = "clear"
            persist(student_dir, image_name, "clear")
        elif key == "2":
            labels[image_name] = "blur"
            persist(student_dir, image_name, "blur")
        elif key == "0":
            if image_name in labels:
                del labels[image_name]
            persist(student_dir, image_name, "")
        else:
            continue

        nxt = next_unlabeled_index(items, label_book, index)
        if nxt is not None:
            index = nxt
            continue
        if index + 1 < len(items):
            index += 1
            continue
        print("标注完成: clear={} blur={}".format(
            count_labels(label_book).get("clear", 0),
            count_labels(label_book).get("blur", 0),
        ))
        break

    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    annotate(os.path.abspath(args.input), args.display_size)


if __name__ == "__main__":
    main()
