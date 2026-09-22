"""Compare init70 and method1 curriculum stages on one frozen val split."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.train_loop import dump_json, load_json
from scripts.evaluate_dual_sharpness import evaluate_single_model, load_val_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare init70 and method1 stages on a frozen val split")
    parser.add_argument("--input", required=True, help="班级根目录，例如 e:\\课堂图片")
    parser.add_argument("--split", default=os.path.join(ROOT, "models", "sharpness_split.json"))
    parser.add_argument("--init70", default=os.path.join(ROOT, "models", "init70", "eye_state_init70.pth"))
    parser.add_argument("--current", default=os.path.join(ROOT, "models", "eye_state_best.pth"))
    parser.add_argument("--method1-dir", default=os.path.join(ROOT, "models", "method1"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 2)
    parser.add_argument("--output", default=os.path.join(ROOT, "output", "sharpness_method_compare.json"))
    parser.add_argument("--doc", default=os.path.join(ROOT, "docs", "清晰模糊对比实验.md"))
    return parser.parse_args()


def maybe_eval(name, path, samples, device, batch_size, num_workers, methods):
    if not os.path.isfile(path):
        methods[name] = {"missing": True, "weights": path}
        print("skip missing", name, path)
        return
    print("eval", name)
    result = evaluate_single_model(path, samples, device, batch_size, num_workers)
    result["missing"] = False
    methods[name] = result


def write_doc(path: str, payload: dict) -> None:
    methods = payload.get("methods") or {}
    lines = [
        "# 清晰 / 模糊课程式训练（方法一）",
        "",
        "生成时间：{}".format(payload.get("generated_at", "")),
        "",
        "只保留方法一：从 init70 在清晰集微调，再分批混入模糊图。清晰/模糊双模型加权已取消。",
        "",
        "## 设定",
        "",
        "- 验证集固定为 [`models/sharpness_split.json`](../models/sharpness_split.json) 中的 val 学生，准确率只算人工标签。",
        "- 微调起点是 [`models/init70/eye_state_init70.pth`](../models/init70/eye_state_init70.pth)，不覆盖该备份。",
        "- 入口：[`scripts/train_method1_active.py`](../scripts/train_method1_active.py)。当前最好通常是 +10% 模糊。",
        "",
        "## 结果",
        "",
        "| 方法 | 总体准确率 | 清晰子集 | 模糊子集 | 样本数 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    def row(name, item):
        if not item or item.get("missing"):
            lines.append("| {} | 缺失 | - | - | - |".format(name))
            return
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {} |".format(
                name,
                item.get("accuracy", 0),
                item.get("clear_accuracy", 0),
                item.get("blur_accuracy", 0),
                item.get("samples", 0),
            )
        )

    row("init70 基线", methods.get("init70"))
    row("方法一 仅清晰", methods.get("method1_stage_clear"))
    row("方法一 +10% 模糊", methods.get("method1_stage_blur10"))
    row("方法一 +20% 模糊", methods.get("method1_stage_blur20"))
    row("方法一 +30% 模糊", methods.get("method1_stage_blur30"))
    row("当前 eye_state_best", methods.get("current_best"))
    if payload.get("status") == "pending_sharpness_annotation":
        lines[3:3] = [
            "",
            "> 当前还没有人工清晰/模糊标注，结果表为空。先跑 `scripts/annotate_sharpness.py`，再跑方法一后重新执行本对比脚本。",
            "",
            "init70 已按第一轮 1366 张人工标签恢复，自身 val 准确率：{}（epoch {}）。".format(
                (payload.get("init70_restore") or {}).get("own_split_val_accuracy", "未知"),
                (payload.get("init70_restore") or {}).get("best_epoch", "?"),
            ),
        ]
    lines.extend(
        [
            "",
            "## 方法说明",
            "",
            "从 init70 只在清晰训练集微调，再按固定 seed 把模糊训练集的 10% / 20% / 30% 嵌套混入继续训。高置信伪标签可并入训练集，但不进入验证集。",
            "",
            "详细数字见 [`output/sharpness_method_compare.json`](../output/sharpness_method_compare.json) 与 [`output/method1_results.json`](../output/method1_results.json)。",
            "",
            "## 复现",
            "",
            "```powershell",
            'python scripts\\annotate_sharpness.py --input "e:\\课堂图片"',
            'python scripts\\train_method1_active.py --input "e:\\课堂图片"',
            'python scripts\\compare_sharpness_methods.py --input "e:\\课堂图片"',
            "```",
            "",
            "标注快捷键：`1` 清晰，`2` 模糊，`0` 跳过，`b` 上一张，`q` / `Esc` 保存退出。",
            "",
        ]
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        samples = load_val_samples(args.input, args.split)
    except SystemExit as exc:
        init70_meta = {}
        meta_path = os.path.splitext(args.init70)[0] + "_meta.json"
        if os.path.isfile(meta_path):
            init70_meta = load_json(meta_path)
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "status": "pending_sharpness_annotation",
            "detail": str(exc),
            "split": args.split,
            "init70_restore": {
                "weights": args.init70,
                "exists": os.path.isfile(args.init70),
                "own_split_val_accuracy": init70_meta.get("val_accuracy"),
                "samples": init70_meta.get("samples"),
                "best_epoch": init70_meta.get("best_epoch"),
                "human_only": init70_meta.get("human_only"),
            },
            "methods": {},
        }
        dump_json(args.output, payload)
        write_doc(args.doc, payload)
        print("wrote", args.output)
        print("wrote", args.doc)
        print(exc)
        return
    print("val samples:", len(samples))
    methods = {}
    maybe_eval("init70", args.init70, samples, device, args.batch_size, args.num_workers, methods)
    maybe_eval(
        "method1_stage_clear",
        os.path.join(args.method1_dir, "stage_clear.pth"),
        samples,
        device,
        args.batch_size,
        args.num_workers,
        methods,
    )
    maybe_eval(
        "method1_stage_blur10",
        os.path.join(args.method1_dir, "stage_blur10.pth"),
        samples,
        device,
        args.batch_size,
        args.num_workers,
        methods,
    )
    maybe_eval(
        "method1_stage_blur20",
        os.path.join(args.method1_dir, "stage_blur20.pth"),
        samples,
        device,
        args.batch_size,
        args.num_workers,
        methods,
    )
    maybe_eval(
        "method1_stage_blur30",
        os.path.join(args.method1_dir, "stage_blur30.pth"),
        samples,
        device,
        args.batch_size,
        args.num_workers,
        methods,
    )

    if os.path.isfile(args.current):
        maybe_eval("current_best", args.current, samples, device, args.batch_size, args.num_workers, methods)
    else:
        methods["current_best"] = {"missing": True, "weights": args.current}

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "split": args.split,
        "val_samples": len(samples),
        "students": sorted({s.student_id for s in samples}),
        "methods": methods,
    }
    dump_json(args.output, payload)
    write_doc(args.doc, payload)
    print("wrote", args.output)
    print("wrote", args.doc)


if __name__ == "__main__":
    main()
