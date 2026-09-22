"""Run the three baseline experiments and write a comparison table."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.train_loop import dump_json, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all classroom baseline experiments")
    parser.add_argument("--skip-exp1", action="store_true")
    parser.add_argument("--skip-exp2", action="store_true")
    parser.add_argument("--skip-exp3", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    return parser.parse_args()


def run_script(rel_path: str, extra: list) -> None:
    cmd = [sys.executable, os.path.join(ROOT, rel_path)] + extra
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def pct(value, n=None) -> str:
    if value is None or n == 0:
        return "-"
    return "{:.2f}%".format(float(value) * 100.0)


def row(name: str, item: dict) -> str:
    return "| {} | {} | {} | {} | {} |".format(
        name,
        item.get("samples", "-"),
        pct(item.get("accuracy"), item.get("samples")),
        pct(item.get("clear_accuracy"), item.get("clear_samples")),
        pct(item.get("blur_accuracy"), item.get("blur_samples")),
    )


def write_report(compare: dict, path: str) -> None:
    exp1 = compare.get("exp1", {}).get("methods", {})
    exp2 = compare.get("exp2", {}).get("models", {})
    exp3 = compare.get("exp3", {}).get("results", {})
    ear = exp1.get("ear_threshold", {})
    lines = [
        "# 基线实验",
        "",
        "三组对比都固定同一课堂验证集：`models/sharpness_split.json` 中 4 名 val 学生、499 张人工图。",
        "自训练对照为 `models/method1/stage_blur30.pth`（不重训、不覆盖 `init70` 与方法一权重）。",
        "",
        "## 实验一：方法基线",
        "",
        "同一课堂 val 上对比 EAR+阈值、HOG+SVM、LBP+SVM 与自训练 MobileNetV2。",
        "EAR 优先 MediaPipe Face Mesh，失败再试 OpenCV Haar；检测失败记为错分。",
        "HOG/LBP 在 human train（划分内）上训练 LinearSVC。",
        "",
        "| 方法 | 样本 | 总体准确率 | 清晰 | 模糊 |",
        "| --- | ---: | ---: | ---: | ---: |",
        row("EAR + 阈值", exp1.get("ear_threshold", {})),
        row("HOG + SVM", exp1.get("hog_svm", {})),
        row("LBP + SVM", exp1.get("lbp_svm", {})),
        row("MobileNetV2 自训练", exp1.get("mobilenetv2_selftrain", {})),
        "",
        "- EAR 检测器：{}".format(ear.get("detector", "-")),
        "- EAR 检测成功率：{}（成功 {} / 失败 {}）".format(
            pct(ear.get("detect_rate")),
            ear.get("detected", "-"),
            ear.get("failed", "-"),
        ),
        "- 检测器计数：MediaPipe {}，Haar {}，失败 {}".format(
            (ear.get("detector_counts") or {}).get("mediapipe", "-"),
            (ear.get("detector_counts") or {}).get("haar", "-"),
            (ear.get("detector_counts") or {}).get("fail", "-"),
        ),
        "- EAR 阈值（仅在 train 上搜索）：{}".format(ear.get("threshold", "-")),
        "- 结论：112×112 课堂人头上传统 EAR 明显弱于自训练 CNN；即使关键点大多能出结果，开合判定仍接近随机。",
        "",
        "## 实验二：结构基线",
        "",
        "数据池与方法一对齐（human train + 已有高置信伪标签，排除 val 学生），输入 112×112，从零训练，无 ImageNet 预训练。",
        "MobileNetV2 仍用 `stage_blur30.pth`。",
        "",
        "| 网络 | 样本 | 总体准确率 | 清晰 | 模糊 |",
        "| --- | ---: | ---: | ---: | ---: |",
        row("LeNet", exp2.get("lenet", {})),
        row("AlexNet", exp2.get("alexnet", {})),
        row("MobileNetV2 自训练", exp2.get("mobilenetv2_selftrain", {})),
        "",
        "- 训练池：human+pseudo 共 {} 张，排除 val 学生。".format(compare.get("exp2", {}).get("train_samples", "-")),
        "- 结论：同数据下 LeNet / AlexNet 弱于 MobileNetV2，且模糊子集掉点更大。",
        "",
        "## 实验三：跨域（CEW Dataset B 人脸 ↔ 课堂）",
        "",
        "CEW `ClosedFace` / `OpenFace` 按图 8:2 自测；课堂 val 仍为冻结 499 张。",
        "课堂对照为 `stage_blur30.pth`。混合模型从零训练：全部 CEW 人脸 + 方法一课堂池，early stop 在课堂 val。",
        "",
        "| 模型 / 测试集 | 样本 | 总体准确率 | 清晰 | 模糊 |",
        "| --- | ---: | ---: | ---: | ---: |",
        row("MNv2 CEW → CEW 自测", exp3.get("mnv2_cew_selfval", {})),
        row("MNv2 CEW → 课堂 val", exp3.get("mnv2_cew_on_classroom", {})),
        row("MNv2 课堂 → 课堂 val", exp3.get("mnv2_classroom_on_classroom", {})),
        row("MNv2 课堂 → CEW 自测", exp3.get("mnv2_classroom_on_cew", {})),
        row("MNv2 CEW+课堂 → 课堂 val", exp3.get("mnv2_mix_on_classroom", {})),
        "",
        "- CEW Dataset B 人脸：ClosedFace + OpenFace 共 {} 张，按图 8:2 自测。".format(
            compare.get("exp3", {}).get("cew_total", "-")
        ),
        "- 混合训练：CEW {} + 课堂池 {} = {}。".format(
            compare.get("exp3", {}).get("cew_total", "-"),
            compare.get("exp3", {}).get("classroom_pool", "-"),
            compare.get("exp3", {}).get("mix_train", "-"),
        ),
        "- 结论：课堂→CEW 与 CEW→课堂都会掉点；混合训练未超过纯课堂模型，说明 CEW 近景人脸帮不上课堂低像素分布。",
        "",
        "## 入口",
        "",
        "- `python scripts/baselines/eval_method_baselines.py`",
        "- `python scripts/baselines/train_classic_cnn.py --arch lenet|alexnet`",
        "- `python scripts/baselines/train_cew_mobilenet.py`",
        "- `python scripts/baselines/run_all_baselines.py`",
        "",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    py = "scripts/baselines"
    if not args.skip_exp1:
        run_script(py + "/eval_method_baselines.py", [])
    if not args.skip_exp2:
        run_script(py + "/train_classic_cnn.py", ["--arch", "lenet", "--epochs", str(args.epochs)])
        run_script(py + "/train_classic_cnn.py", ["--arch", "alexnet", "--epochs", str(args.epochs)])
    if not args.skip_exp3:
        run_script(
            py + "/train_cew_mobilenet.py",
            ["--epochs", str(args.epochs), "--skip-cew-train"],
        )

    exp1_path = os.path.join(ROOT, "output", "baseline_exp1.json")
    exp2_path = os.path.join(ROOT, "output", "baseline_exp2.json")
    exp3_path = os.path.join(ROOT, "output", "baseline_exp3.json")
    compare = {
        "exp1": load_json(exp1_path) if os.path.isfile(exp1_path) else {},
        "exp2": load_json(exp2_path) if os.path.isfile(exp2_path) else {},
        "exp3": load_json(exp3_path) if os.path.isfile(exp3_path) else {},
    }
    out_json = os.path.join(ROOT, "output", "baseline_compare.json")
    out_md = os.path.join(ROOT, "docs", "基线实验.md")
    dump_json(out_json, compare)
    write_report(compare, out_md)
    print("wrote", out_json)
    print("wrote", out_md)


if __name__ == "__main__":
    main()
