#!/usr/bin/env python3
import argparse
import json
import os

import matplotlib.pyplot as plt
from matplotlib import font_manager


def parse_args():
    parser = argparse.ArgumentParser("Plot analysis charts for attribute evaluation runs")
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Run directory, e.g. runs/2026-03-04_attr_eval_run01",
    )
    parser.add_argument(
        "--analysis_dir",
        type=str,
        default="",
        help="Analysis directory. Default: <run_dir>/analysis",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory for plots. Default: <analysis_dir>/plots",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        choices=["en", "zh"],
        help="Plot language for titles/axes. If zh font is unavailable, auto-fallback to en.",
    )
    parser.add_argument(
        "--topk_category",
        type=int,
        default=10,
        help="Top-K categories by task count for category accuracy bar chart.",
    )
    return parser.parse_args()


def setup_font(lang):
    zh_font_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ]
    english_fallback = "DejaVu Sans"

    if lang == "en":
        plt.rcParams["font.family"] = english_fallback
        plt.rcParams["axes.unicode_minus"] = False
        return "en"

    for p in zh_font_candidates:
        if os.path.exists(p):
            try:
                font_manager.fontManager.addfont(p)
                font_name = font_manager.FontProperties(fname=p).get_name()
                plt.rcParams["font.family"] = font_name
                plt.rcParams["axes.unicode_minus"] = False
                return "zh"
            except Exception:
                continue

    # Fallback to English when Chinese font is unavailable.
    plt.rcParams["font.family"] = english_fallback
    plt.rcParams["axes.unicode_minus"] = False
    print("[WARN] No Chinese font found. Falling back to English labels.")
    return "en"


def load_analysis_summary(analysis_dir):
    summary_path = os.path.join(analysis_dir, "analysis_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Missing {summary_path}. Please run scripts/analyze_eval_results.py first."
        )
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_by_attribute(data, output_path, lang):
    rows = data.get("by_attribute", [])
    if not rows:
        return
    labels = [r["attribute_type"] for r in rows]
    accs = [float(r["acc"]) for r in rows]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, accs)
    for b, v in zip(bars, accs):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    if lang == "zh":
        plt.title("各属性类型准确率")
        plt.xlabel("属性类型")
        plt.ylabel("准确率")
    else:
        plt.title("Accuracy by Attribute Type")
        plt.xlabel("Attribute Type")
        plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_by_category(data, output_path, topk, lang):
    rows = data.get("by_category", [])
    if not rows:
        return
    rows = sorted(rows, key=lambda x: (-int(x["num_tasks"]), x["category_en"]))[:topk]
    labels = [r["category_en"] for r in rows]
    accs = [float(r["acc"]) for r in rows]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, accs)
    for b, v in zip(bars, accs):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8, rotation=90)
    if lang == "zh":
        plt.title(f"类别准确率 Top-{topk}（按任务数）")
        plt.xlabel("类别")
        plt.ylabel("准确率")
    else:
        plt.title(f"Category Accuracy Top-{topk} (by task count)")
        plt.xlabel("Category")
        plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_miss_reason_pie(data, output_path, lang):
    rows = data.get("miss_reasons", [])
    if not rows:
        return
    labels = [r["miss_reason"] for r in rows]
    sizes = [int(r["count"]) for r in rows]

    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    if lang == "zh":
        plt.title("误检原因分布")
    else:
        plt.title("Miss Reason Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    analysis_dir = args.analysis_dir or os.path.join(args.run_dir, "analysis")
    output_dir = args.output_dir or os.path.join(analysis_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    data = load_analysis_summary(analysis_dir)
    actual_lang = setup_font(args.lang)

    plot_by_attribute(
        data,
        os.path.join(output_dir, "bar_accuracy_by_attribute.png"),
        actual_lang,
    )
    plot_by_category(
        data,
        os.path.join(output_dir, "bar_accuracy_by_category_topk.png"),
        args.topk_category,
        actual_lang,
    )
    plot_miss_reason_pie(
        data,
        os.path.join(output_dir, "pie_miss_reason_distribution.png"),
        actual_lang,
    )

    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
