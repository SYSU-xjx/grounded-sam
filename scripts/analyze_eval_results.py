#!/usr/bin/env python3
import argparse
import csv
import json
import os
from collections import Counter, defaultdict


def parse_args():
    parser = argparse.ArgumentParser("Analyze attribute evaluation results")
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Run directory, e.g. runs/2026-03-04_attr_eval_run01",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Optional output directory (default: <run_dir>/analysis)",
    )
    parser.add_argument(
        "--topk_fail",
        type=int,
        default=20,
        help="Top-K tasks to export for each failure reason table",
    )
    return parser.parse_args()


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def print_table(title, rows, columns):
    print(f"\n== {title} ==")
    if not rows:
        print("(empty)")
        return
    widths = {}
    for c in columns:
        widths[c] = max(len(c), max(len(str(r.get(c, ""))) for r in rows))
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("-+-".join("-" * widths[c] for c in columns))
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns))


def main():
    args = parse_args()
    run_dir = args.run_dir
    output_dir = args.output_dir or os.path.join(run_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    task_jsonl = os.path.join(run_dir, "task_results.jsonl")
    image_summary_json = os.path.join(run_dir, "image_summary.json")
    overall_summary_json = os.path.join(run_dir, "overall_summary.json")

    if not os.path.exists(task_jsonl):
        raise FileNotFoundError(f"Missing: {task_jsonl}")
    if not os.path.exists(image_summary_json):
        raise FileNotFoundError(f"Missing: {image_summary_json}")
    if not os.path.exists(overall_summary_json):
        raise FileNotFoundError(f"Missing: {overall_summary_json}")

    tasks = []
    with open(task_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))

    images = load_json(image_summary_json)
    overall = load_json(overall_summary_json)

    # 1) By attribute
    by_attr = defaultdict(lambda: {"num_tasks": 0, "num_hit": 0, "iou_sum": 0.0})
    for t in tasks:
        k = t["attribute_type"]
        by_attr[k]["num_tasks"] += 1
        by_attr[k]["num_hit"] += int(t["hit"])
        by_attr[k]["iou_sum"] += float(t["best_iou"])
    rows_by_attr = []
    for k, v in sorted(by_attr.items()):
        rows_by_attr.append(
            {
                "attribute_type": k,
                "num_tasks": v["num_tasks"],
                "num_hit": v["num_hit"],
                "acc": round(safe_div(v["num_hit"], v["num_tasks"]), 4),
                "mean_iou": round(safe_div(v["iou_sum"], v["num_tasks"]), 4),
            }
        )

    # 2) By category
    by_cat = defaultdict(lambda: {"num_tasks": 0, "num_hit": 0, "iou_sum": 0.0})
    for t in tasks:
        k = t["category_en"]
        by_cat[k]["num_tasks"] += 1
        by_cat[k]["num_hit"] += int(t["hit"])
        by_cat[k]["iou_sum"] += float(t["best_iou"])
    rows_by_cat = []
    for k, v in sorted(by_cat.items(), key=lambda x: (-x[1]["num_tasks"], x[0])):
        rows_by_cat.append(
            {
                "category_en": k,
                "num_tasks": v["num_tasks"],
                "num_hit": v["num_hit"],
                "acc": round(safe_div(v["num_hit"], v["num_tasks"]), 4),
                "mean_iou": round(safe_div(v["iou_sum"], v["num_tasks"]), 4),
            }
        )

    # 3) Miss reason stats
    miss_reason_counter = Counter()
    for t in tasks:
        if not t["hit"]:
            miss_reason_counter[t.get("miss_reason", "unknown")] += 1
    miss_total = sum(miss_reason_counter.values())
    rows_miss = []
    for reason, cnt in miss_reason_counter.most_common():
        rows_miss.append(
            {
                "miss_reason": reason,
                "count": cnt,
                "ratio_in_miss": round(safe_div(cnt, miss_total), 4),
                "ratio_in_all_tasks": round(safe_div(cnt, len(tasks)), 4),
            }
        )

    # 4) Image accuracy distribution
    image_accs = [float(x.get("image_acc", 0.0)) for x in images]
    if image_accs:
        sorted_acc = sorted(image_accs)
        n = len(sorted_acc)
        p50 = sorted_acc[int(0.5 * (n - 1))]
        p25 = sorted_acc[int(0.25 * (n - 1))]
        p75 = sorted_acc[int(0.75 * (n - 1))]
        image_stats = {
            "num_images": n,
            "mean_image_acc": round(sum(sorted_acc) / n, 4),
            "p25_image_acc": round(p25, 4),
            "p50_image_acc": round(p50, 4),
            "p75_image_acc": round(p75, 4),
            "min_image_acc": round(sorted_acc[0], 4),
            "max_image_acc": round(sorted_acc[-1], 4),
        }
    else:
        image_stats = {
            "num_images": 0,
            "mean_image_acc": 0.0,
            "p25_image_acc": 0.0,
            "p50_image_acc": 0.0,
            "p75_image_acc": 0.0,
            "min_image_acc": 0.0,
            "max_image_acc": 0.0,
        }

    # 5) Top-K failed tasks by miss reason (highest IoU first)
    def build_fail_rows(filtered_tasks):
        rows = []
        for t in filtered_tasks[: args.topk_fail]:
            rows.append(
                {
                    "task_id": t["task_id"],
                    "image_id": t["image_id"],
                    "file_name": t["file_name"],
                    "attribute_type": t["attribute_type"],
                    "category_en": t["category_en"],
                    "prompt_zh": t["prompt_zh"],
                    "best_iou": round(float(t["best_iou"]), 4),
                    "best_pred_score": round(float(t["best_pred_score"] or 0.0), 4),
                    "miss_reason": t.get("miss_reason", ""),
                }
            )
        return rows

    low_iou_tasks = sorted(
        [t for t in tasks if (not t["hit"]) and t.get("miss_reason") == "low_iou"],
        key=lambda x: (-float(x["best_iou"]), -(float(x.get("best_pred_score") or 0.0))),
    )
    wrong_instance_tasks = sorted(
        [t for t in tasks if (not t["hit"]) and t.get("miss_reason") == "wrong_instance"],
        key=lambda x: (-float(x["best_iou"]), -(float(x.get("best_pred_score") or 0.0))),
    )
    rows_low_iou_topk = build_fail_rows(low_iou_tasks)
    rows_wrong_instance_topk = build_fail_rows(wrong_instance_tasks)

    # 6) Markdown report
    markdown_lines = []
    markdown_lines.append("# Attribute Eval Analysis")
    markdown_lines.append("")
    markdown_lines.append("## Overall")
    markdown_lines.append("")
    markdown_lines.append(f"- num_tasks: {overall.get('num_tasks', len(tasks))}")
    markdown_lines.append(f"- num_hit: {overall.get('num_hit', sum(int(t['hit']) for t in tasks))}")
    markdown_lines.append(
        f"- acc: {round(safe_div(overall.get('num_hit', 0), max(overall.get('num_tasks', 0), 1)), 4)}"
    )
    iou_mean_all = round(safe_div(sum(float(t['best_iou']) for t in tasks), len(tasks)), 4) if tasks else 0.0
    markdown_lines.append(f"- mean_iou_all: {iou_mean_all}")
    markdown_lines.append("")
    markdown_lines.append("## Image-level Stats")
    markdown_lines.append("")
    for k, v in image_stats.items():
        markdown_lines.append(f"- {k}: {v}")
    markdown_lines.append("")

    def add_md_table(title, rows):
        markdown_lines.append(f"## {title}")
        markdown_lines.append("")
        if not rows:
            markdown_lines.append("_empty_")
            markdown_lines.append("")
            return
        headers = list(rows[0].keys())
        markdown_lines.append("| " + " | ".join(headers) + " |")
        markdown_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for r in rows:
            markdown_lines.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
        markdown_lines.append("")

    add_md_table("By Attribute", rows_by_attr)
    add_md_table("By Category", rows_by_cat)
    add_md_table("Miss Reason Distribution", rows_miss)
    add_md_table(f"Top-{args.topk_fail} Low-IoU Tasks", rows_low_iou_topk)
    add_md_table(f"Top-{args.topk_fail} Wrong-Instance Tasks", rows_wrong_instance_topk)

    report_md = "\n".join(markdown_lines)
    with open(os.path.join(output_dir, "analysis_report.md"), "w", encoding="utf-8") as f:
        f.write(report_md)

    # Save machine-readable outputs
    write_json(
        os.path.join(output_dir, "analysis_summary.json"),
        {
            "overall_ref": overall,
            "image_stats": image_stats,
            "by_attribute": rows_by_attr,
            "by_category": rows_by_cat,
            "miss_reasons": rows_miss,
            "top_low_iou_tasks": rows_low_iou_topk,
            "top_wrong_instance_tasks": rows_wrong_instance_topk,
        },
    )
    write_csv(
        os.path.join(output_dir, "by_attribute.csv"),
        rows_by_attr,
        ["attribute_type", "num_tasks", "num_hit", "acc", "mean_iou"],
    )
    write_csv(
        os.path.join(output_dir, "by_category.csv"),
        rows_by_cat,
        ["category_en", "num_tasks", "num_hit", "acc", "mean_iou"],
    )
    write_csv(
        os.path.join(output_dir, "miss_reasons.csv"),
        rows_miss,
        ["miss_reason", "count", "ratio_in_miss", "ratio_in_all_tasks"],
    )
    write_csv(
        os.path.join(output_dir, "top_low_iou_tasks.csv"),
        rows_low_iou_topk,
        [
            "task_id",
            "image_id",
            "file_name",
            "attribute_type",
            "category_en",
            "prompt_zh",
            "best_iou",
            "best_pred_score",
            "miss_reason",
        ],
    )
    write_csv(
        os.path.join(output_dir, "top_wrong_instance_tasks.csv"),
        rows_wrong_instance_topk,
        [
            "task_id",
            "image_id",
            "file_name",
            "attribute_type",
            "category_en",
            "prompt_zh",
            "best_iou",
            "best_pred_score",
            "miss_reason",
        ],
    )

    # Console summary
    print(f"Analysis output: {output_dir}")
    print_table("By Attribute", rows_by_attr, ["attribute_type", "num_tasks", "num_hit", "acc", "mean_iou"])
    print_table("By Category (top 10)", rows_by_cat[:10], ["category_en", "num_tasks", "num_hit", "acc", "mean_iou"])
    print_table("Miss Reasons", rows_miss, ["miss_reason", "count", "ratio_in_miss", "ratio_in_all_tasks"])


if __name__ == "__main__":
    main()
