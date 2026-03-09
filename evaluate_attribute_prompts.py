#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import re
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict


def parse_args():
    parser = argparse.ArgumentParser("Attribute-level batch evaluator for Grounded-SAM/GroundingDINO")
    parser.add_argument("--eval_json", type=str, required=True, help="attribute_eval_v1.json path")
    parser.add_argument("--images_dir", type=str, required=True, help="directory containing COCO images")
    parser.add_argument("--config", type=str, required=True, help="GroundingDINO config path")
    parser.add_argument("--grounded_checkpoint", type=str, required=True, help="GroundingDINO checkpoint path")
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for hit")
    parser.add_argument("--bert_base_uncased_path", type=str, default="", help="optional local bert path")
    parser.add_argument("--enable_translate", action="store_true", help="translate zh prompt to en before inference")
    parser.add_argument(
        "--translation_model",
        type=str,
        default="Helsinki-NLP/opus-mt-zh-en",
        help="HF model name for zh->en translation",
    )
    parser.add_argument("--max_vis_per_attr", type=int, default=20, help="top-K hit/fail visualizations per attr")
    parser.add_argument("--run_root", type=str, default="runs", help="output root folder")
    return parser.parse_args()


class PromptTranslator:
    def __init__(self, model_name, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.model.eval()

    @staticmethod
    def contains_chinese(text):
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    def translate(self, text, max_new_tokens=128):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4, do_sample=False)
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()


def normalize_prompt(original_prompt, translated_prompt):
    prompt = translated_prompt.strip().lower()
    prompt = re.sub(r"\s+", " ", prompt)
    prompt = prompt.rstrip(" .!?")
    prompt = re.sub(r"^(a|an|the)\s+", "", prompt)
    if any(tok in original_prompt for tok in ["所有", "全部", "全部的"]) and not prompt.endswith("s"):
        prompt = f"{prompt}s"
    return prompt


def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    return model.eval().to(device)


def preprocess_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_tensor, _ = transform(image_pil, None)
    return image_pil, image_tensor


def run_grounding(model, image_tensor, caption, box_threshold, device):
    caption = caption.strip().lower()
    if not caption.endswith("."):
        caption = caption + "."
    with torch.no_grad():
        outputs = model(image_tensor[None].to(device), captions=[caption])
    pred_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # [NQ, 256]
    pred_boxes = outputs["pred_boxes"].cpu()[0]  # [NQ, 4] cxcywh normalized
    max_scores = pred_logits.max(dim=1)[0]
    keep = max_scores > box_threshold
    boxes = pred_boxes[keep]
    scores = max_scores[keep]
    return boxes.numpy(), scores.numpy(), caption


def cxcywh_to_xyxy_abs(boxes_cxcywh, width, height):
    if boxes_cxcywh.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    out = np.zeros_like(boxes_cxcywh, dtype=np.float32)
    out[:, 0] = (boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2.0) * width
    out[:, 1] = (boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2.0) * height
    out[:, 2] = (boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2.0) * width
    out[:, 3] = (boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2.0) * height
    return out


def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def draw_vis(image_path, gt_box, pred_box, title_lines, out_path):
    def _get_chinese_font(size=16):
        font_candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        ]
        for path in font_candidates:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size=size)
                except Exception:
                    continue
        return ImageFont.load_default()

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _get_chinese_font(size=16)
    # GT: green, Pred: red
    draw.rectangle(gt_box, outline=(0, 255, 0), width=3)
    if pred_box is not None:
        draw.rectangle(pred_box, outline=(255, 0, 0), width=3)
    text = " | ".join(title_lines)
    text_bbox = draw.textbbox((8, 8), text, font=font)
    box_right = min(text_bbox[2] + 6, image.size[0] - 5)
    box_bottom = min(text_bbox[3] + 4, image.size[1] - 5)
    draw.rectangle((5, 5, box_right, box_bottom), fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255), font=font)
    image.save(out_path)


def ensure_run_dir(run_root):
    today = dt.datetime.now().strftime("%Y-%m-%d")
    base = os.path.join(run_root, f"{today}_attr_eval_run")
    idx = 1
    while True:
        run_dir = f"{base}{idx:02d}"
        if not os.path.exists(run_dir):
            os.makedirs(run_dir, exist_ok=True)
            os.makedirs(os.path.join(run_dir, "vis_fail"), exist_ok=True)
            os.makedirs(os.path.join(run_dir, "vis_hit"), exist_ok=True)
            return run_dir
        idx += 1


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def main():
    args = parse_args()
    run_dir = ensure_run_dir(args.run_root)

    eval_data = json.load(open(args.eval_json, "r", encoding="utf-8"))
    images = eval_data["images"]

    translator = None
    if args.enable_translate:
        translator = PromptTranslator(args.translation_model, args.device)

    model = load_model(args.config, args.grounded_checkpoint, args.bert_base_uncased_path, args.device)

    run_meta = {
        "created_at": dt.datetime.now().isoformat(),
        "args": vars(args),
        "num_images": len(images),
        "eval_json": args.eval_json,
    }
    json.dump(run_meta, open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    task_jsonl_path = os.path.join(run_dir, "task_results.jsonl")
    image_summary_path = os.path.join(run_dir, "image_summary.json")
    overall_summary_path = os.path.join(run_dir, "overall_summary.json")

    overall = {
        "num_images": 0,
        "num_tasks": 0,
        "num_hit": 0,
        "acc": 0.0,
        "by_attribute": defaultdict(lambda: {"num_tasks": 0, "num_hit": 0, "acc": 0.0}),
        "by_category": defaultdict(lambda: {"num_tasks": 0, "num_hit": 0, "acc": 0.0}),
    }
    image_summaries = []
    vis_pool_hit = defaultdict(list)
    vis_pool_fail = defaultdict(list)

    with open(task_jsonl_path, "w", encoding="utf-8") as f_jsonl:
        total_tasks = sum(len(x.get("tasks", [])) for x in images)
        pbar = tqdm(total=total_tasks, desc="AttributeEval", unit="task")

        for img in images:
            image_id = img["image_id"]
            file_name = img["file_name"]
            image_path = os.path.join(args.images_dir, file_name)
            if not os.path.exists(image_path):
                # Skip image but write empty summary for traceability.
                image_summaries.append(
                    {
                        "image_id": image_id,
                        "file_name": file_name,
                        "num_tasks": len(img.get("tasks", [])),
                        "num_hit": 0,
                        "image_acc": 0.0,
                        "hits": [],
                        "misses": [t["task_id"] for t in img.get("tasks", [])],
                        "error": "image_not_found",
                    }
                )
                pbar.update(len(img.get("tasks", [])))
                continue

            image_pil, image_tensor = preprocess_image(image_path)
            width, height = image_pil.size

            # group same prompt to avoid redundant inference
            tasks_by_prompt = defaultdict(list)
            for t in img.get("tasks", []):
                tasks_by_prompt[t["prompt_zh"]].append(t)

            hit_ids = []
            miss_ids = []
            task_count = 0
            hit_count = 0

            # for wrong_instance diagnosis: collect same-category GT boxes in this image
            gt_by_cat = defaultdict(list)
            for t in img.get("tasks", []):
                gt_by_cat[t["category_en"]].append((t["target_ann_id"], t["target_bbox_xyxy"]))

            for prompt_zh, prompt_tasks in tasks_by_prompt.items():
                t0 = time.time()
                infer_prompt = prompt_zh
                translated_prompt = ""
                if translator is not None and translator.contains_chinese(prompt_zh):
                    translated_prompt = translator.translate(prompt_zh)
                    infer_prompt = normalize_prompt(prompt_zh, translated_prompt)
                boxes_cxcywh, scores, final_prompt = run_grounding(
                    model, image_tensor, infer_prompt, args.box_threshold, args.device
                )
                boxes_xyxy = cxcywh_to_xyxy_abs(boxes_cxcywh, width, height)

                for task in prompt_tasks:
                    task_count += 1
                    target = np.array(task["target_bbox_xyxy"], dtype=np.float32)
                    if boxes_xyxy.shape[0] == 0:
                        best_iou = 0.0
                        best_idx = -1
                    else:
                        ious = np.array([iou_xyxy(target, b) for b in boxes_xyxy], dtype=np.float32)
                        best_idx = int(np.argmax(ious))
                        best_iou = float(ious[best_idx])

                    hit = best_iou >= args.iou_threshold
                    best_pred_bbox = boxes_xyxy[best_idx].tolist() if best_idx >= 0 else None
                    best_pred_score = float(scores[best_idx]) if best_idx >= 0 else None

                    if hit:
                        miss_reason = ""
                        hit_count += 1
                        hit_ids.append(task["task_id"])
                    else:
                        if best_idx < 0:
                            miss_reason = "no_box"
                        else:
                            # wrong_instance: pred overlaps another same-category GT, but not target
                            wrong_instance = False
                            best_box = boxes_xyxy[best_idx]
                            alt_ious = []
                            for ann_id, gt_box in gt_by_cat[task["category_en"]]:
                                if ann_id == task["target_ann_id"]:
                                    continue
                                alt_ious.append(iou_xyxy(best_box, np.array(gt_box, dtype=np.float32)))
                            alt_best = max(alt_ious) if alt_ious else 0.0
                            wrong_instance = best_iou < 0.2 and alt_best >= 0.5
                            miss_reason = "wrong_instance" if wrong_instance else "low_iou"
                        miss_ids.append(task["task_id"])

                    elapsed_ms = (time.time() - t0) * 1000.0
                    row = {
                        "task_id": task["task_id"],
                        "image_id": task["image_id"],
                        "file_name": task["file_name"],
                        "category_en": task["category_en"],
                        "category_zh": task.get("category_zh", ""),
                        "attribute_type": task["attribute_type"],
                        "attribute_value": task["attribute_value"],
                        "prompt_zh": prompt_zh,
                        "prompt_infer": final_prompt,
                        "translated_prompt_raw": translated_prompt,
                        "target_ann_id": task["target_ann_id"],
                        "target_bbox_xyxy": task["target_bbox_xyxy"],
                        "best_iou": best_iou,
                        "hit": hit,
                        "best_pred_bbox": best_pred_bbox,
                        "best_pred_score": best_pred_score,
                        "miss_reason": miss_reason,
                        "elapsed_ms": elapsed_ms,
                    }
                    f_jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")

                    overall["num_tasks"] += 1
                    overall["num_hit"] += int(hit)
                    ba = overall["by_attribute"][task["attribute_type"]]
                    ba["num_tasks"] += 1
                    ba["num_hit"] += int(hit)
                    bc = overall["by_category"][task["category_en"]]
                    bc["num_tasks"] += 1
                    bc["num_hit"] += int(hit)

                    vis_item = {
                        "image_path": image_path,
                        "task_id": task["task_id"],
                        "attribute_type": task["attribute_type"],
                        "prompt": prompt_zh,
                        "prompt_en": translated_prompt if translated_prompt else prompt_zh,
                        "target_bbox_xyxy": task["target_bbox_xyxy"],
                        "best_pred_bbox": best_pred_bbox,
                        "best_iou": best_iou,
                        "best_pred_score": best_pred_score if best_pred_score is not None else 0.0,
                    }
                    if hit:
                        vis_pool_hit[task["attribute_type"]].append(vis_item)
                    else:
                        vis_pool_fail[task["attribute_type"]].append(vis_item)

                    pbar.set_postfix(
                        image_id=image_id,
                        file=file_name,
                        iou=f"{best_iou:.3f}",
                        status="hit" if hit else "miss",
                        ms=f"{elapsed_ms:.1f}",
                    )
                    tqdm.write(
                        f"[task] image_id={image_id} file={file_name} "
                        f"prompt='{prompt_zh}' best_iou={best_iou:.3f} "
                        f"{'hit' if hit else 'miss'} time_ms={elapsed_ms:.1f}"
                    )
                    pbar.update(1)

            image_summary = {
                "image_id": image_id,
                "file_name": file_name,
                "num_tasks": task_count,
                "num_hit": hit_count,
                "image_acc": safe_div(hit_count, task_count),
                "hits": hit_ids,
                "misses": miss_ids,
            }
            image_summaries.append(image_summary)
            overall["num_images"] += 1

        pbar.close()

    overall["acc"] = safe_div(overall["num_hit"], overall["num_tasks"])
    overall["by_attribute"] = dict(overall["by_attribute"])
    overall["by_category"] = dict(overall["by_category"])
    for _, v in overall["by_attribute"].items():
        v["acc"] = safe_div(v["num_hit"], v["num_tasks"])
    for _, v in overall["by_category"].items():
        v["acc"] = safe_div(v["num_hit"], v["num_tasks"])

    json.dump(image_summaries, open(image_summary_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(overall, open(overall_summary_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    for attr_type, items in vis_pool_fail.items():
        items_sorted = sorted(items, key=lambda x: x["best_iou"])[: args.max_vis_per_attr]
        for i, item in enumerate(items_sorted):
            out_path = os.path.join(run_dir, "vis_fail", f"{attr_type}_{i:02d}_{item['task_id']}.jpg")
            draw_vis(
                item["image_path"],
                item["target_bbox_xyxy"],
                item["best_pred_bbox"],
                [
                    f"attr={attr_type}",
                    f"prompt={item['prompt_en']}",
                    f"iou={item['best_iou']:.3f}",
                    f"score={item['best_pred_score']:.3f}",
                ],
                out_path,
            )

    for attr_type, items in vis_pool_hit.items():
        items_sorted = sorted(items, key=lambda x: x["best_iou"], reverse=True)[: args.max_vis_per_attr]
        for i, item in enumerate(items_sorted):
            out_path = os.path.join(run_dir, "vis_hit", f"{attr_type}_{i:02d}_{item['task_id']}.jpg")
            draw_vis(
                item["image_path"],
                item["target_bbox_xyxy"],
                item["best_pred_bbox"],
                [
                    f"attr={attr_type}",
                    f"prompt={item['prompt_en']}",
                    f"iou={item['best_iou']:.3f}",
                    f"score={item['best_pred_score']:.3f}",
                ],
                out_path,
            )

    print(f"Run dir: {run_dir}")
    print(f"task_results: {task_jsonl_path}")
    print(f"image_summary: {image_summary_path}")
    print(f"overall_summary: {overall_summary_path}")


if __name__ == "__main__":
    main()
