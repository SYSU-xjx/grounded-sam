import argparse
import json
import os
import re
import sys
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib import font_manager
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import SamPredictor, sam_hq_model_registry, sam_model_registry


def configure_matplotlib_font():
    candidate_fonts = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "SimHei",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidate_fonts:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


configure_matplotlib_font()


ATTRIBUTE_PATTERNS = [
    ("leftmost", "最左侧"),
    ("rightmost", "最右侧"),
    ("topmost", "最上方"),
    ("bottommost", "最下方"),
    ("largest", "最大"),
    ("smallest", "最小"),
]


class PromptTranslator:
    def __init__(self, model_name, device):
        self.device = device
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    @staticmethod
    def contains_chinese(text):
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    def translate(self, text, max_new_tokens=128):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                do_sample=False,
            )
        return self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()


def normalize_english_prompt(original_prompt, translated_prompt):
    prompt = translated_prompt.strip()
    prompt = re.sub(r"\s+", " ", prompt)
    prompt = prompt.rstrip(" .!?")
    prompt = prompt.lower()
    prompt = re.sub(r"^(a|an|the)\s+", "", prompt)
    if any(tok in original_prompt for tok in ["所有", "全部", "全部的"]) and not prompt.endswith("s"):
        prompt = f"{prompt}s"
    return prompt


def parse_attribute_prompt(prompt_zh):
    text = prompt_zh.strip()
    text = re.split(r"[。.!?;；\n]+", text)[0].strip()
    for attribute_value, attribute_zh in ATTRIBUTE_PATTERNS:
        if text.startswith(attribute_zh):
            category_zh = text[len(attribute_zh):].lstrip("的").strip()
            if not category_zh:
                raise ValueError(f"Missing category in prompt: {prompt_zh}")
            attribute_type = "size" if attribute_value in {"largest", "smallest"} else "spatial"
            return {
                "prompt_zh": prompt_zh,
                "normalized_prompt_zh": text,
                "attribute_type": attribute_type,
                "attribute_value": attribute_value,
                "attribute_zh": attribute_zh,
                "category_zh": category_zh,
            }
    raise ValueError(
        "Unsupported attribute prompt. Supported prefixes: 最大, 最小, 最左侧, 最右侧, 最上方, 最下方"
    )


def load_image(image_path):
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


def load_grounding_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    return model.eval().to(device)


def load_sam_predictor(sam_version, sam_checkpoint, sam_hq_checkpoint, use_sam_hq, device):
    if use_sam_hq:
        if not sam_hq_checkpoint:
            raise ValueError("--sam_hq_checkpoint is required when --use_sam_hq is set")
        sam = sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device)
    else:
        if not sam_checkpoint:
            raise ValueError("--sam_checkpoint is required")
        sam = sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device)
    return SamPredictor(sam)


def grounding_predict(model, image_tensor, caption, box_threshold, text_threshold, device):
    caption = caption.strip().lower()
    if not caption.endswith("."):
        caption = caption + "."

    with torch.no_grad():
        outputs = model(image_tensor[None].to(device), captions=[caption])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]
    prediction_boxes = outputs["pred_boxes"].cpu()[0]
    max_scores = prediction_logits.max(dim=1)[0]
    keep = max_scores > box_threshold

    boxes = prediction_boxes[keep]
    scores = max_scores[keep]
    logits = prediction_logits[keep]
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    phrases = [
        get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).strip()
        for logit in logits
    ]
    return boxes, scores, phrases, caption


def boxes_cxcywh_to_xyxy_abs(boxes, width, height):
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.clone().cpu().numpy()
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    out = np.zeros_like(boxes, dtype=np.float32)
    out[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2.0) * width
    out[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2.0) * height
    out[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2.0) * width
    out[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2.0) * height
    return out


def iou_xyxy(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def truncate_candidates(candidates, topk_candidates):
    if topk_candidates <= 0 or len(candidates) <= topk_candidates:
        return candidates
    return sorted(candidates, key=lambda item: item["det_score"], reverse=True)[:topk_candidates]


def deduplicate_candidates(candidates, merge_iou_thresh):
    if len(candidates) <= 1:
        return candidates
    deduped = []
    for candidate in sorted(candidates, key=lambda item: item["det_score"], reverse=True):
        matched = False
        for kept in deduped:
            if iou_xyxy(candidate["box_xyxy"], kept["box_xyxy"]) >= merge_iou_thresh:
                matched = True
                break
        if not matched:
            deduped.append(candidate)
    return deduped


def merge_candidate_sets(subject_candidates, attr_candidates, merge_iou_thresh):
    merged = []
    for candidate in subject_candidates + attr_candidates:
        merged.append(candidate)
    return deduplicate_candidates(merged, merge_iou_thresh)


def _normalize_values(values, reverse=False):
    if len(values) == 0:
        return []
    values = np.asarray(values, dtype=np.float32)
    if reverse:
        values = -values
    min_v = float(values.min())
    max_v = float(values.max())
    if max_v - min_v < 1e-6:
        return [1.0] * len(values)
    return ((values - min_v) / (max_v - min_v)).tolist()


def compute_attribute_scores(candidates, attribute_value):
    if len(candidates) == 0:
        return candidates

    if attribute_value in {"leftmost", "rightmost"}:
        values = [0.5 * (item["box_xyxy"][0] + item["box_xyxy"][2]) for item in candidates]
        reverse = attribute_value == "leftmost"
    elif attribute_value in {"topmost", "bottommost"}:
        values = [0.5 * (item["box_xyxy"][1] + item["box_xyxy"][3]) for item in candidates]
        reverse = attribute_value == "topmost"
    elif attribute_value in {"largest", "smallest"}:
        values = [
            max(0.0, item["box_xyxy"][2] - item["box_xyxy"][0]) *
            max(0.0, item["box_xyxy"][3] - item["box_xyxy"][1])
            for item in candidates
        ]
        reverse = attribute_value == "smallest"
    else:
        raise ValueError(f"Unsupported attribute value: {attribute_value}")

    attr_scores = _normalize_values(values, reverse=reverse)
    for item, attr_score in zip(candidates, attr_scores):
        item["attr_score"] = float(attr_score)
    return candidates


def rerank_candidates(candidates, alpha, beta, apply_attribute_rerank=True):
    ranked = []
    for candidate in candidates:
        attr_score = candidate.get("attr_score", 0.0) if apply_attribute_rerank else 0.0
        candidate["final_score"] = float(alpha * candidate["det_score"] + beta * attr_score)
        ranked.append(candidate)
    return sorted(ranked, key=lambda item: item["final_score"], reverse=True)


def candidate_from_prediction(box_xyxy, det_score, phrase, source):
    return {
        "box_xyxy": [float(v) for v in box_xyxy],
        "det_score": float(det_score),
        "phrase": phrase,
        "source": source,
        "attr_score": 0.0,
        "final_score": 0.0,
    }


def build_candidates_from_grounding(model, image_tensor, image_size, prompt_en, box_threshold, text_threshold, device, source):
    boxes, scores, phrases, used_prompt = grounding_predict(
        model, image_tensor, prompt_en, box_threshold, text_threshold, device
    )
    width, height = image_size
    boxes_xyxy = boxes_cxcywh_to_xyxy_abs(boxes, width, height)
    candidates = [
        candidate_from_prediction(box_xyxy, score, phrase, source)
        for box_xyxy, score, phrase in zip(boxes_xyxy, scores.tolist(), phrases)
    ]
    return candidates, used_prompt


def infer_prompt_texts(prompt_info, translator, enable_translate):
    translated_prompt_raw = ""
    if enable_translate and translator is not None and translator.contains_chinese(prompt_info["normalized_prompt_zh"]):
        translated_prompt_raw = translator.translate(prompt_info["normalized_prompt_zh"])
        attr_prompt_en = normalize_english_prompt(prompt_info["normalized_prompt_zh"], translated_prompt_raw)
        category_translated_raw = translator.translate(prompt_info["category_zh"])
        subject_prompt_en = normalize_english_prompt(prompt_info["category_zh"], category_translated_raw)
    else:
        attr_prompt_en = prompt_info["normalized_prompt_zh"]
        subject_prompt_en = prompt_info["category_zh"]
    return {
        "translated_prompt_raw": translated_prompt_raw,
        "subject_prompt_en": subject_prompt_en,
        "attr_prompt_en": attr_prompt_en,
    }


def predict_mask_from_box(predictor, image_rgb, box_xyxy, device):
    if box_xyxy is None:
        return None
    predictor.set_image(image_rgb)
    box_tensor = torch.tensor([box_xyxy], dtype=torch.float32)
    transformed_boxes = predictor.transform.apply_boxes_torch(box_tensor, image_rgb.shape[:2]).to(device)
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    if masks is None or masks.numel() == 0:
        return None
    return masks[0, 0].detach().cpu().numpy()


def overlay_and_save(image_rgb, selected_box, mask, output_path, label):
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    if mask is not None:
        color = np.array([30 / 255.0, 144 / 255.0, 255 / 255.0, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        plt.gca().imshow(mask_image)
    if selected_box is not None:
        x0, y0, x1, y1 = selected_box
        plt.gca().add_patch(
            plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
        )
        plt.gca().text(x0, y0, label)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
    plt.close()


def save_mask_image(mask, output_path):
    if mask is None:
        return
    mask_img = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_img).save(output_path)


def save_all_candidates_image(image_rgb, candidates, output_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    ax = plt.gca()
    for idx, candidate in enumerate(candidates):
        x0, y0, x1, y1 = candidate["box_xyxy"]
        ax.add_patch(
            plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="yellow", facecolor=(0, 0, 0, 0), lw=1.5)
        )
        ax.text(x0, y0, f"{idx}:{candidate['final_score']:.2f}")
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", dpi=250, pad_inches=0.0)
    plt.close()


def export_result(
    output_dir,
    experiment_name,
    image_path,
    prompt_info,
    prompt_texts,
    selected_candidate,
    candidates,
    ablation_flags,
    timing_ms,
    extra=None,
):
    result = {
        "experiment": experiment_name,
        "image_path": image_path,
        "original_prompt_zh": prompt_info["normalized_prompt_zh"],
        "translated_prompt_raw": prompt_texts["translated_prompt_raw"],
        "subject_prompt_en": prompt_texts["subject_prompt_en"],
        "attr_prompt_en": prompt_texts["attr_prompt_en"],
        "selected_box_xyxy": selected_candidate["box_xyxy"] if selected_candidate else None,
        "selected_score_det": selected_candidate["det_score"] if selected_candidate else None,
        "selected_score_attr": selected_candidate.get("attr_score") if selected_candidate else None,
        "selected_score_final": selected_candidate.get("final_score") if selected_candidate else None,
        "selected_phrase": selected_candidate["phrase"] if selected_candidate else None,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "ablation_flags": ablation_flags,
        "timing_ms": timing_ms,
    }
    if extra:
        result.update(extra)

    with open(os.path.join(output_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def add_common_args(parser):
    parser.add_argument("--config", type=str, required=True, help="path to GroundingDINO config file")
    parser.add_argument("--grounded_checkpoint", type=str, required=True, help="path to GroundingDINO checkpoint")
    parser.add_argument("--sam_version", type=str, default="vit_h", help="SAM ViT version: vit_b / vit_l / vit_h")
    parser.add_argument("--sam_checkpoint", type=str, required=False, help="path to SAM checkpoint")
    parser.add_argument("--sam_hq_checkpoint", type=str, default=None, help="path to SAM-HQ checkpoint")
    parser.add_argument("--use_sam_hq", action="store_true", help="use SAM-HQ for mask prediction")
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="Chinese attribute prompt")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="output directory")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="GroundingDINO box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="GroundingDINO text threshold")
    parser.add_argument("--device", type=str, default="cpu", help="cpu/cuda")
    parser.add_argument("--bert_base_uncased_path", type=str, default="", help="optional local bert path")
    parser.add_argument("--enable_translate", action="store_true", help="translate Chinese prompt to English")
    parser.add_argument("--translation_model", type=str, default="opus-mt-zh-en", help="translation model name")
    parser.add_argument("--alpha", type=float, default=0.5, help="weight for detection score")
    parser.add_argument("--beta", type=float, default=0.5, help="weight for attribute score")
    parser.add_argument("--topk_candidates", type=int, default=10, help="max candidates kept before reranking")
    parser.add_argument("--merge_iou_thresh", type=float, default=0.7, help="IoU threshold for candidate deduplication")
    parser.add_argument("--save_all_candidates", action="store_true", help="save visualization with all candidates")
    return parser


def bootstrap_common(args):
    os.makedirs(args.output_dir, exist_ok=True)
    prompt_info = parse_attribute_prompt(args.text_prompt)
    translator = None
    if args.enable_translate:
        translator = PromptTranslator(args.translation_model, args.device)
    prompt_texts = infer_prompt_texts(prompt_info, translator, args.enable_translate)
    image_pil, image_tensor = load_image(args.input_image)
    image_rgb = np.array(image_pil)
    grounding_model = load_grounding_model(
        args.config, args.grounded_checkpoint, args.bert_base_uncased_path, args.device
    )
    predictor = load_sam_predictor(
        args.sam_version, args.sam_checkpoint, args.sam_hq_checkpoint, args.use_sam_hq, args.device
    )
    return {
        "prompt_info": prompt_info,
        "prompt_texts": prompt_texts,
        "image_pil": image_pil,
        "image_tensor": image_tensor,
        "image_rgb": image_rgb,
        "grounding_model": grounding_model,
        "sam_predictor": predictor,
    }


def finalize_experiment(
    experiment_name,
    args,
    prompt_info,
    prompt_texts,
    image_rgb,
    predictor,
    selected_candidate,
    candidates,
    ablation_flags,
    start_time,
    extra=None,
):
    if selected_candidate is not None:
        mask = predict_mask_from_box(predictor, image_rgb, selected_candidate["box_xyxy"], args.device)
    else:
        mask = None

    label = (
        f"{prompt_info['normalized_prompt_zh']} "
        f"det={selected_candidate['det_score']:.3f} "
        f"attr={selected_candidate.get('attr_score', 0.0):.3f} "
        f"final={selected_candidate.get('final_score', 0.0):.3f}"
        if selected_candidate is not None
        else f"{prompt_info['normalized_prompt_zh']} no_detection"
    )

    overlay_and_save(
        image_rgb,
        selected_candidate["box_xyxy"] if selected_candidate else None,
        mask,
        os.path.join(args.output_dir, "annotated.jpg"),
        label,
    )
    save_mask_image(mask, os.path.join(args.output_dir, "mask.png"))

    if args.save_all_candidates:
        save_all_candidates_image(image_rgb, candidates, os.path.join(args.output_dir, "all_candidates.jpg"))

    export_result(
        args.output_dir,
        experiment_name,
        args.input_image,
        prompt_info,
        prompt_texts,
        selected_candidate,
        candidates,
        ablation_flags,
        timing_ms=(time.time() - start_time) * 1000.0,
        extra=extra,
    )


def run_experiment_exp2(context, args):
    start_time = time.time()
    prompt_info = context["prompt_info"]
    prompt_texts = context["prompt_texts"]
    candidates, subject_prompt_used = build_candidates_from_grounding(
        context["grounding_model"],
        context["image_tensor"],
        context["image_pil"].size,
        prompt_texts["subject_prompt_en"],
        args.box_threshold,
        args.text_threshold,
        args.device,
        source="subject",
    )
    candidates = deduplicate_candidates(candidates, args.merge_iou_thresh)
    candidates = truncate_candidates(candidates, args.topk_candidates)
    candidates = compute_attribute_scores(candidates, prompt_info["attribute_value"])
    ranked = rerank_candidates(candidates, args.alpha, args.beta, apply_attribute_rerank=not args.attribute_only_rerank)
    selected = ranked[0] if ranked else None

    finalize_experiment(
        "exp2",
        args,
        prompt_info,
        prompt_texts,
        context["image_rgb"],
        context["sam_predictor"],
        selected,
        ranked,
        {
            "attribute_only_rerank": bool(args.attribute_only_rerank),
        },
        start_time,
        extra={
            "subject_prompt_used": subject_prompt_used,
            "attr_prompt_en_role": "informational_only",
        },
    )


def run_experiment_exp3(context, args):
    start_time = time.time()
    prompt_info = context["prompt_info"]
    prompt_texts = context["prompt_texts"]
    candidates, attr_prompt_used = build_candidates_from_grounding(
        context["grounding_model"],
        context["image_tensor"],
        context["image_pil"].size,
        prompt_texts["attr_prompt_en"],
        args.box_threshold,
        args.text_threshold,
        args.device,
        source="attr",
    )
    candidates = deduplicate_candidates(candidates, args.merge_iou_thresh)
    candidates = truncate_candidates(candidates, args.topk_candidates)
    candidates = compute_attribute_scores(candidates, prompt_info["attribute_value"])
    ranked = rerank_candidates(candidates, args.alpha, args.beta, apply_attribute_rerank=True)
    selected = ranked[0] if ranked else None

    finalize_experiment(
        "exp3",
        args,
        prompt_info,
        prompt_texts,
        context["image_rgb"],
        context["sam_predictor"],
        selected,
        ranked,
        {},
        start_time,
        extra={
            "attr_prompt_used": attr_prompt_used,
            "subject_prompt_en_role": "informational_only",
        },
    )


def run_experiment_exp4(context, args):
    start_time = time.time()
    prompt_info = context["prompt_info"]
    prompt_texts = context["prompt_texts"].copy()
    if args.subject_prompt_override:
        prompt_texts["subject_prompt_en"] = args.subject_prompt_override.strip().lower()
    if args.attr_prompt_override:
        prompt_texts["attr_prompt_en"] = args.attr_prompt_override.strip().lower()

    subject_candidates, subject_prompt_used = build_candidates_from_grounding(
        context["grounding_model"],
        context["image_tensor"],
        context["image_pil"].size,
        prompt_texts["subject_prompt_en"],
        args.box_threshold,
        args.text_threshold,
        args.device,
        source="subject",
    )
    attr_candidates, attr_prompt_used = build_candidates_from_grounding(
        context["grounding_model"],
        context["image_tensor"],
        context["image_pil"].size,
        prompt_texts["attr_prompt_en"],
        args.box_threshold,
        args.text_threshold,
        args.device,
        source="attr",
    )
    subject_candidates = truncate_candidates(deduplicate_candidates(subject_candidates, args.merge_iou_thresh), args.topk_candidates)
    attr_candidates = truncate_candidates(deduplicate_candidates(attr_candidates, args.merge_iou_thresh), args.topk_candidates)

    if args.disable_merge:
        merged = subject_candidates + attr_candidates
    else:
        merged = merge_candidate_sets(subject_candidates, attr_candidates, args.merge_iou_thresh)
    merged = truncate_candidates(merged, args.topk_candidates)
    merged = compute_attribute_scores(merged, prompt_info["attribute_value"])
    ranked = rerank_candidates(merged, args.alpha, args.beta, apply_attribute_rerank=not args.disable_rerank)
    selected = ranked[0] if ranked else None

    finalize_experiment(
        "exp4",
        args,
        prompt_info,
        prompt_texts,
        context["image_rgb"],
        context["sam_predictor"],
        selected,
        ranked,
        {
            "disable_merge": bool(args.disable_merge),
            "disable_rerank": bool(args.disable_rerank),
            "subject_prompt_override": args.subject_prompt_override or "",
            "attr_prompt_override": args.attr_prompt_override or "",
        },
        start_time,
        extra={
            "subject_prompt_used": subject_prompt_used,
            "attr_prompt_used": attr_prompt_used,
            "subject_path_candidate_count": len(subject_candidates),
            "attr_path_candidate_count": len(attr_candidates),
            "merged_candidate_count": len(merged),
            "winning_source": selected["source"] if selected else None,
            "subject_candidates": subject_candidates,
            "attr_candidates": attr_candidates,
            "post_merge_candidates": ranked,
        },
    )
