#!/usr/bin/env python3
import argparse
import json
import os
import random
import time
from collections import defaultdict

import requests
from pycocotools.coco import COCO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a COCO image subset by category."
    )
    parser.add_argument(
        "--ann_file",
        type=str,
        required=True,
        help="Path to COCO annotation json (e.g., instances_val2017.json)",
    )
    parser.add_argument(
        "--categories",
        type=str,
        required=True,
        help="Comma-separated COCO category names, e.g. person,dog,car,bicycle,chair",
    )
    parser.add_argument(
        "--per_category",
        type=int,
        default=50,
        help="Number of images per category.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/coco_subset",
        help="Output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--unique_images",
        action="store_true",
        help="Enforce globally unique images across categories.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="HTTP timeout in seconds.",
    )
    return parser.parse_args()


def safe_download(url, save_path, timeout):
    if os.path.exists(save_path):
        return True, "exists"
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True, "downloaded"
    except Exception as e:
        return False, str(e)


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    coco = COCO(args.ann_file)
    category_names = [c.strip() for c in args.categories.split(",") if c.strip()]
    cat_ids = coco.getCatIds(catNms=category_names)
    cats = coco.loadCats(cat_ids)
    id_to_name = {c["id"]: c["name"] for c in cats}
    found_names = {c["name"] for c in cats}
    missing = [name for name in category_names if name not in found_names]
    if missing:
        raise ValueError(f"Categories not found in annotation: {missing}")

    selected = defaultdict(list)
    used_img_ids = set()
    download_log = []

    for cat_name in category_names:
        cat_id = next(k for k, v in id_to_name.items() if v == cat_name)
        img_ids = coco.getImgIds(catIds=[cat_id])
        random.shuffle(img_ids)

        for img_id in img_ids:
            if len(selected[cat_name]) >= args.per_category:
                break
            if args.unique_images and img_id in used_img_ids:
                continue
            selected[cat_name].append(img_id)
            used_img_ids.add(img_id)

        if len(selected[cat_name]) < args.per_category:
            print(
                f"[WARN] Category '{cat_name}' only got {len(selected[cat_name])} images "
                f"(requested {args.per_category})."
            )

    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Download unique image ids to avoid redundant downloads.
    all_img_ids = sorted({img_id for ids in selected.values() for img_id in ids})
    print(f"Total unique images to download: {len(all_img_ids)}")

    for idx, img_id in enumerate(all_img_ids, start=1):
        img_info = coco.loadImgs([img_id])[0]
        file_name = img_info["file_name"]
        url = img_info.get("coco_url")
        if not url:
            # fallback to Flickr url if available
            url = img_info.get("flickr_url")
        if not url:
            download_log.append(
                {"img_id": img_id, "file_name": file_name, "ok": False, "reason": "no_url"}
            )
            continue
        save_path = os.path.join(images_dir, file_name)
        ok, msg = safe_download(url, save_path, timeout=args.timeout)
        download_log.append(
            {"img_id": img_id, "file_name": file_name, "ok": ok, "reason": msg}
        )
        if idx % 20 == 0 or idx == len(all_img_ids):
            print(f"Downloaded {idx}/{len(all_img_ids)}")
        time.sleep(0.02)

    manifest = {
        "ann_file": args.ann_file,
        "categories": category_names,
        "per_category": args.per_category,
        "unique_images": args.unique_images,
        "selected_img_ids": {k: v for k, v in selected.items()},
        "download_log": download_log,
    }
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    stats_path = os.path.join(args.output_dir, "stats.txt")
    ok_count = sum(1 for x in download_log if x["ok"])
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"requested_per_category={args.per_category}\n")
        for cat_name in category_names:
            f.write(f"{cat_name}={len(selected[cat_name])}\n")
        f.write(f"unique_images_selected={len(all_img_ids)}\n")
        f.write(f"download_ok={ok_count}\n")
        f.write(f"download_fail={len(download_log) - ok_count}\n")

    print(f"Done. Manifest: {manifest_path}")
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
