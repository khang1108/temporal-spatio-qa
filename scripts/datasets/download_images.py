"""Fast downloader for SpatialMQA images (only the required 5,063 images, ~800MB).

Avoids downloading the full 6.2GB COCO2017 zip file which is heavily throttled.
Downloads directly from Hugging Face dataset CDN at high speed.
"""

import os
import sys
import argparse

def download_from_hf(output_dir="data/spatial_mqa"):
    print("=" * 65)
    print("Downloading SpatialMQA images directly from Hugging Face CDN...")
    print(f"Target directory: {os.path.abspath(output_dir)}")
    print("Total size: ~800 MB (only the 5,063 images used by SpatialMQA)")
    print("=" * 65)

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="liuziyan/SpatialMQA",
            repo_type="dataset",
            allow_patterns=["images/*"],
            local_dir=output_dir,
            resume_download=True
        )
        images_dir = os.path.join(output_dir, "images")
        count = len(os.listdir(images_dir)) if os.path.exists(images_dir) else 0
        print(f"\n[SUCCESS] Download completed! Total images: {count} in {images_dir}")
    except Exception as e:
        print(f"\n[ERROR] Hugging Face download failed: {e}")
        print("Falling back to multi-threaded direct image retrieval...")
        download_via_fallback(output_dir)

def download_via_fallback(output_dir="data/spatial_mqa"):
    import json
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Collect all needed image names from train, dev, test
    needed_images = set()
    for split in ["train.jsonl", "dev.jsonl", "test.jsonl"]:
        path = os.path.join(output_dir, split)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    needed_images.add(json.loads(line.strip())["image"])

    missing = [img for img in needed_images if not os.path.exists(os.path.join(images_dir, img))]
    print(f"Total needed: {len(needed_images)}, already downloaded: {len(needed_images) - len(missing)}, to fetch: {len(missing)}")

    def fetch(img):
        url = f"https://huggingface.co/datasets/liuziyan/SpatialMQA/resolve/main/images/{img}"
        dst = os.path.join(images_dir, img)
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception:
            # Fallback to COCO S3
            url2 = f"http://images.cocodataset.org/test2017/{img}"
            try:
                urllib.request.urlretrieve(url2, dst)
            except Exception as e2:
                print(f"Failed {img}: {e2}")

    with ThreadPoolExecutor(max_workers=32) as executor:
        list(tqdm(executor.map(fetch, missing), total=len(missing), desc="Downloading images"))

    print(f"\n[SUCCESS] Finished! Total images in {images_dir}: {len(os.listdir(images_dir))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    download_from_hf()
