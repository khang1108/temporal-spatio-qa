"""Evaluate LLaVA-1.5 or SpaceLLaVA (Zero-shot or with LoRA Adapter) on SpatialMQA test set.

Usage:
  # Evaluate Base LLaVA-1.5 (Zero-shot):
  python scripts/run_eval.py --model_path liuhaotian/llava-v1.5-7b

  # Evaluate LoRA fine-tuned checkpoint:
  python scripts/run_eval.py \
      --model_path liuhaotian/llava-v1.5-7b \
      --lora_path experiments/checkpoints/llava_1.5_7b_lora

  # Quick debug on 10 samples:
  python scripts/run_eval.py --model_path liuhaotian/llava-v1.5-7b --max_samples 10
"""

import os
import sys
import re
import json
import argparse
import torch
from tqdm import tqdm
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.evaluation.metrics import evaluate_predictions, print_evaluation_report


TASK_PROMPT = (
    "You are currently a senior expert in spatial relation reasoning. \n "
    "Given an Image, a Question and Options, your task is to answer the correct spatial relation. "
    "Note that you only need to choose one option from the all options without explaining any reason. \n "
    "Input: Image: , Question: {question}, Options: {options}. \n Output:"
)


def load_model_and_tokenizer(model_path: str, lora_path: str = None, device: str = "cuda", load_4bit: bool = False, load_8bit: bool = False):
    try:
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from llava.utils import disable_torch_init
    except ImportError:
        raise ImportError(
            "llava package is required for inference. "
            "Please install via: pip install git+https://github.com/haotian-liu/LLaVA.git"
        )

    # Disable torchao check in peft if incompatible torchao version is present on environment
    try:
        import peft.import_utils as peft_import_utils
        peft_import_utils.is_torchao_available = lambda: False
    except (ImportError, AttributeError):
        pass
    try:
        import peft.tuners.lora.torchao as peft_torchao
        peft_torchao.is_torchao_available = lambda: False
    except (ImportError, AttributeError):
        pass

    disable_torch_init()

    if lora_path:
        actual_lora_dir = lora_path
        if not os.path.exists(lora_path):
            print(f"Downloading LoRA adapter checkpoint from Hugging Face Hub: {lora_path} ...")
            from huggingface_hub import snapshot_download
            actual_lora_dir = snapshot_download(
                repo_id=lora_path,
                allow_patterns=["adapter_*", "non_lora_trainables.bin", "config.json"]
            )
            print(f"Downloaded to local cache: {actual_lora_dir}")

        raw_name = get_model_name_from_path(lora_path)
        model_name = f"llava-{raw_name}" if "llava" not in raw_name.lower() else raw_name
        if "lora" not in model_name.lower():
            model_name = f"{model_name}-lora"

        print(f"Detected model_name: '{model_name}' for LLaVA LoRA loading.")
        print(f"Loading LoRA model from base '{model_path}' and adapter '{actual_lora_dir}' (4bit={load_4bit}, 8bit={load_8bit})...")

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=actual_lora_dir,
            model_base=model_path,
            model_name=model_name,
            load_4bit=load_4bit,
            load_8bit=load_8bit,
            device_map="auto" if device == "cuda" else "cpu"
        )
    else:
        model_name = get_model_name_from_path(model_path)
        print(f"Loading base model from '{model_path}' (4bit={load_4bit}, 8bit={load_8bit})...")
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            load_4bit=load_4bit,
            load_8bit=load_8bit,
            device_map="auto" if device == "cuda" else "cpu"
        )

    return tokenizer, model, image_processor




def evaluate_benchmark(
    model_path: str,
    lora_path: str = None,
    test_jsonl: str = "data/spatial_mqa/test.jsonl",
    image_dir: str = "data/spatial_mqa/images",
    output_jsonl: str = "experiments/predictions/llava_predictions.jsonl",
    max_samples: int = None,
    temperature: float = 0.4,
    load_4bit: bool = False,
    load_8bit: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running evaluation on device: {device} (load_4bit={load_4bit}, load_8bit={load_8bit})")

    tokenizer, model, image_processor = load_model_and_tokenizer(
        model_path=model_path,
        lora_path=lora_path,
        device=device,
        load_4bit=load_4bit,
        load_8bit=load_8bit
    )


    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN
    )
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token, process_images

    # Load test set
    with open(test_jsonl, "r", encoding="utf-8") as f:
        samples = [json.loads(line.strip()) for line in f]

    if max_samples:
        samples = samples[:max_samples]

    print(f"Evaluating {len(samples)} samples on SpatialMQA...")
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    predictions = []

    for item in tqdm(samples, desc="Inference"):
        q = item["question"]
        opts = item["options"]
        gt_answer = item["answer"]
        img_name = item["image"]
        img_path = os.path.join(image_dir, img_name)

        if not os.path.exists(img_path):
            # Fallback placeholder if image not downloaded
            image = Image.new("RGB", (336, 336), color=(128, 128, 128))
        else:
            image = Image.open(img_path).convert("RGB")

        prompt_text = TASK_PROMPT.format(question=q, options="; ".join(opts))
        if getattr(model.config, 'mm_use_im_start_end', False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt_text
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text

        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()


        images_tensor = process_images([image], image_processor, model.config)
        if device == "cuda":
            images_tensor = images_tensor.to(model.device, dtype=torch.float16)

        input_ids = tokenizer_image_token(
            full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=[image.size],
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                max_new_tokens=64,
                use_cache=True,
            )

        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        pred_record = {
            "id": item.get("id"),
            "image": img_name,
            "question": q,
            "options": opts,
            "answer": gt_answer,
            "output": output_text
        }
        predictions.append(pred_record)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nSaved raw predictions to: {output_jsonl}")

    # Compute & Print Metrics
    results = evaluate_predictions(predictions)
    print_evaluation_report(results, model_name=os.path.basename(lora_path or model_path))

    metrics_json = output_jsonl.rsplit(".", 1)[0] + "_metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved evaluation metrics summary to: {metrics_json}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--test_jsonl", type=str, default="data/spatial_mqa/test.jsonl")
    parser.add_argument("--image_dir", type=str, default="data/spatial_mqa/images")
    parser.add_argument("--output_jsonl", type=str, default="experiments/predictions/eval_preds.jsonl")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--load_4bit", action="store_true", help="Load model in 4-bit quantization (useful for T4 16GB GPU)")
    parser.add_argument("--load_8bit", action="store_true", help="Load model in 8-bit quantization")
    args = parser.parse_args()

    evaluate_benchmark(
        model_path=args.model_path,
        lora_path=args.lora_path,
        test_jsonl=args.test_jsonl,
        image_dir=args.image_dir,
        output_jsonl=args.output_jsonl,
        max_samples=args.max_samples,
        temperature=args.temperature,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit
    )



if __name__ == "__main__":
    main()
