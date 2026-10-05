"""Convert SpatialMQA JSONL files into standard LLaVA instruction tuning format.

Each sample is structured into:
- id: unique sample identifier
- image: image filename
- conversations: [
    {"from": "human", "value": "<image>\n{TASK_PROMPT} Input: ... Output:"},
    {"from": "gpt", "value": "{answer}"}
  ]
"""

import json
import os
import argparse

TASK_PROMPT_TEMPLATE = (
    "You are currently a senior expert in spatial relation reasoning. \n "
    "Given an Image, a Question and Options, your task is to answer the correct spatial relation. "
    "Note that you only need to choose one option from the all options without explaining any reason. \n "
    "Input: Image: <image>, Question: {question}, Options: {options}. \n Output:"
)


def format_split(input_jsonl, output_json, split_name="train"):
    samples = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            data = json.loads(line.strip())
            question = data["question"]
            options_str = "; ".join(data["options"])
            answer = data["answer"]
            image_name = data["image"]

            human_prompt = (
                f"<image>\n"
                f"{TASK_PROMPT_TEMPLATE.format(question=question, options=options_str)}"
            )

            samples.append({
                "id": f"spatial_mqa_{split_name}_{idx:05d}",
                "image": image_name,
                "conversations": [
                    {"from": "human", "value": human_prompt},
                    {"from": "gpt", "value": answer}
                ]
            })

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Successfully converted {len(samples)} samples from {input_jsonl} to {output_json}")


def main():
    parser = argparse.ArgumentParser(description="Format SpatialMQA data for LLaVA fine-tuning")
    parser.add_argument("--data_dir", type=str, default="data/spatial_mqa", help="Path to SpatialMQA directory")
    args = parser.parse_args()

    train_jsonl = os.path.join(args.data_dir, "train.jsonl")
    train_json = os.path.join(args.data_dir, "train_3780.json")
    dev_jsonl = os.path.join(args.data_dir, "dev.jsonl")
    dev_json = os.path.join(args.data_dir, "dev_536.json")

    if os.path.exists(train_jsonl):
        format_split(train_jsonl, train_json, "train")
    if os.path.exists(dev_jsonl):
        format_split(dev_jsonl, dev_json, "dev")


if __name__ == "__main__":
    main()
