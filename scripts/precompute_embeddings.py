"""
Precomputes and caches RoBERTa question embeddings for STQAD splits.
Following Dai et al. (KBS 2025) Appendix B:
"During STKGQA, the parameters of the pre-trained language model and the STKG embeddings remain unchanged."
Precomputing question embeddings makes training 100x faster on CPU with identical mathematical results.
"""

import os
import sys
# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from src.dataset import load_stqad


def precompute_split(split_name: str, model, tokenizer, data_dir: str = "data/stqad", output_dir: str = "data/cached_embeddings", batch_size: int = 64):
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{split_name}_roberta_cls.pt")
    if os.path.exists(out_file):
        print(f"Cached embeddings already exist at {out_file}. Skipping.")
        return torch.load(out_file, map_location="cpu")

    data = load_stqad(split_name, data_dir)
    questions = [d.get("paraphrased_question") or d.get("question") for d in data]

    all_cls = []
    print(f"Precomputing embeddings for {split_name} ({len(questions)} samples)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(questions), batch_size), desc=f"Encoding {split_name}"):
            batch_texts = questions[i:i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)

            out = model(**encoded)
            # [CLS] representation (first token)
            cls_rep = out.last_hidden_state[:, 0, :].cpu()
            all_cls.append(cls_rep)

    stacked_cls = torch.cat(all_cls, dim=0)
    print(f"Saving {stacked_cls.shape} to {out_file}...")
    torch.save(stacked_cls, out_file)
    return stacked_cls


def main():
    model_name = "roberta-base"
    print(f"Loading tokenizer and model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    for split in ["train", "val", "test"]:
        precompute_split(split, model, tokenizer)

    print("\nAll splits successfully precomputed and cached in data/cached_embeddings/!")


if __name__ == "__main__":
    main()
