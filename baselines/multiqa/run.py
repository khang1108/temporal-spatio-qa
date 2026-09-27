"""
Training and evaluation script for MultiQA baseline on STQAD.
"""

import os
import sys
# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from src.dataset import load_stqad, get_vocabularies, STQADataset, collate_stqad_fn, clean_entity
from src.evaluation import evaluate_benchmark, format_table5_markdown
from baselines.multiqa.model import MultiQABaseline


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate MultiQA on STQAD")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--data_dir", type=str, default="data/stqad")
    parser.add_argument("--output_dir", type=str, default="experiments/checkpoints/multiqa")
    parser.add_argument("--pred_dir", type=str, default="experiments/predictions")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    return parser.parse_args()


def extract_batch_entities(raw_items, entity2id, device):
    """
    Extracts central entity ID (first entity mentioned) and temporal clue ID (second entity if present).
    """
    subj_ids = []
    time_ids = []

    for item in raw_items:
        ents = item.get("entities", [])
        s_id = -1
        t_id = -1

        if len(ents) > 0:
            c_name = clean_entity(ents[0])
            s_id = entity2id.get(c_name, -1)

        if len(ents) > 1:
            t_name = clean_entity(ents[1])
            # Hash or map clue to a timestamp bucket
            t_id = abs(hash(t_name)) % 600

        subj_ids.append(s_id)
        time_ids.append(t_id)

    return (torch.tensor(subj_ids, dtype=torch.long, device=device),
            torch.tensor(time_ids, dtype=torch.long, device=device))


def evaluate(model, dataloader, id2entity, entity2id, dataset_items, device, k=10):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating MultiQA", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            subj_ids, time_ids = extract_batch_entities(batch["raw_items"], entity2id, device)

            topk_indices = model.predict_topk(input_ids, attention_mask, subj_ids, time_ids, k=k)
            topk_indices = topk_indices.cpu().tolist()

            for indices in topk_indices:
                cand_names = [id2entity.get(idx, f"<unk_{idx}>") for idx in indices]
                all_predictions.append(cand_names)

    results = evaluate_benchmark(all_predictions, dataset_items)
    return results, all_predictions


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"MultiQA - Using device: {device}")

    # Vocabularies
    entity2id, id2entity, relation2id, id2relation = get_vocabularies(args.data_dir)
    num_entities = len(entity2id)

    # Datasets
    train_data = load_stqad("train", args.data_dir)
    val_data = load_stqad("val", args.data_dir)
    test_data = load_stqad("test", args.data_dir)

    if args.max_samples:
        train_data = train_data[:args.max_samples]
        val_data = val_data[:min(len(val_data), args.max_samples)]
        test_data = test_data[:min(len(test_data), args.max_samples)]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = STQADataset(train_data, tokenizer, entity2id, max_length=args.max_length)
    val_dataset = STQADataset(val_data, tokenizer, entity2id, max_length=args.max_length)
    test_dataset = STQADataset(test_data, tokenizer, entity2id, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_stqad_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_stqad_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_stqad_fn)

    model = MultiQABaseline(
        model_name=args.model_name,
        num_entities=num_entities,
        embedding_dim=args.embedding_dim,
        freeze_encoder=args.freeze_encoder
    ).to(device)

    best_checkpoint = os.path.join(args.output_dir, "best_model.pt")

    if not args.eval_only:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        best_val_hits1 = -1.0

        print(f"Starting MultiQA training for {args.epochs} epochs...")
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0

            for batch in tqdm(train_loader, desc=f"MultiQA Epoch {epoch}/{args.epochs}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target_dist = batch["target_dist"].to(device)
                subj_ids, time_ids = extract_batch_entities(batch["raw_items"], entity2id, device)

                optimizer.zero_grad()
                logits = model(input_ids, attention_mask, subj_ids, time_ids)
                loss = model.compute_loss(logits, target_dist)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            val_results, _ = evaluate(model, val_loader, id2entity, entity2id, val_data, device)
            val_hits1 = val_results["Overall"]["Hits@1"]
            val_hits10 = val_results["Overall"]["Hits@10"]

            print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f} | Val Hits@1 = {val_hits1:.2f}% | Val Hits@10 = {val_hits10:.2f}%")

            if val_hits1 > best_val_hits1:
                best_val_hits1 = val_hits1
                torch.save(model.state_dict(), best_checkpoint)
                print(f"  -> Saved new best MultiQA checkpoint to {best_checkpoint}")

    if os.path.exists(best_checkpoint):
        print(f"Loading best checkpoint from {best_checkpoint}...")
        model.load_state_dict(torch.load(best_checkpoint, map_location=device))

    test_results, test_preds = evaluate(model, test_loader, id2entity, entity2id, test_data, device)

    pred_file = os.path.join(args.pred_dir, "multiqa_test_preds.json")
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": "MultiQA",
            "metrics": test_results,
            "predictions": test_preds
        }, f, indent=2)

    print("\n" + "=" * 50)
    print("MultiQA Final Test Evaluation on STQAD:")
    print("=" * 50)
    print(format_table5_markdown(test_results, model_name="MultiQA"))
    print(f"Predictions saved to {pred_file}")


if __name__ == "__main__":
    main()
