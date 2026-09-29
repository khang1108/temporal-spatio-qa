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

from src.data.dataset import load_stqad, get_vocabularies, STQADataset, collate_stqad_fn, clean_entity, classify_question_clues
from src.evaluation.metrics import evaluate_benchmark, format_table5_markdown
from src.utils.checkpoint import ensure_checkpoint, load_model_checkpoint
from baselines.multiqa.model import MultiQABaseline


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate MultiQA on STQAD")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--data_dir", type=str, default="data/stqad")
    parser.add_argument("--output_dir", type=str, default="experiments/checkpoints/multiqa")
    parser.add_argument("--pred_dir", type=str, default="experiments/predictions")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs (paper: 60)")
    parser.add_argument("--batch_size", type=int, default=150, help="Training batch size (paper: 150)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (paper: 2e-5)")
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint_url", type=str, default=None, help="Download URL if checkpoint is not found locally")
    return parser.parse_args()


def extract_batch_entities(raw_items, entity2id, device):
    """
    Extracts entity IDs for MultiQA (Temporal KGQA baseline):
      - STEP 1: Disentangle entities into Central Subject [ENT] and Temporal Clue [TS] (Section 5.2).
      - STEP 2: Map Central Subject [ENT] to global KG entity ID.
      - STEP 3: Map Temporal Clue [TS] to time bucket ID (modulo 600).
                (Note: MultiQA has no spatial embedding or spatial reasoning module).
      - STEP 4: Collate into PyTorch tensors on the target device.
    """
    subj_ids = []
    time_ids = []

    for item in raw_items:
        ents = item.get("entities", [])
        q_text = item.get("question", "")

        # STEP 1: Determine functional entity roles
        central, time_clue, _ = classify_question_clues(q_text, ents)

        # STEP 2: Central entity vocabulary ID lookup
        s_id = entity2id.get(clean_entity(central), -1) if central else -1

        # STEP 3: Bucket hashing for temporal clue representation
        t_id = (abs(hash(clean_entity(time_clue))) % 600) if time_clue else -1

        subj_ids.append(s_id)
        time_ids.append(t_id)

    # STEP 4: Construct PyTorch long tensors
    return (torch.tensor(subj_ids, dtype=torch.long, device=device),
            torch.tensor(time_ids, dtype=torch.long, device=device))


def evaluate(model, dataloader, id2entity, entity2id, dataset_items, device, k=10, return_details=False):
    """
    Evaluates MultiQA on STQAD benchmark:
      - STEP 1: Batched temporal clue extraction (Question + Subject + Time).
      - STEP 2: Forward pass through MultiQA fusion network.
      - STEP 3: Retrieve top-k candidate entities from output logits.
      - STEP 4: Map predicted indices to entity names.
      - STEP 5: Compute Hits@1 and Hits@10 across question categories.
    """
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating MultiQA", leave=False):
            cls_rep = batch["cls_rep"].to(device) if batch.get("cls_rep") is not None else None
            input_ids = batch["input_ids"].to(device) if batch.get("input_ids") is not None else None
            attention_mask = batch["attention_mask"].to(device) if batch.get("attention_mask") is not None else None

            # STEP 1: Subject and temporal clue tensor extraction
            subj_ids, time_ids = extract_batch_entities(batch["raw_items"], entity2id, device)

            # STEP 2 & 3: Model scoring and top-k candidate retrieval
            topk_indices = model.predict_topk(input_ids, attention_mask, subj_ids, time_ids, cls_rep=cls_rep, k=k)
            topk_indices = topk_indices.cpu().tolist()

            # STEP 4: Map indices to entity names
            for indices in topk_indices:
                cand_names = [id2entity.get(idx, f"<unk_{idx}>") for idx in indices]
                all_predictions.append(cand_names)

    # STEP 5: Metric calculation across all categories
    if return_details:
        results, sample_details, failure_summary = evaluate_benchmark(all_predictions, dataset_items, return_details=True)
        return results, all_predictions, sample_details, failure_summary

    results = evaluate_benchmark(all_predictions, dataset_items)
    return results, all_predictions


def main():
    # =========================================================================
    # STEP 1: Parse Command-Line Arguments & Initialize Directory Structure
    # =========================================================================
    # - args.output_dir: directory to store model checkpoints (e.g. best_model.pt)
    # - args.pred_dir: directory to store JSON prediction logs & evaluation outputs
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)

    # =========================================================================
    # STEP 2: Configure Hardware Compute Device (CUDA GPU / CPU)
    # =========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"MultiQA - Using device: {device}")

    # =========================================================================
    # STEP 3: Load Knowledge Graph Vocabularies & Entity-to-ID Mappings
    # =========================================================================
    # Loads entity2id and relation2id to map textual entities into integer index spaces.
    entity2id, id2entity, relation2id, id2relation = get_vocabularies(args.data_dir)
    num_entities = len(entity2id)

    # =========================================================================
    # STEP 4: Load Dataset Splits (Train, Validation, Test)
    # =========================================================================
    # STQAD json splits containing questions, paraphrase variants, entities, and answers.
    train_data = load_stqad("train", args.data_dir)
    val_data = load_stqad("val", args.data_dir)
    test_data = load_stqad("test", args.data_dir)

    # =========================================================================
    # STEP 5: Check & Load Precomputed RoBERTa CLS Embeddings (Cache Acceleration)
    # =========================================================================
    # If cached CLS vectors exist under data/cached_embeddings/, bypass on-the-fly
    # transformer forward passes to drastically speed up training on CPU/GPU.
    cached_dir = "data/cached_embeddings"
    train_cached, val_cached, test_cached = None, None, None
    if not args.max_samples and os.path.exists(os.path.join(cached_dir, "train_roberta_cls.pt")):
        print("Found cached RoBERTa embeddings in data/cached_embeddings/! Fast CPU training enabled.")
        train_cached = torch.load(os.path.join(cached_dir, "train_roberta_cls.pt"), map_location="cpu")
        val_cached = torch.load(os.path.join(cached_dir, "val_roberta_cls.pt"), map_location="cpu")
        test_cached = torch.load(os.path.join(cached_dir, "test_roberta_cls.pt"), map_location="cpu")

    if args.max_samples:
        train_data = train_data[:args.max_samples]
        val_data = val_data[:min(len(val_data), args.max_samples)]
        test_data = test_data[:min(len(test_data), args.max_samples)]

    # =========================================================================
    # STEP 6: Initialize Text Tokenizer & PyTorch DataLoaders
    # =========================================================================
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = STQADataset(train_data, tokenizer, entity2id, max_length=args.max_length, cached_cls=train_cached)
    val_dataset = STQADataset(val_data, tokenizer, entity2id, max_length=args.max_length, cached_cls=val_cached)
    test_dataset = STQADataset(test_data, tokenizer, entity2id, max_length=args.max_length, cached_cls=test_cached)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_stqad_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_stqad_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_stqad_fn)

    # =========================================================================
    # STEP 7: Initialize MultiQA Temporal Baseline Model Architecture
    # =========================================================================
    # Components:
    #   - Text Encoder: RoBERTa-base (768-d text representations)
    #   - Subject Entity Embedding: [num_entities, 512]
    #   - Temporal Bucket Embedding: [600, 512]
    #   - (Note: MultiQA has NO spatial reasoning or geo-coordinate embeddings)
    #   - Fusion: Element-wise fusion / Concatenation projection head
    model = MultiQABaseline(
        model_name=args.model_name,
        num_entities=num_entities,
        embedding_dim=args.embedding_dim,
        freeze_encoder=args.freeze_encoder
    ).to(device)

    best_checkpoint = os.path.join(args.output_dir, "best_model.pt")

    # =========================================================================
    # STEP 8: Training Loop with Validation & Early Checkpointing (If Not eval_only)
    # =========================================================================
    if not args.eval_only:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        best_val_hits1 = -1.0

        print(f"Starting MultiQA training for {args.epochs} epochs...")
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0

            for batch in tqdm(train_loader, desc=f"MultiQA Epoch {epoch}/{args.epochs}"):
                cls_rep = batch["cls_rep"].to(device) if batch.get("cls_rep") is not None else None
                input_ids = batch["input_ids"].to(device) if batch.get("input_ids") is not None else None
                attention_mask = batch["attention_mask"].to(device) if batch.get("attention_mask") is not None else None
                target_dist = batch["target_dist"].to(device)

                # Extract Central Subject & Temporal Clue
                subj_ids, time_ids = extract_batch_entities(batch["raw_items"], entity2id, device)

                optimizer.zero_grad()
                logits = model(input_ids, attention_mask, subj_ids, time_ids, cls_rep=cls_rep)
                loss = model.compute_loss(logits, target_dist)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # Evaluate on Validation split to track progress
            val_results, _ = evaluate(model, val_loader, id2entity, entity2id, val_data, device)
            val_hits1 = val_results["Overall"]["Hits@1"]
            val_hits10 = val_results["Overall"]["Hits@10"]

            print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f} | Val Hits@1 = {val_hits1:.2f}% | Val Hits@10 = {val_hits10:.2f}%")

            # Checkpoint best model based on validation Hits@1
            if val_hits1 > best_val_hits1:
                best_val_hits1 = val_hits1
                torch.save(model.state_dict(), best_checkpoint)

    # =========================================================================
    # STEP 9: Load Best Checkpoint for Evaluation
    # =========================================================================
    # Checks whether local best_model.pt exists; if absent, attempts download from URL
    if args.eval_only:
        load_model_checkpoint(model, "multiqa", checkpoint_path=best_checkpoint, url=args.checkpoint_url, device=device)
    elif os.path.exists(best_checkpoint):
        load_model_checkpoint(model, "multiqa", checkpoint_path=best_checkpoint, url=args.checkpoint_url, device=device)

    # =========================================================================
    # STEP 10: Run Final Evaluation on Test Split (Temporal Baseline Ranking)
    # =========================================================================
    test_results, test_preds, sample_details, failure_summary = evaluate(
        model, test_loader, id2entity, entity2id, test_data, device, return_details=True
    )

    # =========================================================================
    # STEP 11: Export Predictions, Metrics, and Failure Diagnostic Logs
    # =========================================================================
    pred_file = os.path.join(args.pred_dir, "multiqa_test_preds.json")
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": "MultiQA",
            "metrics": test_results,
            "failure_summary": failure_summary,
            "predictions": test_preds,
            "sample_details": sample_details
        }, f, indent=2)

    # =========================================================================
    # STEP 12: Display Benchmark Report (Table 5 Format) & Error Statistics
    # =========================================================================
    print("\n" + "=" * 50)
    print("MultiQA Final Test Evaluation on STQAD:")
    print("=" * 50)
    print(format_table5_markdown(test_results, model_name="MultiQA"))
    print("-" * 50)
    print("Error & Failure Analysis Summary:")
    print(f"Total Test Questions: {failure_summary['total_samples']}")
    print(f"Failed Hits@1:  {failure_summary['total_failed_hit1']} ({(failure_summary['total_failed_hit1']/failure_summary['total_samples'])*100:.1f}%)")
    print(f"Failed Hits@10: {failure_summary['total_failed_hit10']} ({(failure_summary['total_failed_hit10']/failure_summary['total_samples'])*100:.1f}%)")
    print("\nFailures by Constraint (Missed in Top 10):")
    for t_k, t_v in failure_summary['failed_by_temporal'].items():
        total_t = test_results.get(t_k, {}).get("count", 0)
        pct = (t_v / total_t * 100) if total_t > 0 else 0
        print(f"  - {t_k}: {t_v}/{total_t} questions failed ({pct:.1f}%)")
    for s_k, s_v in failure_summary['failed_by_spatial'].items():
        total_s = test_results.get(s_k, {}).get("count", 0)
        pct = (s_v / total_s * 100) if total_s > 0 else 0
        print(f"  - {s_k}: {s_v}/{total_s} questions failed ({pct:.1f}%)")
    print(f"\nDetailed per-question predictions & failure logs saved to: {pred_file}")


if __name__ == "__main__":
    main()
