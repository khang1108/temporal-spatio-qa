"""
STKG Knowledge Graph Embedding (KGE) Pre-training Script.
Implements Spatio-Temporal ComplEx (ST-TComplEx) pre-training on STKG facts:
    Dai et al., "Question answering over spatio-temporal knowledge graph", KBS 2025.
    Equations (2), (3) and Appendix B & C:
    - Scoring: phi_ST(s, r, o, t, l) = Re(<e_s, r (x) t (x) l, conj(e_o)>)
    - Optimizer: Adagrad with lr=0.1
    - Batch size: 1,000, Epochs: 50
    - ComplEx multiclass cross-entropy loss with N3 nuclear regularization
    - 8:1:1 train/val/test split of STKG facts (Section 6.4)
"""

import os
import sys
import json
import random
import argparse
from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from baselines.stcqa.st_embedding import STComplExEmbedding, complex_mul


class STKGFactsDataset(Dataset):
    """
    Dataset of STKG quadruplet facts: (s_id, r_id, o_id, time_id, loc_id).
    """
    def __init__(self,
                 facts: List[Tuple[int, int, int, int, int]]):
        self.facts = facts

    def __len__(self) -> int:
        return len(self.facts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s, r, o, t, l = self.facts[idx]
        return (torch.tensor(s, dtype=torch.long),
                torch.tensor(r, dtype=torch.long),
                torch.tensor(o, dtype=torch.long),
                torch.tensor(t, dtype=torch.long),
                torch.tensor(l, dtype=torch.long))


def load_facts(facts_path: str,
               entity2id: Dict[str, int],
               relation2id: Dict[str, int],
               time2id: Dict[str, int],
               coord2id: Dict[str, int],
               meta: Dict[str, Dict]) -> List[Tuple[int, int, int, int, int]]:
    """
    Loads raw TSV facts and maps them to canonical discrete vocabulary indices.
    """
    parsed = []
    with open(facts_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            s, r, o, ts, te, loc = parts[:6]

            s_id = entity2id.get(s)
            o_id = entity2id.get(o)
            r_id = relation2id.get(r)
            if s_id is None or o_id is None or r_id is None:
                continue

            # Temporal index: check start time, else end time, else entity interval
            t_id = 0
            if ts != "NONE" and ts in time2id:
                t_id = time2id[ts]
            elif te != "NONE" and te in time2id:
                t_id = time2id[te]
            elif s in meta and meta[s].get("interval"):
                inv_yr = str(meta[s]["interval"][0])
                t_id = time2id.get(inv_yr, 0)

            # Spatial index: check fact location entity coordinates
            l_id = 0
            if loc != "NONE" and loc in meta and meta[loc].get("coords"):
                lat, lon = meta[loc]["coords"]
                key = f"{lat:.4f},{lon:.4f}"
                l_id = coord2id.get(key, 0)
            elif o in meta and meta[o].get("coords"):
                lat, lon = meta[o]["coords"]
                key = f"{lat:.4f},{lon:.4f}"
                l_id = coord2id.get(key, 0)

            parsed.append((s_id, r_id, o_id, t_id, l_id))

    return parsed


def compute_n3_reg(factors: List[Tuple[torch.Tensor, torch.Tensor]], p: float = 3.0) -> torch.Tensor:
    """
    N3 regularization for complex factors per Lacroix et al. (2020).
    ||(x_re, x_im)||_3^3 = (x_re^2 + x_im^2)^(3/2).
    """
    norm = torch.tensor(0.0, device=factors[0][0].device)
    for re, im in factors:
        mag_sq = re.pow(2) + im.pow(2)
        norm = norm + mag_sq.pow(p / 2.0).sum()
    return norm


def evaluate_kge(model: STComplExEmbedding,
                 dataloader: DataLoader,
                 device: torch.device,
                 max_eval_batches: int = 50) -> Dict[str, float]:
    """
    Evaluates MRR and Hits@1, Hits@3, Hits@10 for object entity prediction.
    """
    model.eval()
    ranks = []
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            s_ids, r_ids, o_ids, t_ids, l_ids = [x.to(device) for x in batch]

            s_re, s_im = model.ent_re(s_ids), model.ent_im(s_ids)
            r_re, r_im = model.rel_re(r_ids), model.rel_im(r_ids)
            t_re, t_im = model.time_re(t_ids), model.time_im(t_ids)
            l_re, l_im = model.loc_re(l_ids), model.loc_im(l_ids)

            scores = model.score_fact(s_re, s_im, r_re, r_im, t_re, t_im, l_re, l_im)  # (B, N)
            target_scores = scores.gather(1, o_ids.unsqueeze(1))  # (B, 1)

            # Rank is number of entities with score >= target score
            batch_ranks = (scores >= target_scores).sum(dim=1).float()
            ranks.extend(batch_ranks.cpu().tolist())

            count += 1
            if count >= max_eval_batches:
                break

    if not ranks:
        return {"mrr": 0.0, "hits@1": 0.0, "hits@10": 0.0}

    ranks_tensor = torch.tensor(ranks)
    mrr = (1.0 / ranks_tensor).mean().item()
    hits1 = (ranks_tensor <= 1.0).float().mean().item()
    hits3 = (ranks_tensor <= 3.0).float().mean().item()
    hits10 = (ranks_tensor <= 10.0).float().mean().item()

    return {
        "mrr": round(mrr * 100, 2),
        "hits@1": round(hits1 * 100, 2),
        "hits@3": round(hits3 * 100, 2),
        "hits@10": round(hits10 * 100, 2)
    }


def pretrain_stkg(
    stkg_dir: str = "data/stkg",
    output_dir: str = "experiments/checkpoints/stkg",
    epochs: int = 50,
    batch_size: int = 1000,
    lr: float = 0.1,
    reg_weight: float = 1e-2,
    embedding_dim: int = 512,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_samples: Optional[int] = None,
    seed: int = 42
):
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_str)
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== STKG Pre-training (ST-TComplEx) on {device} ===")
    print(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={lr}, reg_weight={reg_weight}, dim={embedding_dim}")

    # 1. Load Vocabularies
    with open(os.path.join(stkg_dir, "entity2id.json"), "r", encoding="utf-8") as f:
        entity2id = json.load(f)
    with open(os.path.join(stkg_dir, "relation2id.json"), "r", encoding="utf-8") as f:
        relation2id = json.load(f)
    with open(os.path.join(stkg_dir, "time2id.json"), "r", encoding="utf-8") as f:
        time2id = json.load(f)
    with open(os.path.join(stkg_dir, "coord2id.json"), "r", encoding="utf-8") as f:
        coord2id = json.load(f)
    with open(os.path.join(stkg_dir, "entity_metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 2. Load & Split Facts (8:1:1 split per Section 6.4)
    facts_path = os.path.join(stkg_dir, "facts.tsv")
    all_facts = load_facts(facts_path, entity2id, relation2id, time2id, coord2id, meta)
    random.shuffle(all_facts)

    if max_samples:
        all_facts = all_facts[:max_samples]

    n_total = len(all_facts)
    n_val = max(1, int(0.1 * n_total))
    n_test = max(1, int(0.1 * n_total))
    n_train = n_total - n_val - n_test

    train_facts = all_facts[:n_train]
    val_facts = all_facts[n_train:n_train + n_val]
    test_facts = all_facts[n_train + n_val:]

    print(f"Dataset split: Train={len(train_facts)}, Val={len(val_facts)}, Test={len(test_facts)}")

    train_loader = DataLoader(STKGFactsDataset(train_facts), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(STKGFactsDataset(val_facts), batch_size=batch_size, shuffle=False)

    # 3. Instantiate ST-TComplEx Model
    model = STComplExEmbedding(
        num_entities=len(entity2id),
        num_relations=len(relation2id),
        num_timestamps=len(time2id),
        num_locations=len(coord2id),
        embedding_dim=embedding_dim
    ).to(device)

    # Set neutral identity for <NONE> (index 0) in time and loc
    with torch.no_grad():
        model.time_re.weight[0].fill_(1.0)
        model.time_im.weight[0].fill_(0.0)
        model.loc_re.weight[0].fill_(1.0)
        model.loc_im.weight[0].fill_(0.0)

    # 4. Optimizer: Adagrad lr=0.1 per paper Appendix B & Lacroix et al.
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)

    best_val_mrr = -1.0
    best_checkpoint_path = os.path.join(output_dir, "stkg_pretrained_50ep.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            s_ids, r_ids, o_ids, t_ids, l_ids = [x.to(device) for x in batch]

            s_re, s_im = model.ent_re(s_ids), model.ent_im(s_ids)
            r_re, r_im = model.rel_re(r_ids), model.rel_im(r_ids)
            t_re, t_im = model.time_re(t_ids), model.time_im(t_ids)
            l_re, l_im = model.loc_re(l_ids), model.loc_im(l_ids)
            o_re, o_im = model.ent_re(o_ids), model.ent_im(o_ids)

            # Score against all candidate entities (multiclass cross-entropy)
            scores = model.score_fact(s_re, s_im, r_re, r_im, t_re, t_im, l_re, l_im)
            ce_loss = F.cross_entropy(scores, o_ids)

            # N3 Regularization
            factors = [(s_re, s_im), (r_re, r_im), (t_re, t_im), (l_re, l_im), (o_re, o_im)]
            reg_loss = compute_n3_reg(factors, p=3.0) / s_ids.size(0)

            loss = ce_loss + reg_weight * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Enforce <NONE> neutral constraints
            with torch.no_grad():
                model.time_re.weight[0].fill_(1.0)
                model.time_im.weight[0].fill_(0.0)
                model.loc_re.weight[0].fill_(1.0)
                model.loc_im.weight[0].fill_(0.0)

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(1, n_batches)

        # Validation evaluation
        if epoch % 5 == 0 or epoch == epochs or max_samples is not None:
            metrics = evaluate_kge(model, val_loader, device)
            print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Val MRR: {metrics['mrr']}% | Hits@1: {metrics['hits@1']}% | Hits@10: {metrics['hits@10']}%")

            if metrics["mrr"] > best_val_mrr or epoch == 1:
                best_val_mrr = metrics["mrr"]
                torch.save({
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_metrics": metrics,
                    "config": {
                        "embedding_dim": embedding_dim,
                        "num_entities": len(entity2id),
                        "num_relations": len(relation2id),
                        "num_timestamps": len(time2id),
                        "num_locations": len(coord2id),
                        "lr": lr,
                        "batch_size": batch_size
                    }
                }, best_checkpoint_path)
        else:
            print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f}")

    print(f"\nPre-training completed. Best checkpoint saved to {best_checkpoint_path}")
    return best_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STKG Pre-training (ST-TComplEx)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (paper: 50)")
    parser.add_argument("--batch_size", type=int, default=1000, help="Training batch size (paper: 1000)")
    parser.add_argument("--lr", type=float, default=0.1, help="Adagrad learning rate (paper: 0.1)")
    parser.add_argument("--reg_weight", type=float, default=1e-2, help="N3 regularization weight")
    parser.add_argument("--embedding_dim", type=int, default=512, help="Embedding dimension D (paper: 512)")
    parser.add_argument("--output_dir", type=str, default="experiments/checkpoints/stkg")
    parser.add_argument("--stkg_dir", type=str, default="data/stkg")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of facts for rapid dry-run test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    pretrain_stkg(
        stkg_dir=args.stkg_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        reg_weight=args.reg_weight,
        embedding_dim=args.embedding_dim,
        device_str=args.device,
        max_samples=args.max_samples
    )
