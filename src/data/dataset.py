"""
Dataset loading and batch processing for STQAD.
Supports raw and paraphrased questions, entity-to-ID mappings, and multi-target label vectors.
"""

import os
import re
import json
from typing import List, Dict, Any, Tuple, Optional
import torch
from torch.utils.data import Dataset


def clean_entity(entity_str: str) -> str:
    """Cleans entity string by trimming whitespace and trailing commas."""
    return entity_str.strip().rstrip(",").strip()


_SPATIAL_PATTERN = re.compile(
    r'\b(northeast|northwest|southeast|southwest|north|south|east|west|within\s+\d+\s+miles)\b',
    re.IGNORECASE
)

_TEMPORAL_PATTERN = re.compile(
    r'\b(later than|posterior to|prior to|cessation|dissolution|termination|founded|during|while|before|after)\b',
    re.IGNORECASE
)


def classify_question_clues(q_text: str,
                            entities: Optional[List[str]] = None,
                            entity_vocab: Optional[Dict[str, int]] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Classifies entities mentioned in a question into their functional spatio-temporal roles:
      - Central Entity [ENT]: Subject of the query around which knowledge subgraphs expand
      - Temporal Clue Entity [TS]: Anchor for temporal intervals (e.g. lifetime/events)
      - Spatial Clue Entity [GEO]: Anchor for geographic coordinates / spatial relationships

    Algorithm follows Section 5.2 of Dai et al. (KBS 2025):
      "The determination of entity types relies on constraint keywords that precede the entities,
       such as before, northeast of, and within 3 miles of."
    """
    q_lower = q_text.lower()

    # =========================================================================
    # STEP 1: Extract Entities Directly from Question Text & Filter Vocabulary
    # =========================================================================
    text_tags = [clean_entity(t) for t in re.findall(r'<[^>]+>', q_text)]
    if entity_vocab:
        cand_ents = [t for t in text_tags if t in entity_vocab]
    else:
        cand_ents = text_tags

    # If question text has no tags or candidate entities are empty, fallback to entities list
    if not cand_ents and entities:
        cand_ents = [clean_entity(e) for e in entities]
    elif entities:
        # Include any provided entities that actually appear in the question text
        for e in entities:
            ce = clean_entity(e)
            if ce.lower().strip("<>") in q_lower and ce not in cand_ents:
                cand_ents.append(ce)

    # =========================================================================
    # STEP 2: Preceding Context Window Extraction & Closest Keyword Matching
    # =========================================================================
    ent_positions = []
    for e in cand_ents:
        clean_e = e.lower().strip("<>")
        pos = q_lower.find(clean_e)
        if pos != -1:
            ent_positions.append((e, clean_e, pos))

    assigned: Dict[str, str] = {}
    for e, clean_e, pos in ent_positions:
        # Check immediate 60-character prefix preceding the entity mention
        prefix = q_lower[max(0, pos - 60):pos]
        s_matches = list(_SPATIAL_PATTERN.finditer(prefix))
        t_matches = list(_TEMPORAL_PATTERN.finditer(prefix))
        s_last = s_matches[-1].end() if s_matches else -1
        t_last = t_matches[-1].end() if t_matches else -1

        # Assign role based on which constraint keyword is closest to the entity
        if s_last > t_last:
            assigned[e] = 'loc'
        elif t_last > s_last:
            assigned[e] = 'time'

    # =========================================================================
    # STEP 3: Assign Functional Roles Based on Classified Constraints
    # =========================================================================
    # Entities preceded by spatial constraint keywords -> loc_clue
    # Entities preceded by temporal constraint keywords -> time_clue
    # Unconstrained entity -> central entity (subject of the question)
    central: Optional[str] = None
    time_clue: Optional[str] = None
    loc_clue: Optional[str] = None

    for e, _, _ in ent_positions:
        role = assigned.get(e)
        if role == 'loc' and loc_clue is None:
            loc_clue = e
        elif role == 'time' and time_clue is None:
            time_clue = e
        elif central is None:
            central = e

    return central, time_clue, loc_clue



def get_vocabularies(data_dir: str = "data/stqad") -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int], Dict[int, str]]:
    """Loads entity and relation vocabularies."""
    entity_path = os.path.join(data_dir, "entity2id.json")
    relation_path = os.path.join(data_dir, "relation2id.json")

    assert os.path.exists(entity_path), f"Entity vocabulary not found at {entity_path}"
    assert os.path.exists(relation_path), f"Relation vocabulary not found at {relation_path}"

    with open(entity_path, "r", encoding="utf-8") as f:
        entity2id = json.load(f)
    with open(relation_path, "r", encoding="utf-8") as f:
        relation2id = json.load(f)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    return entity2id, id2entity, relation2id, id2relation


def load_stqad(split: str, data_dir: str = "data/stqad") -> List[Dict[str, Any]]:
    """
    Loads STQAD json file for split: 'train', 'val', or 'test'.
    """
    file_path = os.path.join(data_dir, f"{split}_datas.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset split file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


class STQADataset(Dataset):
    """
    PyTorch Dataset for STQAD question answering.
    """
    def __init__(self,
                 data: List[Dict[str, Any]],
                 tokenizer,
                 entity2id: Dict[str, int],
                 max_length: int = 128,
                 use_paraphrased: bool = True,
                 cached_cls: Optional[torch.Tensor] = None):
        self.data = data
        self.tokenizer = tokenizer
        self.entity2id = entity2id
        self.num_entities = len(entity2id)
        self.max_length = max_length
        self.use_paraphrased = use_paraphrased
        self.cached_cls = cached_cls

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        # =====================================================================
        # STEP 1: Build Multi-Target Binary Vector & Normalized Distribution
        # =====================================================================
        # Many questions in STQAD have multiple valid answer entities.
        # target_vec: binary float vector of size [num_entities] (for BCE loss)
        # target_dist: normalized probability vector summing to 1.0 (for KL/CE)
        target_vec = torch.zeros(self.num_entities, dtype=torch.float32)
        ans = item.get("answers", [])
        if isinstance(ans, str):
            ans = [ans]

        target_ids = []
        for a in ans:
            cleaned = clean_entity(a)
            if cleaned in self.entity2id:
                eid = self.entity2id[cleaned]
                target_vec[eid] = 1.0
                target_ids.append(eid)

        num_targets = len(target_ids)
        target_dist = target_vec / max(1, num_targets)

        # =====================================================================
        # STEP 2: Extract Mentioned Entity IDs from Question
        # =====================================================================
        question_entity_ids = []
        for e in item.get("entities", []):
            cleaned = clean_entity(e)
            if cleaned in self.entity2id:
                question_entity_ids.append(self.entity2id[cleaned])

        result = {
            "target_vec": target_vec,
            "target_dist": target_dist,
            "target_ids": target_ids,
            "question_entity_ids": question_entity_ids,
            "raw_item": item
        }

        # =====================================================================
        # STEP 3: Encode Question Text or Use Precomputed CLS Embeddings
        # =====================================================================
        # If precomputed RoBERTa CLS tensors exist, bypass tokenizer for fast training;
        # otherwise, tokenize on-the-fly with padding and truncation.
        if self.cached_cls is not None:
            result["cls_rep"] = self.cached_cls[idx]
            result["input_ids"] = torch.zeros(1, dtype=torch.long)
            result["attention_mask"] = torch.zeros(1, dtype=torch.long)
        else:
            q_text = item.get("paraphrased_question", "") if self.use_paraphrased else item.get("question", "")
            if not q_text:
                q_text = item.get("question", "")

            encoded = self.tokenizer(
                q_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            result["input_ids"] = encoded["input_ids"].squeeze(0)
            result["attention_mask"] = encoded["attention_mask"].squeeze(0)

        return result


def collate_stqad_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for DataLoader batching:
      - STEP 1: Stack target binary vectors and probability distributions.
      - STEP 2: Package raw sample dictionaries and entity metadata.
      - STEP 3: Stack token IDs and attention masks (or cached CLS embeddings).
    """
    target_vec = torch.stack([b["target_vec"] for b in batch])
    target_dist = torch.stack([b["target_dist"] for b in batch])
    raw_items = [b["raw_item"] for b in batch]
    question_entity_ids = [b["question_entity_ids"] for b in batch]

    out = {
        "target_vec": target_vec,
        "target_dist": target_dist,
        "question_entity_ids": question_entity_ids,
        "raw_items": raw_items
    }

    if "cls_rep" in batch[0]:
        out["cls_rep"] = torch.stack([b["cls_rep"] for b in batch])
        out["input_ids"] = None
        out["attention_mask"] = None
    else:
        out["input_ids"] = torch.stack([b["input_ids"] for b in batch])
        out["attention_mask"] = torch.stack([b["attention_mask"] for b in batch])
        out["cls_rep"] = None

    return out
