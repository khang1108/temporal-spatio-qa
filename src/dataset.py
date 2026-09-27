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
                 use_paraphrased: bool = True):
        self.data = data
        self.tokenizer = tokenizer
        self.entity2id = entity2id
        self.num_entities = len(entity2id)
        self.max_length = max_length
        self.use_paraphrased = use_paraphrased

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        q_text = item.get("paraphrased_question", "") if self.use_paraphrased else item.get("question", "")
        if not q_text:
            q_text = item.get("question", "")

        # Tokenize question
        encoded = self.tokenizer(
            q_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Multi-target answer vector
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

        # Normalize target_vec for cross-entropy with soft probabilities if needed
        num_targets = len(target_ids)
        target_dist = target_vec / max(1, num_targets)

        # Extract entity IDs mentioned in question
        question_entity_ids = []
        for e in item.get("entities", []):
            cleaned = clean_entity(e)
            if cleaned in self.entity2id:
                question_entity_ids.append(self.entity2id[cleaned])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_vec": target_vec,
            "target_dist": target_dist,
            "target_ids": target_ids,
            "question_entity_ids": question_entity_ids,
            "raw_item": item
        }


def collate_stqad_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for DataLoader."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    target_vec = torch.stack([b["target_vec"] for b in batch])
    target_dist = torch.stack([b["target_dist"] for b in batch])
    raw_items = [b["raw_item"] for b in batch]
    question_entity_ids = [b["question_entity_ids"] for b in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_vec": target_vec,
        "target_dist": target_dist,
        "question_entity_ids": question_entity_ids,
        "raw_items": raw_items
    }
