"""
Vocabulary extraction and canonical mapping builder for STKG and STQAD.
Aligns:
- 15,403 entities (first 5,897 preserve exact STQAD entity2id mapping)
- 32 relations (first 11 preserve STQAD relation2id mapping)
- Discrete timestamps (years) with index 0 reserved for <NONE>
- Discrete spatial coordinates (lat, lon) with index 0 reserved for <NONE>

Outputs saved to: data/stkg/
- entity2id.json
- relation2id.json
- time2id.json
- coord2id.json
"""

import os
import json
import argparse
from typing import Dict, Tuple, Set


def build_stkg_vocabularies(
    stqad_dir: str = "data/stqad",
    stkg_dir: str = "data/stkg"
) -> Dict[str, int]:
    print("Building canonical STKG vocabularies...")

    # 1. Load base STQAD vocabularies
    stqad_ent_path = os.path.join(stqad_dir, "entity2id.json")
    stqad_rel_path = os.path.join(stqad_dir, "relation2id.json")
    with open(stqad_ent_path, "r", encoding="utf-8") as f:
        stqad_ent2id = json.load(f)
    with open(stqad_rel_path, "r", encoding="utf-8") as f:
        stqad_rel2id = json.load(f)

    # 2. Load entity metadata (coords & intervals)
    meta_path = os.path.join(stkg_dir, "entity_metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 3. Expand entities and relations from facts.tsv while preserving STQAD order
    entity2id = dict(stqad_ent2id)
    relation2id = dict(stqad_rel2id)
    raw_timestamps: Set[str] = set()
    loc_entities: Set[str] = set()

    facts_path = os.path.join(stkg_dir, "facts.tsv")
    with open(facts_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            s, r, o, ts, te, loc = parts[:6]
            if s not in entity2id:
                entity2id[s] = len(entity2id)
            if o not in entity2id:
                entity2id[o] = len(entity2id)
            if r not in relation2id:
                relation2id[r] = len(relation2id)
            if ts != "NONE":
                raw_timestamps.add(ts)
            if te != "NONE":
                raw_timestamps.add(te)
            if loc != "NONE":
                loc_entities.add(loc)

    # Include intervals from metadata into timestamps
    for e, m in meta.items():
        inv = m.get("interval")
        if inv:
            raw_timestamps.add(str(inv[0]))
            raw_timestamps.add(str(inv[1]))

    # Sort timestamps numerically where possible
    def time_sort_key(t: str):
        try:
            return (0, int(t))
        except ValueError:
            return (1, t)

    sorted_times = sorted(raw_timestamps, key=time_sort_key)
    time2id = {"<NONE>": 0}
    for t in sorted_times:
        time2id[t] = len(time2id)

    # Build unique coordinate vocabulary
    unique_coords: Set[str] = set()
    for e, m in meta.items():
        if m.get("coords"):
            lat, lon = m["coords"]
            unique_coords.add(f"{lat:.4f},{lon:.4f}")

    sorted_coords = sorted(unique_coords)
    coord2id = {"<NONE>": 0}
    for c in sorted_coords:
        coord2id[c] = len(coord2id)

    # Save all vocabularies
    os.makedirs(stkg_dir, exist_ok=True)
    with open(os.path.join(stkg_dir, "entity2id.json"), "w", encoding="utf-8") as f:
        json.dump(entity2id, f, indent=2, ensure_ascii=False)
    with open(os.path.join(stkg_dir, "relation2id.json"), "w", encoding="utf-8") as f:
        json.dump(relation2id, f, indent=2, ensure_ascii=False)
    with open(os.path.join(stkg_dir, "time2id.json"), "w", encoding="utf-8") as f:
        json.dump(time2id, f, indent=2, ensure_ascii=False)
    with open(os.path.join(stkg_dir, "coord2id.json"), "w", encoding="utf-8") as f:
        json.dump(coord2id, f, indent=2, ensure_ascii=False)

    summary = {
        "num_entities": len(entity2id),
        "num_stqad_entities": len(stqad_ent2id),
        "num_relations": len(relation2id),
        "num_timestamps": len(time2id),
        "num_coordinates": len(coord2id)
    }
    print(f"Vocabularies created successfully:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build canonical vocabularies for STKG")
    parser.add_argument("--stqad_dir", type=str, default="data/stqad")
    parser.add_argument("--stkg_dir", type=str, default="data/stkg")
    args = parser.parse_args()

    build_stkg_vocabularies(stqad_dir=args.stqad_dir, stkg_dir=args.stkg_dir)
