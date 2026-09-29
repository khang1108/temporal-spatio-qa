#!/usr/bin/env python3
"""
STKG Construction and Entity Metadata Resolution.
Based on Section 4.1 & Appendix B of:
    Dai et al., "Question answering over spatio-temporal knowledge graph",
    Knowledge-Based Systems 329 (2025) 114314.

Pipeline:
1. Parse YAGO15k splits (138,056 facts) to extract temporal intervals:
   (s, r, o, occursSince, t_start, occursUntil, t_end)
2. Identify all key spatial entities needed for STQAD reasoning:
   - All spatial clues and target answers from STQAD QA pairs (~1,273 entities)
   - Parent locations via <isLocatedIn> hierarchy (~300 entities)
3. Fetch ground-truth geographic coordinates (lat, lon) using Wikipedia API:
   - Optimized batching with `colimit=500` and `redirects=1`
   - Multi-hop hierarchical inheritance via <isLocatedIn>
4. Compile consolidated datasets:
   - `data/stkg/entity_metadata.json`: for ConstraintFilter (coords & temporal intervals)
   - `data/stkg/facts.tsv`: for STComplEx pre-training (138,056 facts, 50 epochs)
   - `data/stkg/stats.json`: verification statistics matching paper Table 2
"""

import os
import re
import json
import time
import urllib.request
import urllib.parse
from collections import defaultdict
from typing import Dict, Any, Tuple, Optional, Set, List


def clean_entity_str(raw: str) -> str:
    """Cleans punctuation artifacts from entity strings (e.g. '<Quebec_City>,' -> '<Quebec_City>')."""
    s = raw.strip()
    if s.endswith(","):
        s = s[:-1].strip()
    return s


def parse_year(val: str) -> Optional[int]:
    """Extracts integer year from YAGO temporal literals like '\"1998-##-##\"'."""
    clean = val.strip().strip('"')
    parts = clean.split("-")
    if parts:
        try:
            return int(parts[0])
        except ValueError:
            pass
    return None


def fetch_wikipedia_coordinates(raw_entities: List[str], batch_size: int = 50) -> Dict[str, Tuple[float, float]]:
    """
    Fetches latitude and longitude from Wikipedia API for a list of entity strings.
    Uses `colimit=500` to prevent Wikipedia from truncating to only 10 coordinates per batch.
    Handles redirects and normalization correctly.
    """
    coords_map = {}
    total = len(raw_entities)
    num_batches = (total + batch_size - 1) // batch_size
    print(f"  Fetching Wikipedia coordinates for {total:,} entities ({num_batches} batches)...", flush=True)

    for b_idx in range(num_batches):
        batch = raw_entities[b_idx * batch_size : (b_idx + 1) * batch_size]
        
        # Build mapping from title variations to standard '<Entity>' format
        clean_batch = []
        variation_to_orig = {}
        for ent in batch:
            c = ent.strip("<>")
            clean_batch.append(c)
            variation_to_orig[c] = ent
            variation_to_orig[c.replace("_", " ")] = ent

        titles_str = "|".join(clean_batch)
        url = (
            f"https://en.wikipedia.org/w/api.php?action=query&prop=coordinates"
            f"&colimit=500&redirects=1&titles={urllib.parse.quote(titles_str)}&format=json"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "STKGQA-Research-Bot/2.0 (academic-reproduction@university.edu)"}
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                query_info = data.get("query", {})
                
                # Trace redirects
                redirect_map = {r["to"]: r["from"] for r in query_info.get("redirects", [])}
                # Trace normalizations
                norm_map = {n["to"]: n["from"] for n in query_info.get("normalized", [])}

                pages = query_info.get("pages", {})
                batch_found = 0
                for pid, p in pages.items():
                    if "coordinates" in p and p["coordinates"]:
                        c = p["coordinates"][0]
                        lat, lon = round(float(c["lat"]), 4), round(float(c["lon"]), 4)
                        p_title = p.get("title", "")
                        
                        resolved_title = p_title
                        if resolved_title in redirect_map:
                            resolved_title = redirect_map[resolved_title]
                        if resolved_title in norm_map:
                            resolved_title = norm_map[resolved_title]

                        orig_ent = (
                            variation_to_orig.get(p_title)
                            or variation_to_orig.get(p_title.replace(" ", "_"))
                            or variation_to_orig.get(resolved_title)
                            or variation_to_orig.get(resolved_title.replace(" ", "_"))
                            or f"<{resolved_title.replace(' ', '_')}>"
                        )
                        coords_map[orig_ent] = (lat, lon)
                        batch_found += 1
                        
            print(f"    Batch {b_idx + 1:2d}/{num_batches:2d}: resolved {batch_found:2d} coordinates (total: {len(coords_map):,})", flush=True)
        except Exception as e:
            print(f"    Batch {b_idx + 1:2d}/{num_batches:2d}: Warning - request error: {e}", flush=True)

        time.sleep(0.08)

    return coords_map


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_yago_dir = os.path.join(root_dir, "data", "raw_yago15k")
    stqad_dir = os.path.join(root_dir, "data", "stqad")
    out_dir = os.path.join(root_dir, "data", "stkg")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70, flush=True)
    print("STEP 1: Constructing Spatio-Temporal Knowledge Graph (STKG)", flush=True)
    print(f"Output directory: {out_dir}", flush=True)
    print("=" * 70, flush=True)

    # -------------------------------------------------------------------------
    # 1. Parse YAGO15k Facts and Temporal Metadata
    # -------------------------------------------------------------------------
    print("\n[Phase 1] Parsing raw YAGO15k facts and temporal annotations...", flush=True)
    
    fact_temporals = defaultdict(lambda: {"since": [], "until": []})
    all_raw_lines = []
    unique_entities = set()
    unique_relations = set()
    located_in_graph = {}  # child -> parent location
    entity_years = defaultdict(lambda: {"starts": [], "ends": []})

    yago_files = [
        os.path.join(raw_yago_dir, "train"),
        os.path.join(raw_yago_dir, "valid"),
        os.path.join(raw_yago_dir, "test"),
    ]

    for fpath in yago_files:
        assert os.path.exists(fpath), f"File not found: {fpath}"
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if not parts or len(parts) < 3:
                    continue
                s = clean_entity_str(parts[0])
                r = parts[1]
                o = clean_entity_str(parts[2])
                all_raw_lines.append((s, r, o))
                unique_entities.add(s)
                unique_entities.add(o)
                unique_relations.add(r)

                if r == "<isLocatedIn>":
                    located_in_graph[s] = o

                if len(parts) == 5:
                    temp_rel, temp_val = parts[3], parts[4]
                    y = parse_year(temp_val)
                    if y is not None:
                        if temp_rel == "<occursSince>":
                            fact_temporals[(s, r, o)]["since"].append(y)
                            entity_years[s]["starts"].append(y)
                            entity_years[o]["starts"].append(y)
                        elif temp_rel == "<occursUntil>":
                            fact_temporals[(s, r, o)]["until"].append(y)
                            entity_years[s]["ends"].append(y)
                            entity_years[o]["ends"].append(y)

    print(f"  Total raw lines processed: {len(all_raw_lines):,}", flush=True)
    print(f"  Unique entities: {len(unique_entities):,}", flush=True)
    print(f"  Unique relations: {len(unique_relations):,}", flush=True)
    print(f"  Entities with temporal data: {len(entity_years):,}", flush=True)
    print(f"  <isLocatedIn> hierarchical links: {len(located_in_graph):,}", flush=True)

    # -------------------------------------------------------------------------
    # 2. Extract Spatial Clues and Target Answers from STQAD Questions
    # -------------------------------------------------------------------------
    print("\n[Phase 2] Extracting spatial & temporal clues from STQAD QA pairs...", flush=True)
    
    clue_geos = set()
    clue_times = set()
    target_answers = set()
    
    direction_words = [
        "northeast", "northwest", "southeast", "southwest",
        "north", "south", "east", "west"
    ]
    time_words = [
        "before", "after", "during", "while", "posterior to",
        "later than", "prior to"
    ]

    for split in ["train", "val", "test"]:
        path = os.path.join(stqad_dir, f"{split}_datas.json")
        with open(path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        for s in samples:
            q = s.get("question", "")
            raw_ans = s.get("answers", "")
            ans_list = raw_ans if isinstance(raw_ans, list) else [raw_ans]
            for a in ans_list:
                target_answers.add(clean_entity_str(a))

            # Distance clues
            m_dist = re.search(r"within\s+\d+(?:\.\d+)?\s+miles of\s+(<[^>]+>)", q)
            if m_dist:
                clue_geos.add(clean_entity_str(m_dist.group(1)))

            # Directional clues
            for d in direction_words:
                m_dir = re.search(rf"\b{d}\s+of\s+(?:the\s+)?(<[^>]+>)", q, re.IGNORECASE)
                if m_dir:
                    clue_geos.add(clean_entity_str(m_dir.group(1)))

            # Temporal clues
            for t in time_words:
                m_t = re.search(
                    rf"\b{t}\s+(?:the\s+)?(?:termination of\s+|cessation of\s+|dissolution of\s+|it\s+<[^>]+>\s+)?(<[^>]+>)",
                    q, re.IGNORECASE
                )
                if m_t:
                    clue_times.add(clean_entity_str(m_t.group(1)))

    print(f"  Unique spatial clues extracted: {len(clue_geos):,}", flush=True)
    print(f"  Unique temporal clues extracted: {len(clue_times):,}", flush=True)
    print(f"  Unique target answers: {len(target_answers):,}", flush=True)

    # Focus geocoding on: Clues + Answers + Parent Locations
    target_geos = set(clue_geos).union(target_answers)
    for e in list(target_geos):
        curr = e
        for _ in range(3):
            parent = located_in_graph.get(curr)
            if parent:
                target_geos.add(parent)
                curr = parent
            else:
                break

    print(f"  Total target geographic entities to geocode: {len(target_geos):,}", flush=True)

    # -------------------------------------------------------------------------
    # 3. Geocode Entities via Wikipedia API + colimit=500 + Multi-Hop Inheritance
    # -------------------------------------------------------------------------
    print("\n[Phase 3] Resolving coordinates via Wikipedia API...", flush=True)
    
    geo_list = sorted(list(target_geos))
    direct_coords = fetch_wikipedia_coordinates(geo_list, batch_size=50)
    print(f"  Direct coordinates resolved from Wikipedia: {len(direct_coords):,}", flush=True)

    # Hierarchical multi-hop fallback via <isLocatedIn>
    resolved_coords = dict(direct_coords)
    inherited_count = 0
    for e in geo_list:
        if e not in resolved_coords:
            curr = e
            for _ in range(5):
                parent = located_in_graph.get(curr)
                if parent and parent in resolved_coords:
                    resolved_coords[e] = resolved_coords[parent]
                    inherited_count += 1
                    break
                curr = parent
                if not curr:
                    break

    # Stem-based city matching for clubs (e.g. <Manchester_City_F.C.> -> <Manchester>)
    stem_resolved = 0
    for e in geo_list:
        if e not in resolved_coords:
            clean = e.strip("<>")
            # Match city name before common club suffixes
            stem = re.sub(
                r"_(F\.C\.|A\.F\.C\.|CF|United_F\.C\.|City_F\.C\.|Rovers_F\.C\.|County_F\.C\.|Town_F\.C\.|Wanderers_F\.C\.)",
                "", clean
            )
            candidate = f"<{stem}>"
            if candidate in resolved_coords:
                resolved_coords[e] = resolved_coords[candidate]
                stem_resolved += 1

    # Well-known fallback coordinates for historic states and London clubs
    manual_coords = {
        "<Kazakhstan>": (48.0196, 66.9237),
        "<Mongolia>": (46.8625, 103.8467),
        "<Democratic_Republic_of_the_Congo>": (-4.0383, 21.7587),
        "<Republic_of_the_Congo>": (-0.2280, 15.8277),
        "<Soviet_Union>": (61.5240, 105.3188),
        "<West_Germany>": (50.7374, 7.0982),
        "<Ottoman_Empire>": (41.0082, 28.9784),
        "<Kingdom_of_Italy>": (41.9028, 12.4964),
        "<Kingdom_of_Prussia>": (52.5200, 13.4050),
        "<Kingdom_of_the_Netherlands>": (52.3676, 4.9041),
        "<AFC_Wimbledon>": (51.4311, -0.1883),
        "<Wimbledon_F.C.>": (51.4311, -0.1883),
        "<Barnet_F.C.>": (51.5992, -0.2981),
        "<Brentford_F.C.>": (51.4889, -0.3014),
        "<Charlton_Athletic_F.C.>": (51.4864, 0.0361),
        "<Fulham_F.C.>": (51.4749, -0.2217),
        "<Tottenham_Hotspur_F.C.>": (51.6042, -0.0664),
        "<West_Ham_United_F.C.>": (51.5388, -0.0166),
        "<Wycombe_Wanderers_F.C.>": (51.6287, -0.7482),
        "<Aston_Villa_F.C.>": (52.5092, -1.8847),
        "<Arminia_Bielefeld>": (52.0302, 8.5168),
        "<Derby_County_F.C.>": (52.9150, -1.4472),
        "<Reading_F.C.>": (51.4222, -0.9828),
        "<Stoke_City_F.C.>": (52.9883, -2.1756),
        "<Sunderland_A.F.C.>": (54.9144, -1.3883),
        "<West_Bromwich_Albion_F.C.>": (52.5091, -1.9639),
        "<Wigan_Athletic_F.C.>": (53.5478, -2.6542),
        "<Ayr_United_F.C.>": (55.4597, -4.6192),
        "<Hibernian_F.C.>": (55.9617, -3.1656),
        "<CFR_Cluj>": (46.7797, 23.5786),
        "<FC_Twente>": (52.2366, 6.8378),
        "<FK_Vojvodina>": (45.2444, 19.8458),
        "<Monarcas_Morelia>": (19.7214, -101.2333),
        "<Morocco_national_football_team>": (33.9716, -6.8498),
        "<Northwich_Victoria_F.C.>": (53.2592, -2.5161),
        "<S.C._Olhanense>": (37.0278, -7.8394),
        "<Villarreal_CF>": (39.9442, -0.1039),
        "<Quebec_City>": (46.8139, -71.2080),
    }
    for e, coords in manual_coords.items():
        if e not in resolved_coords:
            resolved_coords[e] = coords

    print(f"  Inherited coordinates via <isLocatedIn>: {inherited_count:,}", flush=True)
    print(f"  Resolved coordinates via club stem matching: {stem_resolved:,}", flush=True)
    print(f"  Total entities with coordinates: {len(resolved_coords):,}", flush=True)

    # -------------------------------------------------------------------------
    # 4. Construct Entity Metadata (Coords + Temporal Interval)
    # -------------------------------------------------------------------------
    print("\n[Phase 4] Compiling `data/stkg/entity_metadata.json`...", flush=True)
    
    entity_meta = {}
    for ent in unique_entities.union(target_geos):
        coords = resolved_coords.get(ent)
        
        y_info = entity_years.get(ent)
        interval = None
        if y_info and (y_info["starts"] or y_info["ends"]):
            all_y = y_info["starts"] + y_info["ends"]
            min_y = min(all_y)
            max_y = max(all_y)
            interval = [min_y, max_y]

        meta_entry = {}
        if coords:
            meta_entry["coords"] = [coords[0], coords[1]]
        if interval:
            meta_entry["interval"] = interval

        if meta_entry:
            entity_meta[ent] = meta_entry

    meta_out_path = os.path.join(out_dir, "entity_metadata.json")
    with open(meta_out_path, "w", encoding="utf-8") as f:
        json.dump(entity_meta, f, indent=2)
    print(f"  Saved entity metadata to: {meta_out_path}", flush=True)
    print(f"  Total entities in metadata: {len(entity_meta):,}", flush=True)

    # Coverage verification
    stqad_geo_coverage = sum(1 for e in clue_geos if e in entity_meta and "coords" in entity_meta[e])
    stqad_ans_coverage = sum(1 for e in target_answers if e in entity_meta and "coords" in entity_meta[e])
    stqad_time_coverage = sum(1 for e in clue_times if e in entity_meta and "interval" in entity_meta[e])

    print(f"  STQAD Spatial Clues Coordinate Coverage: {stqad_geo_coverage}/{len(clue_geos)} ({stqad_geo_coverage/len(clue_geos)*100:.1f}%)", flush=True)
    print(f"  STQAD Answer Entities Coordinate Coverage: {stqad_ans_coverage}/{len(target_answers)} ({stqad_ans_coverage/len(target_answers)*100:.1f}%)", flush=True)
    print(f"  STQAD Temporal Clues Interval Coverage: {stqad_time_coverage}/{len(clue_times)} ({stqad_time_coverage/len(clue_times)*100:.1f}%)", flush=True)

    # -------------------------------------------------------------------------
    # 5. Construct `data/stkg/facts.tsv` for KGE Pre-training
    # -------------------------------------------------------------------------
    print("\n[Phase 5] Compiling `data/stkg/facts.tsv` for STComplEx pre-training...", flush=True)
    
    facts_out_path = os.path.join(out_dir, "facts.tsv")
    written_facts = 0
    spatio_temporal_annotated = 0

    with open(facts_out_path, "w", encoding="utf-8") as f_out:
        for s, r, o in all_raw_lines:
            t_data = fact_temporals.get((s, r, o), {"since": [], "until": []})
            t_start = str(min(t_data["since"])) if t_data["since"] else "NONE"
            t_end = str(max(t_data["until"])) if t_data["until"] else "NONE"
            
            loc_ent = o if o in resolved_coords else "NONE"
            
            if t_start != "NONE" or t_end != "NONE" or loc_ent != "NONE":
                spatio_temporal_annotated += 1

            f_out.write(f"{s}\t{r}\t{o}\t{t_start}\t{t_end}\t{loc_ent}\n")
            written_facts += 1

    print(f"  Saved facts to: {facts_out_path}", flush=True)
    print(f"  Total written facts: {written_facts:,}", flush=True)
    print(f"  Facts with Spatio-Temporal annotation: {spatio_temporal_annotated:,}", flush=True)

    # -------------------------------------------------------------------------
    # 6. Save Stats JSON
    # -------------------------------------------------------------------------
    stats = {
        "total_facts": written_facts,
        "facts_spatio_temporal": spatio_temporal_annotated,
        "unique_entities": len(unique_entities),
        "unique_relations": len(unique_relations),
        "entities_with_coords": len(resolved_coords),
        "entities_with_intervals": len(entity_years),
        "stqad_spatial_clue_coverage_pct": round(stqad_geo_coverage / len(clue_geos) * 100, 2),
        "stqad_answer_coverage_pct": round(stqad_ans_coverage / len(target_answers) * 100, 2),
        "stqad_time_coverage_pct": round(stqad_time_coverage / len(clue_times) * 100, 2),
    }
    stats_path = os.path.join(out_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 70, flush=True)
    print("STKG Construction Complete!", flush=True)
    print(f"Summary: {stats}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
