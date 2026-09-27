"""
Evaluation metric computations for STKGQA benchmark.
Computes Hits@1, Hits@3, and Hits@10 overall and broken down by constraint types:
- DTC: Double Timestamp Constraint
- STC: Single Timestamp Constraint
- DDC: Double Direction Constraint
- SDC: Single Direction Constraint
- DC: Distance Constraint
"""

from typing import List, Dict, Any, Union, Set
from collections import defaultdict


def classify_question_constraints(question: str) -> Dict[str, str]:
    """
    Classifies question constraints into temporal and spatial categories
    matching the STQAD paper taxonomy:
    - temporal: 'DTC' (during, while) or 'STC' (before, after, later than, posterior to)
    - spatial: 'DC' (within X miles), 'DDC' (northeast, etc.), 'SDC' (north, etc.)
    """
    q = question.lower()

    # Temporal classification
    if "during" in q or "while" in q:
        temporal_type = "DTC"
    else:
        temporal_type = "STC"

    # Spatial classification
    ddc_keywords = ["northeast", "northwest", "southeast", "southwest"]
    sdc_keywords = ["north", "south", "east", "west"]

    if "within" in q and "mile" in q:
        spatial_type = "DC"
    elif any(kw in q for kw in ddc_keywords):
        spatial_type = "DDC"
    elif any(kw in q for kw in sdc_keywords):
        spatial_type = "SDC"
    else:
        spatial_type = "UNKNOWN"

    return {"temporal": temporal_type, "spatial": spatial_type}


def compute_hits_at_k(predictions: List[List[str]],
                      ground_truths: List[Union[str, List[str]]],
                      k: int) -> float:
    """
    Computes Hits@k.
    A prediction hits if at least one ground-truth entity appears in the top-k ranked predictions.
    """
    if not predictions:
        return 0.0

    hits = 0
    for preds, gts in zip(predictions, ground_truths):
        target_set: Set[str] = set([gts]) if isinstance(gts, str) else set(gts)
        top_k = preds[:k]
        if any(p in target_set for p in top_k):
            hits += 1

    return (hits / len(predictions)) * 100.0


def evaluate_benchmark(predictions: List[List[str]],
                       dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Full evaluation breakdown matching Table 5 of Dai et al. (KBS 2025).
    dataset items contain: 'question', 'answers' (or 'answers' list).
    predictions: list of top-K predicted entity strings for each sample.
    """
    assert len(predictions) == len(dataset), f"Length mismatch: {len(predictions)} vs {len(dataset)}"

    # Buckets for breakdown
    buckets: Dict[str, Dict[str, List[Any]]] = {
        "Overall": {"preds": [], "gts": []},
        "DTC": {"preds": [], "gts": []},
        "STC": {"preds": [], "gts": []},
        "DDC": {"preds": [], "gts": []},
        "SDC": {"preds": [], "gts": []},
        "DC": {"preds": [], "gts": []},
    }

    for pred, item in zip(predictions, dataset):
        gt = item["answers"]
        q = item.get("question", "")
        constraints = classify_question_constraints(q)

        # Overall
        buckets["Overall"]["preds"].append(pred)
        buckets["Overall"]["gts"].append(gt)

        # Temporal breakdown
        t_type = constraints["temporal"]
        if t_type in buckets:
            buckets[t_type]["preds"].append(pred)
            buckets[t_type]["gts"].append(gt)

        # Spatial breakdown
        s_type = constraints["spatial"]
        if s_type in buckets:
            buckets[s_type]["preds"].append(pred)
            buckets[s_type]["gts"].append(gt)

    results = {}
    for cat, data in buckets.items():
        if len(data["preds"]) > 0:
            h1 = compute_hits_at_k(data["preds"], data["gts"], k=1)
            h3 = compute_hits_at_k(data["preds"], data["gts"], k=3)
            h10 = compute_hits_at_k(data["preds"], data["gts"], k=10)
            results[cat] = {
                "count": len(data["preds"]),
                "Hits@1": round(h1, 2),
                "Hits@3": round(h3, 2),
                "Hits@10": round(h10, 2),
            }
        else:
            results[cat] = {"count": 0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}

    return results


def format_table5_markdown(results: Dict[str, Any], model_name: str = "Model") -> str:
    """
    Formats results as a Markdown table matching Table 5 structure.
    """
    headers = ["Model", "Metric", "Overall", "DTC", "STC", "DDC", "SDC", "DC"]
    row_h1 = [
        model_name,
        "Hits@1",
        f"{results['Overall']['Hits@1']:.2f}%",
        f"{results.get('DTC', {}).get('Hits@1', 0.0):.2f}%",
        f"{results.get('STC', {}).get('Hits@1', 0.0):.2f}%",
        f"{results.get('DDC', {}).get('Hits@1', 0.0):.2f}%",
        f"{results.get('SDC', {}).get('Hits@1', 0.0):.2f}%",
        f"{results.get('DC', {}).get('Hits@1', 0.0):.2f}%",
    ]
    row_h10 = [
        model_name,
        "Hits@10",
        f"{results['Overall']['Hits@10']:.2f}%",
        f"{results.get('DTC', {}).get('Hits@10', 0.0):.2f}%",
        f"{results.get('STC', {}).get('Hits@10', 0.0):.2f}%",
        f"{results.get('DDC', {}).get('Hits@10', 0.0):.2f}%",
        f"{results.get('SDC', {}).get('Hits@10', 0.0):.2f}%",
        f"{results.get('DC', {}).get('Hits@10', 0.0):.2f}%",
    ]

    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join([":---"] * 2 + [":---:"] * (len(headers) - 2)) + " |"
    h1_line = "| " + " | ".join(row_h1) + " |"
    h10_line = "| " + " | ".join(row_h10) + " |"

    return f"\n{header_line}\n{sep_line}\n{h1_line}\n{h10_line}\n"
