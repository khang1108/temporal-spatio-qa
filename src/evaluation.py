"""
Evaluation metric computations for STKGQA benchmark.
Computes Hits@1, Hits@3, and Hits@10 overall and broken down by constraint types:
- DTC: Double Timestamp Constraint
- STC: Single Timestamp Constraint
- DDC: Double Direction Constraint
- SDC: Single Direction Constraint
- DC: Distance Constraint
"""

from typing import List, Dict, Any, Union, Set, Tuple
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
                       dataset: List[Dict[str, Any]],
                       return_details: bool = False) -> Union[Dict[str, Any], Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]]:
    """
    Full evaluation breakdown matching Table 5 of Dai et al. (KBS 2025).
    dataset items contain: 'question', 'answers' (or 'answers' list).
    predictions: list of top-K predicted entity strings for each sample.

    Args:
        predictions: Top-K predicted entities for each question
        dataset: List of dataset dicts containing questions and answers
        return_details: If True, also returns (sample_details, failure_summary)

    Returns:
        results: Dictionary of metrics per constraint bucket
        sample_details (optional): Per-sample breakdown with pass/fail and gold rank
        failure_summary (optional): Summary statistics on where models failed
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

    sample_details = []

    for idx, (pred, item) in enumerate(zip(predictions, dataset)):
        gt = item["answers"]
        target_set: Set[str] = set([gt]) if isinstance(gt, str) else set(gt)
        q = item.get("question", "")
        paraphrased_q = item.get("paraphrased_question", "")
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

        # Per-sample rank and hit calculation
        gold_rank = None
        for r, p in enumerate(pred, start=1):
            if p in target_set:
                gold_rank = r
                break

        is_hit_1 = (gold_rank == 1)
        is_hit_3 = (gold_rank is not None and gold_rank <= 3)
        is_hit_10 = (gold_rank is not None and gold_rank <= 10)

        sample_details.append({
            "sample_id": idx,
            "question": q,
            "paraphrased_question": paraphrased_q,
            "gold_answers": list(target_set),
            "top_predictions": pred[:10],
            "gold_rank": gold_rank,
            "is_hit_1": is_hit_1,
            "is_hit_3": is_hit_3,
            "is_hit_10": is_hit_10,
            "status": "PASS_HIT1" if is_hit_1 else ("PASS_HIT10" if is_hit_10 else "FAIL"),
            "temporal_constraint": constraints["temporal"],
            "spatial_constraint": constraints["spatial"],
        })

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

    failure_summary = {
        "total_samples": len(dataset),
        "total_failed_hit1": sum(1 for s in sample_details if not s["is_hit_1"]),
        "total_failed_hit10": sum(1 for s in sample_details if not s["is_hit_10"]),
        "failed_by_temporal": {
            "DTC": sum(1 for s in sample_details if not s["is_hit_10"] and s["temporal_constraint"] == "DTC"),
            "STC": sum(1 for s in sample_details if not s["is_hit_10"] and s["temporal_constraint"] == "STC"),
        },
        "failed_by_spatial": {
            "DC": sum(1 for s in sample_details if not s["is_hit_10"] and s["spatial_constraint"] == "DC"),
            "DDC": sum(1 for s in sample_details if not s["is_hit_10"] and s["spatial_constraint"] == "DDC"),
            "SDC": sum(1 for s in sample_details if not s["is_hit_10"] and s["spatial_constraint"] == "SDC"),
        }
    }

    if return_details:
        return results, sample_details, failure_summary

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
