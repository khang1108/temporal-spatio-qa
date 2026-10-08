"""Comprehensive evaluation metrics for SpatialMQA benchmark (ACL 2025).

Calculates:
1. Overall Metrics: Accuracy, Precision, Recall, Macro-F1
2. 3D Coordinate Axis Metrics:
   - A_x: Horizontal axis ("left of", "right of")
   - A_y: Depth axis ("in front of", "behind")
   - A_z: Vertical axis ("on/above", "below")
3. Perspective Rule Metrics:
   - Rule 1 / Q1: Out-of-image perspective (observer outside scene)
   - Rule 2 / Q2: In-image first-person perspective (observer is target object)
   - Rule 3 / Q3: In-image third-person perspective (observer is 3rd entity)
"""

import json
from collections import defaultdict
from typing import Dict, List, Any
import numpy as np


RELATIONS = ["left of", "right of", "in front of", "behind", "on/above", "below"]

AXIS_MAPPING = {
    "left of": "A_x (Horizontal)",
    "right of": "A_x (Horizontal)",
    "in front of": "A_y (Depth)",
    "behind": "A_y (Depth)",
    "on/above": "A_z (Vertical)",
    "below": "A_z (Vertical)"
}


def normalize_spatial_relation(output_text: str) -> str:
    """Normalize model output text into standard SpatialMQA relation labels per official code."""
    t = output_text.strip().lower()
    if "in front of" in t or "front" in t:
        return "in front of"
    elif "behind" in t:
        return "behind"
    elif "left of" in t or "left" in t:
        return "left of"
    elif "right of" in t or "right" in t:
        return "right of"
    elif "on/above" in t or "above" in t or "on" in t:
        return "on/above"
    elif "below" in t or "under" in t:
        return "below"
    return ""


def classify_perspective(question: str) -> str:
    ql = question.lower()
    if "from your perspective" in ql or "from the perspective" in ql:
        return "Q3_ThirdPerson"
    elif "if you" in ql:
        return "Q2_FirstPerson"
    else:
        return "Q1_OutOfImage"


def evaluate_predictions(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate predictions against ground truth labels.

    Each prediction dict should have:
    - 'answer' (str): ground truth relation
    - 'output' or 'prediction' (str): model generated text
    - 'question' (str, optional): question text for perspective breakdown
    """
    total = len(predictions)
    if total == 0:
        return {}

    correct_total = 0
    
    # Class-wise counters for Precision, Recall, F1
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)

    # Axis counters
    axis_correct = defaultdict(int)
    axis_total = defaultdict(int)

    # Perspective counters
    persp_correct = defaultdict(int)
    persp_total = defaultdict(int)

    for item in predictions:
        gt = item["answer"].strip().lower()
        pred_raw = str(item.get("output", item.get("prediction", ""))).strip().lower()

        # Matching heuristic per official paper code and normalization
        norm_rel = normalize_spatial_relation(pred_raw)
        pred_rel = norm_rel if norm_rel in RELATIONS else "unknown"

        is_correct = (
            (gt == pred_rel) or
            (gt in pred_raw) or
            (pred_raw in gt and len(pred_raw) > 0)
        )

        if is_correct:
            correct_total += 1
            tp[gt] += 1
        else:
            fn[gt] += 1
            if pred_rel in RELATIONS:
                fp[pred_rel] += 1

        support[gt] += 1

        # 3D Axis
        axis = AXIS_MAPPING.get(gt, "Other")
        axis_total[axis] += 1
        if is_correct:
            axis_correct[axis] += 1

        # Perspective
        q_text = item.get("question", "")
        persp = classify_perspective(q_text)
        persp_total[persp] += 1
        if is_correct:
            persp_correct[persp] += 1

    # Macro Precision, Recall, F1
    precisions = []
    recalls = []
    f1s = []

    for r in RELATIONS:
        p = tp[r] / (tp[r] + fp[r]) if (tp[r] + fp[r]) > 0 else 0.0
        rec = tp[r] / (tp[r] + fn[r]) if (tp[r] + fn[r]) > 0 else 0.0
        f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
        precisions.append(p)
        recalls.append(rec)
        f1s.append(f1)

    overall_acc = (correct_total / total) * 100
    macro_p = np.mean(precisions) * 100
    macro_r = np.mean(recalls) * 100
    macro_f1 = np.mean(f1s) * 100

    results = {
        "overall": {
            "total_samples": total,
            "correct_samples": correct_total,
            "accuracy": round(overall_acc, 2),
            "precision": round(macro_p, 2),
            "recall": round(macro_r, 2),
            "f1": round(macro_f1, 2),
        },
        "by_axis": {
            "A_x": round((axis_correct["A_x (Horizontal)"] / axis_total["A_x (Horizontal)"]) * 100, 2) if axis_total["A_x (Horizontal)"] else 0.0,
            "A_y": round((axis_correct["A_y (Depth)"] / axis_total["A_y (Depth)"]) * 100, 2) if axis_total["A_y (Depth)"] else 0.0,
            "A_z": round((axis_correct["A_z (Vertical)"] / axis_total["A_z (Vertical)"]) * 100, 2) if axis_total["A_z (Vertical)"] else 0.0,
        },
        "by_perspective": {
            "Q1_OutOfImage": round((persp_correct["Q1_OutOfImage"] / persp_total["Q1_OutOfImage"]) * 100, 2) if persp_total["Q1_OutOfImage"] else 0.0,
            "Q2_FirstPerson": round((persp_correct["Q2_FirstPerson"] / persp_total["Q2_FirstPerson"]) * 100, 2) if persp_total["Q2_FirstPerson"] else 0.0,
            "Q3_ThirdPerson": round((persp_correct["Q3_ThirdPerson"] / persp_total["Q3_ThirdPerson"]) * 100, 2) if persp_total["Q3_ThirdPerson"] else 0.0,
        },
        "counts": {
            "axis_total": dict(axis_total),
            "persp_total": dict(persp_total)
        }
    }

    return results


PAPER_BASELINES = {
    "LLaVA-1.5-7B (Zero-Shot)": {
        "accuracy": 29.28,
        "precision": 30.72,
        "recall": 31.18,
        "f1": 30.95,
        "A_x": None,
        "A_y": None,
        "A_z": None,
        "Q1": None,
        "Q2": None,
        "Q3": None,
    },
    "LLaVA-1.5-7B (LoRA)": {
        "accuracy": 46.85,  # 46.56% in group breakdown
        "precision": 46.10,
        "recall": 44.56,
        "f1": 45.32,
        "A_x": 55.71,
        "A_y": 29.64,
        "A_z": 48.13,
        "Q1": 53.14,
        "Q2": 40.99,
        "Q3": 64.71,
    },
    "SpaceLLaVA (LoRA)": {
        "accuracy": 48.14,
        "precision": 47.96,
        "recall": 46.18,
        "f1": 47.05,
        "A_x": 56.00,
        "A_y": 51.85,
        "A_z": 31.41,
        "Q1": 54.87,
        "Q2": 42.37,
        "Q3": 58.82,
    }
}


def print_evaluation_report(results: Dict[str, Any], model_name: str = "Model"):
    ov = results["overall"]
    ax = results["by_axis"]
    ps = results["by_perspective"]
    paper_ref = PAPER_BASELINES["LLaVA-1.5-7B (LoRA)"]

    def delta_str(val, ref):
        if ref is None:
            return "--"
        diff = val - ref
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff:.2f}%"

    print("=" * 80)
    print(f"  SPATIALMQA BENCHMARK EVALUATION REPORT (ACL 2025 Long): {model_name}")
    print("=" * 80)
    print(f"Total Test Samples: {ov['total_samples']} | Correct: {ov['correct_samples']}")
    print("-" * 80)
    print(f"{'Metric':<30} | {'Current Run':<12} | {'Paper (LoRA)':<14} | {'Delta vs Paper':<14}")
    print("-" * 80)
    print(f"{'Overall Accuracy':<30} | {ov['accuracy']:>10.2f}% | {paper_ref['accuracy']:>12.2f}% | {delta_str(ov['accuracy'], paper_ref['accuracy']):>14}")
    print(f"{'Macro-Precision':<30} | {ov['precision']:>10.2f}% | {paper_ref['precision']:>12.2f}% | {delta_str(ov['precision'], paper_ref['precision']):>14}")
    print(f"{'Macro-Recall':<30} | {ov['recall']:>10.2f}% | {paper_ref['recall']:>12.2f}% | {delta_str(ov['recall'], paper_ref['recall']):>14}")
    print(f"{'Macro-F1':<30} | {ov['f1']:>10.2f}% | {paper_ref['f1']:>12.2f}% | {delta_str(ov['f1'], paper_ref['f1']):>14}")
    print("-" * 80)
    print("3D Spatial Axes Accuracy:")
    print(f"  {'A_x (Horizontal: left/right)':<28} | {ax['A_x']:>10.2f}% | {paper_ref['A_x']:>12.2f}% | {delta_str(ax['A_x'], paper_ref['A_x']):>14}")
    print(f"  {'A_y (Depth: front/behind)':<28} | {ax['A_y']:>10.2f}% | {paper_ref['A_y']:>12.2f}% | {delta_str(ax['A_y'], paper_ref['A_y']):>14}")
    print(f"  {'A_z (Vertical: above/below)':<28} | {ax['A_z']:>10.2f}% | {paper_ref['A_z']:>12.2f}% | {delta_str(ax['A_z'], paper_ref['A_z']):>14}")
    print("-" * 80)
    print("Perspective Rule Accuracy:")
    print(f"  {'Q1 (Rule 1: Out-of-image)':<28} | {ps['Q1_OutOfImage']:>10.2f}% | {paper_ref['Q1']:>12.2f}% | {delta_str(ps['Q1_OutOfImage'], paper_ref['Q1']):>14}")
    print(f"  {'Q2 (Rule 2: First-person)':<28} | {ps['Q2_FirstPerson']:>10.2f}% | {paper_ref['Q2']:>12.2f}% | {delta_str(ps['Q2_FirstPerson'], paper_ref['Q2']):>14}")
    print(f"  {'Q3 (Rule 3: Third-person)':<28} | {ps['Q3_ThirdPerson']:>10.2f}% | {paper_ref['Q3']:>12.2f}% | {delta_str(ps['Q3_ThirdPerson'], paper_ref['Q3']):>14}")
    print("=" * 80)

