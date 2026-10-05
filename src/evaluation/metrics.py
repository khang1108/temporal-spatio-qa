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

        # Matching heuristic per paper
        is_correct = (gt in pred_raw) or (pred_raw in gt and len(pred_raw) > 0)
        
        # Exact relation extraction
        pred_rel = None
        for r in RELATIONS:
            if r in pred_raw:
                pred_rel = r
                break
        if pred_rel is None:
            pred_rel = "unknown"

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


def print_evaluation_report(results: Dict[str, Any], model_name: str = "Model"):
    ov = results["overall"]
    ax = results["by_axis"]
    ps = results["by_perspective"]

    print("=" * 65)
    print(f"  SPATIALMQA BENCHMARK EVALUATION REPORT: {model_name}")
    print("=" * 65)
    print(f"Total Samples: {ov['total_samples']} | Correct: {ov['correct_samples']}")
    print(f"  Accuracy:  {ov['accuracy']:.2f}%")
    print(f"  Precision: {ov['precision']:.2f}%")
    print(f"  Recall:    {ov['recall']:.2f}%")
    print(f"  Macro-F1:  {ov['f1']:.2f}%")
    print("-" * 65)
    print("3D Spatial Axes Accuracy:")
    print(f"  A_x (Horizontal: left/right):     {ax['A_x']:.2f}%")
    print(f"  A_y (Depth: in front/behind):     {ax['A_y']:.2f}%")
    print(f"  A_z (Vertical: on-above/below):   {ax['A_z']:.2f}%")
    print("-" * 65)
    print("Perspective Rule Accuracy:")
    print(f"  Q1 (Rule 1: Out-of-image):        {ps['Q1_OutOfImage']:.2f}%")
    print(f"  Q2 (Rule 2: First-person):        {ps['Q2_FirstPerson']:.2f}%")
    print(f"  Q3 (Rule 3: Third-person):        {ps['Q3_ThirdPerson']:.2f}%")
    print("=" * 65)
