"""
Unified Benchmark Suite for Spatio-Temporal Knowledge Graph Question Answering.
Evaluates and compares the 3 reproduced models:
1. RoBERTa-base (PLM)
2. MultiQA (Temporal KGQA)
3. STCQA (Proposed Spatio-Temporal Model)
Outputs comparative results matching Table 5 from Dai et al. (KBS 2025).
"""

import os
import sys
# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
from typing import Dict, Any

from src.dataset import load_stqad, get_vocabularies
from src.evaluation import evaluate_benchmark


# Paper reported numbers for reference in Table 5
PAPER_REPORTED = {
    "RoBERTa-base": {
        "Hits@1": {"Overall": 31.89, "DTC": 32.14, "STC": 31.88, "DDC": 33.53, "SDC": 32.04, "DC": 27.61},
        "Hits@10": {"Overall": 60.77, "DTC": 57.14, "STC": 60.87, "DDC": 61.85, "SDC": 58.29, "DC": 62.25},
    },
    "MultiQA": {
        "Hits@1": {"Overall": 51.93, "DTC": 46.43, "STC": 52.08, "DDC": 55.20, "SDC": 57.46, "DC": 39.44},
        "Hits@10": {"Overall": 76.39, "DTC": 78.57, "STC": 76.33, "DDC": 77.17, "SDC": 75.69, "DC": 76.34},
    },
    "STCQA": {
        "Hits@1": {"Overall": 61.52, "DTC": 60.71, "STC": 61.55, "DDC": 63.01, "SDC": 63.25, "DC": 53.52},
        "Hits@10": {"Overall": 84.29, "DTC": 82.14, "STC": 84.38, "DDC": 83.82, "SDC": 82.32, "DC": 86.76},
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Benchmark Runner for STKGQA")
    parser.add_argument("--pred_dir", type=str, default="experiments/predictions")
    parser.add_argument("--data_dir", type=str, default="data/stqad")
    parser.add_argument("--output_report", type=str, default="experiments/benchmark_report.md")
    return parser.parse_args()


def build_comparison_table(results_dict: Dict[str, Dict[str, Any]]) -> str:
    """
    Builds a markdown table comparing reproduced models alongside paper reference scores.
    """
    headers = [
        "Model", "Source", "Metric", "Overall", "DTC", "STC", "DDC", "SDC", "DC"
    ]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join([":---"] * 3 + [":---:"] * (len(headers) - 3)) + " |")

    models_order = ["RoBERTa-base", "MultiQA", "STCQA"]

    for model_name in models_order:
        # Paper Reported Rows
        if model_name in PAPER_REPORTED:
            for metric in ["Hits@1", "Hits@10"]:
                row = [
                    model_name,
                    "Paper Ref",
                    metric,
                    f"{PAPER_REPORTED[model_name][metric]['Overall']:.2f}%",
                    f"{PAPER_REPORTED[model_name][metric]['DTC']:.2f}%",
                    f"{PAPER_REPORTED[model_name][metric]['STC']:.2f}%",
                    f"{PAPER_REPORTED[model_name][metric]['DDC']:.2f}%",
                    f"{PAPER_REPORTED[model_name][metric]['SDC']:.2f}%",
                    f"{PAPER_REPORTED[model_name][metric]['DC']:.2f}%",
                ]
                lines.append("| " + " | ".join(row) + " |")

        # Reproduced Rows
        if model_name in results_dict:
            res = results_dict[model_name]
            for metric in ["Hits@1", "Hits@10"]:
                row = [
                    model_name,
                    "Reproduced",
                    metric,
                    f"{res.get('Overall', {}).get(metric, 0.0):.2f}%",
                    f"{res.get('DTC', {}).get(metric, 0.0):.2f}%",
                    f"{res.get('STC', {}).get(metric, 0.0):.2f}%",
                    f"{res.get('DDC', {}).get(metric, 0.0):.2f}%",
                    f"{res.get('SDC', {}).get(metric, 0.0):.2f}%",
                    f"{res.get('DC', {}).get(metric, 0.0):.2f}%",
                ]
                lines.append("| " + " | ".join(row) + " |")

        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    return "\n".join(lines)


def main():
    args = parse_args()
    pred_files = {
        "RoBERTa-base": os.path.join(args.pred_dir, "roberta_test_preds.json"),
        "MultiQA": os.path.join(args.pred_dir, "multiqa_test_preds.json"),
        "STCQA": os.path.join(args.pred_dir, "stcqa_test_preds.json"),
    }

    reproduced_results = {}

    for model_name, path in pred_files.items():
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            reproduced_results[model_name] = data.get("metrics", {})
            print(f"Loaded reproduced results for {model_name} from {path}")
        else:
            print(f"Notice: Prediction file for {model_name} not yet found at {path}")

    report_md = f"""# STKGQA Benchmark Evaluation Report

Reproducing **Dai et al. (KBS 2025)** on the **STQAD** benchmark dataset (1,063 test questions).

## Table 5 Reproduction Comparison

{build_comparison_table(reproduced_results)}

### Constraint Categories:
- **DTC (Double Timestamp Constraint):** 2-bound interval containment (`during`, `while`).
- **STC (Single Timestamp Constraint):** 1-bound timestamp comparison (`before`, `after`, `posterior to`).
- **DDC (Double Direction Constraint):** 2-axis orientation comparison (`northeast`, `southwest`, etc.).
- **SDC (Single Direction Constraint):** 1-axis orientation comparison (`north`, `south`, `east`, `west`).
- **DC (Distance Constraint):** Haversine distance ceiling calculation (`within X miles`).
"""

    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write(report_md)

    print("\n" + report_md)
    print(f"\nReport saved to: {args.output_report}")


if __name__ == "__main__":
    main()
