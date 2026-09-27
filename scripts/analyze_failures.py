"""
Diagnostic Error & Failure Analysis Script for STKGQA Baselines.

Inspects prediction files (experiments/predictions/*_test_preds.json) to analyze:
1. Failure rate breakdown across constraint categories (DTC, STC, DDC, SDC, DC).
2. Concrete failure examples (question, gold answer, top-k model predictions, rank).
3. Export detailed markdown diagnostic report.

Usage:
    python scripts/analyze_failures.py --model stcqa
    python scripts/analyze_failures.py --model stcqa --constraint DC --limit 5
    python scripts/analyze_failures.py --model multiqa --export_report
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_args():
    parser = argparse.ArgumentParser(description="Error & Failure Analysis for STKGQA Models")
    parser.add_argument(
        "--model",
        type=str,
        default="stcqa",
        choices=["roberta", "multiqa", "stcqa"],
        help="Model to analyze (default: stcqa)"
    )
    parser.add_argument(
        "--pred_file",
        type=str,
        default=None,
        help="Path to prediction JSON (overrides default path for --model)"
    )
    parser.add_argument(
        "--constraint",
        type=str,
        default="all",
        choices=["all", "DTC", "STC", "DDC", "SDC", "DC"],
        help="Filter failure cases by constraint type (default: all)"
    )
    parser.add_argument(
        "--failure_type",
        type=str,
        default="missed_hit10",
        choices=["missed_hit1", "missed_hit10"],
        help="Type of failure to inspect (default: missed_hit10)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of concrete failure examples to display (default: 5)"
    )
    parser.add_argument(
        "--export_report",
        action="store_true",
        help="Export comprehensive failure analysis report to experiments/failure_report_<model>.md"
    )
    return parser.parse_args()


def load_prediction_data(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Prediction file not found at: {file_path}\n"
            f"Please run the model evaluation first: python baselines/<model>/run.py --eval_only"
        )
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def format_example_markdown(sample: Dict[str, Any], rank_idx: int) -> str:
    lines = [
        f"### Case #{rank_idx} (Sample ID: {sample['sample_id']})",
        f"- **Question:** `{sample['question']}`",
    ]
    if sample.get("paraphrased_question"):
        lines.append(f"- **Paraphrased:** *\"{sample['paraphrased_question']}\"*")

    lines.extend([
        f"- **Constraints:** Temporal: `{sample['temporal_constraint']}` | Spatial: `{sample['spatial_constraint']}`",
        f"- **Gold Answer(s):** **{', '.join(sample['gold_answers'])}**",
        f"- **Gold Rank:** `{'#' + str(sample['gold_rank']) if sample['gold_rank'] else 'Not in Top 10'}`",
        f"- **Top Predictions:**",
    ])
    for i, p in enumerate(sample["top_predictions"][:5], start=1):
        is_gold = p in set(sample["gold_answers"])
        mark = " (CORRECT)" if is_gold else ""
        lines.append(f"  {i}. `{p}`{mark}")

    return "\n".join(lines)


def main():
    args = parse_args()
    default_pred_paths = {
        "roberta": "experiments/predictions/roberta_test_preds.json",
        "multiqa": "experiments/predictions/multiqa_test_preds.json",
        "stcqa": "experiments/predictions/stcqa_test_preds.json",
    }
    pred_path = args.pred_file or default_pred_paths.get(args.model.lower())
    print(f"Loading predictions from: {pred_path}...")
    data = load_prediction_data(pred_path)

    model_name = data.get("model", args.model.upper())
    metrics = data.get("metrics", {})
    sample_details = data.get("sample_details", [])

    if not sample_details:
        print(f"\nNotice: {pred_path} contains summary metrics but no per-sample details.")
        print("Please re-run evaluation with the updated run.py script to generate full per-sample details.")
        return

    # Filter failures
    failures = []
    for s in sample_details:
        # Check failure criterion
        if args.failure_type == "missed_hit1" and s["is_hit_1"]:
            continue
        if args.failure_type == "missed_hit10" and s["is_hit_10"]:
            continue

        # Check constraint filter
        if args.constraint != "all":
            if s["temporal_constraint"] != args.constraint and s["spatial_constraint"] != args.constraint:
                continue

        failures.append(s)

    total_samples = len(sample_details)
    fail_rate = (len(failures) / total_samples) * 100.0 if total_samples > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"Failure Analysis Report: {model_name}")
    print("=" * 60)
    print(f"Total Test Questions: {total_samples}")
    print(f"Criterion: {args.failure_type} (Constraint Filter: {args.constraint})")
    print(f"Total Matched Failures: {len(failures)} ({fail_rate:.2f}%)")

    # Metrics Summary Table
    print("\nBenchmark Accuracy Overview:")
    print(f"{'Category':<10} | {'Count':<6} | {'Hits@1':<8} | {'Hits@10':<8} | {'Fail Rate (Hits@10)':<20}")
    print("-" * 65)
    for cat in ["Overall", "DTC", "STC", "DDC", "SDC", "DC"]:
        if cat in metrics:
            m = metrics[cat]
            h1 = m.get("Hits@1", 0.0)
            h10 = m.get("Hits@10", 0.0)
            fail_p = 100.0 - h10
            cnt = m.get("count", 0)
            print(f"{cat:<10} | {cnt:<6} | {h1:>6.2f}% | {h10:>6.2f}% | {fail_p:>18.2f}%")

    print("\n" + "=" * 60)
    print(f"Top {min(args.limit, len(failures))} Concrete Failure Examples:")
    print("=" * 60)

    for i, f_case in enumerate(failures[:args.limit], start=1):
        print(f"\n[Failure Example #{i}] (Sample ID: {f_case['sample_id']})")
        print(f"  Question:      {f_case['question']}")
        if f_case.get("paraphrased_question"):
            print(f"  Paraphrased:   {f_case['paraphrased_question']}")
        print(f"  Constraints:   Temporal={f_case['temporal_constraint']} | Spatial={f_case['spatial_constraint']}")
        print(f"  Gold Answer:   {', '.join(f_case['gold_answers'])}")
        print(f"  Gold Rank:     {'#' + str(f_case['gold_rank']) if f_case['gold_rank'] else 'Not in Top 10'}")
        print(f"  Top 3 Preds:   {', '.join(f_case['top_predictions'][:3])}")

    # Export markdown report if requested
    if args.export_report:
        report_path = f"experiments/failure_report_{args.model.lower()}.md"
        md_lines = [
            f"# Detailed Failure Analysis Report: {model_name}",
            "",
            f"- **Source File:** `{pred_path}`",
            f"- **Total Samples:** {total_samples}",
            f"- **Total Missed in Top 10:** {len([s for s in sample_details if not s['is_hit_10']])} ({(len([s for s in sample_details if not s['is_hit_10']]) / total_samples) * 100:.2f}%)",
            "",
            "## Accuracy & Failure Rate Breakdown",
            "",
            "| Category | Count | Hits@1 | Hits@10 | Missed in Top 10 (%) |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
        for cat in ["Overall", "DTC", "STC", "DDC", "SDC", "DC"]:
            if cat in metrics:
                m = metrics[cat]
                md_lines.append(
                    f"| {cat} | {m.get('count', 0)} | {m.get('Hits@1', 0.0):.2f}% | {m.get('Hits@10', 0.0):.2f}% | {100.0 - m.get('Hits@10', 0.0):.2f}% |"
                )

        md_lines.extend([
            "",
            "## Selected Failure Cases for Qualitative Inspection",
            "",
        ])
        for idx, f_case in enumerate(failures[:20], start=1):
            md_lines.append(format_example_markdown(f_case, idx))
            md_lines.append("")

        with open(report_path, "w", encoding="utf-8") as rf:
            rf.write("\n".join(md_lines))
        print(f"\n[Exported] Comprehensive failure report saved to: {report_path}")


if __name__ == "__main__":
    main()
