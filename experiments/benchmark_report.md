# STKGQA Benchmark Evaluation Report

Reproducing **Dai et al. (KBS 2025)** on the **STQAD** benchmark dataset (1,063 test questions).

## Table 5 Reproduction Comparison

| Model | Source | Metric | Overall | DTC | STC | DDC | SDC | DC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| RoBERTa-base | Paper Ref | Hits@1 | 31.89% | 32.14% | 31.88% | 33.53% | 32.04% | 27.61% |
| RoBERTa-base | Paper Ref | Hits@10 | 60.77% | 57.14% | 60.87% | 61.85% | 58.29% | 62.25% |
| RoBERTa-base | Reproduced | Hits@1 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| RoBERTa-base | Reproduced | Hits@10 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MultiQA | Paper Ref | Hits@1 | 51.93% | 46.43% | 52.08% | 55.20% | 57.46% | 39.44% |
| MultiQA | Paper Ref | Hits@10 | 76.39% | 78.57% | 76.33% | 77.17% | 75.69% | 76.34% |
| MultiQA | Reproduced | Hits@1 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| MultiQA | Reproduced | Hits@10 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| STCQA | Paper Ref | Hits@1 | 61.52% | 60.71% | 61.55% | 63.01% | 63.25% | 53.52% |
| STCQA | Paper Ref | Hits@10 | 84.29% | 82.14% | 84.38% | 83.82% | 82.32% | 86.76% |
| STCQA | Reproduced | Hits@1 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| STCQA | Reproduced | Hits@10 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

### Constraint Categories:
- **DTC (Double Timestamp Constraint):** 2-bound interval containment (`during`, `while`).
- **STC (Single Timestamp Constraint):** 1-bound timestamp comparison (`before`, `after`, `posterior to`).
- **DDC (Double Direction Constraint):** 2-axis orientation comparison (`northeast`, `southwest`, etc.).
- **SDC (Single Direction Constraint):** 1-axis orientation comparison (`north`, `south`, `east`, `west`).
- **DC (Distance Constraint):** Haversine distance ceiling calculation (`within X miles`).
