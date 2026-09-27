# STKGQA Benchmark Evaluation Report

Reproducing **Dai et al. (KBS 2025)** on the **STQAD** benchmark dataset (1,063 test questions).

## Table 5 Reproduction Comparison

| Model | Source | Metric | Overall | DTC | STC | DDC | SDC | DC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| RoBERTa-base | Paper Ref | Hits@1 | 31.89% | 32.14% | 31.88% | 33.53% | 32.04% | 27.61% |
| RoBERTa-base | Paper Ref | Hits@10 | 60.77% | 57.14% | 60.87% | 61.85% | 58.29% | 62.25% |
| RoBERTa-base | Reproduced | Hits@1 | 46.00% | 42.86% | 46.09% | 20.30% | 34.66% | 81.97% |
| RoBERTa-base | Reproduced | Hits@10 | 72.72% | 75.00% | 72.66% | 54.55% | 65.34% | 97.46% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MultiQA | Paper Ref | Hits@1 | 51.93% | 46.43% | 52.08% | 55.20% | 57.46% | 39.44% |
| MultiQA | Paper Ref | Hits@10 | 76.39% | 78.57% | 76.33% | 77.17% | 75.69% | 76.34% |
| MultiQA | Reproduced | Hits@1 | 6.21% | 3.57% | 6.28% | 5.76% | 9.52% | 3.10% |
| MultiQA | Reproduced | Hits@10 | 25.21% | 7.14% | 25.70% | 27.58% | 34.39% | 13.24% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| STCQA | Paper Ref | Hits@1 | 61.52% | 60.71% | 61.55% | 63.01% | 63.25% | 53.52% |
| STCQA | Paper Ref | Hits@10 | 84.29% | 82.14% | 84.38% | 83.82% | 82.32% | 86.76% |
| STCQA | Reproduced | Hits@1 | 0.94% | 0.00% | 0.97% | 1.21% | 1.59% | 0.00% |
| STCQA | Reproduced | Hits@10 | 10.16% | 7.14% | 10.24% | 9.09% | 15.34% | 5.63% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

### Constraint Categories:
- **DTC (Double Timestamp Constraint):** 2-bound interval containment (`during`, `while`).
- **STC (Single Timestamp Constraint):** 1-bound timestamp comparison (`before`, `after`, `posterior to`).
- **DDC (Double Direction Constraint):** 2-axis orientation comparison (`northeast`, `southwest`, etc.).
- **SDC (Single Direction Constraint):** 1-axis orientation comparison (`north`, `south`, `east`, `west`).
- **DC (Distance Constraint):** Haversine distance ceiling calculation (`within X miles`).
