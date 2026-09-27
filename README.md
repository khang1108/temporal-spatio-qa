# Spatio-Temporal Knowledge Graph Question Answering (STKGQA)

A lightweight, research-grade reproduction and benchmarking framework for Spatio-Temporal Question Answering over Knowledge Graphs, reproducing:
> **Question answering over spatio-temporal knowledge graph**  
> Xinbang Dai, Huiying Li, Nan Hu, Yongrui Chen, Rihui Jin, Huikang Hu, Guilin Qi  
> *Knowledge-Based Systems*, Volume 329, 2025, 114314.

---

## Target Baselines

This repository implements the 3 primary models from the paper:
1. **RoBERTa-base (PLM Baseline):** Text-only representation projected into entity embedding space.
2. **MultiQA (Temporal KGQA Baseline):** Multi-granularity temporal question reasoning utilizing TComplEx scoring.
3. **STCQA (Proposed Method):** Joint Spatio-Temporal Complex Embedding (ST-TComplEx) with 2-layer Transformer Fusion and Dynamic Constraint Filtering.

---

## Repository Structure

```
Spatial-Temporal-KG/
├── data/
│   └── stqad/
│       ├── train_datas.json          # 8,505 training questions
│       ├── val_datas.json            # 1,063 validation questions
│       ├── test_datas.json           # 1,063 test questions
│       ├── entity2id.json            # 5,897 unique entities
│       └── relation2id.json          # 11 relation types
├── src/
│   ├── dataset.py                   # PyTorch Dataset & DataLoader
│   ├── evaluation.py                # Hits@1, Hits@3, Hits@10 with constraint breakdown
│   ├── utils_geo.py                 # Haversine distance and direction checking
│   └── utils_time.py                # Interval and timestamp comparison logic
├── baselines/
│   ├── roberta/                     # RoBERTa-base PLM baseline
│   │   ├── model.py
│   │   └── run.py
│   ├── multiqa/                     # MultiQA Temporal baseline
│   │   ├── model.py
│   │   └── run.py
│   └── stcqa/                       # STCQA Proposed Method
│       ├── st_embedding.py          # ST-TComplEx formulation
│       ├── model.py                 # Transformer Fusion & Bidirectional Scoring
│       ├── constraint_filter.py     # Deterministic Answer Filtering
│       └── run.py
├── experiments/
│   ├── checkpoints/                 # Saved best model checkpoints
│   ├── predictions/                 # Output prediction JSON files
│   └── benchmark_report.md          # Generated Table 5 comparative report
├── benchmark.py                     # Aggregates and formats comparative benchmark
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Environment Setup
```bash
# Using uv (recommended)
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Running Individual Baselines

#### Run RoBERTa-base:
```bash
python baselines/roberta/run.py --epochs 10 --batch_size 32 --lr 2e-5
```

#### Run MultiQA:
```bash
python baselines/multiqa/run.py --epochs 10 --batch_size 32 --lr 2e-5
```

#### Run Proposed STCQA:
```bash
python baselines/stcqa/run.py --epochs 10 --batch_size 32 --lr 2e-5
```

*Tip for quick debugging on CPU:* Add `--max_samples 50 --epochs 1` to test the pipeline in seconds.

### 3. Generate Comparative Benchmark Table (Table 5)
```bash
python benchmark.py
```
Outputs the markdown report comparing all reproduced models against paper reference scores across:
- **DTC:** Double Timestamp Constraints (`during`, `while`)
- **STC:** Single Timestamp Constraints (`before`, `after`, `posterior to`)
- **DDC:** Double Direction Constraints (`northeast`, `southwest`)
- **SDC:** Single Direction Constraints (`north`, `south`, `east`, `west`)
- **DC:** Distance Constraints (`within X miles`)

---

## How to Mutate & Try New Methods

To test a new architecture or variation (e.g. rotary embeddings, LLM verifier):
1. Create a new folder under `baselines/my_mutation/`.
2. Reuse `src.dataset.STQADataset` and `src.dataset.get_vocabularies` for data loading.
3. Call `src.evaluation.evaluate_benchmark(predictions, dataset_items)` to compute metrics.
4. Save your predictions to `experiments/predictions/my_mutation_test_preds.json`.
5. Run `python benchmark.py` to see your new method added to the leaderboard!
