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
│   ├── stqad/                   # 10,631 QA pairs (train/val/test) & entity/relation dicts
│   │   ├── train_datas.json     # 8,505 training questions
│   │   ├── val_datas.json       # 1,063 validation questions
│   │   ├── test_datas.json      # 1,063 test questions
│   │   ├── entity2id.json       # 5,897 unique entities
│   │   └── relation2id.json     # 11 relation types
│   ├── raw_yago15k/             # Raw YAGO3-15k quadruplet facts
│   ├── stkg/                    # Constructed STKG per Section 4.1
│   │   ├── facts.tsv            # 138,056 quadruplets (s, r, o, ts, te, loc)
│   │   ├── entity_metadata.json # Geo coordinates & temporal lifespan (100% QA coverage)
│   │   └── stats.json           # STKG graph statistics
│   └── cached_embeddings/       # Precomputed RoBERTa [CLS] vectors for CPU acceleration
├── src/
│   ├── data/
│   │   ├── dataset.py           # PyTorch Dataset & DataLoader
│   │   ├── construct_stkg.py    # Section 4.1 STKG extraction & Wikipedia API geo-resolution
│   │   └── precompute.py        # Offline RoBERTa [CLS] embedding cache script
│   ├── evaluation/
│   │   ├── metrics.py           # Hits@1, Hits@3, Hits@10 with constraint breakdown
│   │   ├── benchmark.py         # Table 5 comparative aggregator & markdown reporter
│   │   └── analyze_failures.py  # Error inspection and diagnostic report generator
│   └── utils/
│       ├── geo.py               # Haversine distance and directional calculations
│       ├── time.py              # Interval and timestamp comparison logic
│       ├── checkpoint.py        # Auto-download and verification for .pt weights
│       └── download_checkpoints.py # Standalone CLI tool to download .pt checkpoints
├── baselines/
│   ├── roberta/                 # RoBERTa-base PLM baseline
│   │   ├── model.py
│   │   └── run.py
│   ├── multiqa/                 # MultiQA Temporal baseline
│   │   ├── model.py
│   │   └── run.py
│   └── stcqa/                   # STCQA Proposed Method
│       ├── st_embedding.py      # ST-TComplEx formulation
│       ├── model.py             # Transformer Fusion & Bidirectional Scoring
│       ├── constraint_filter.py # Deterministic Spatio-Temporal Answer Filtering
│       └── run.py
├── experiments/
│   ├── checkpoints/             # Model checkpoint weights (.pt files)
│   ├── predictions/             # Output prediction JSON files
│   └── benchmark_report.md      # Generated Table 5 comparative report
├── requirements.txt
└── README.md
```

---

## Model Checkpoint Auto-Download Logic

When evaluating models or calling `load_model()` / `model.load_checkpoint()`:
- The loader first checks whether the `.pt` checkpoint file exists locally.
- If the `.pt` file is missing, it automatically downloads the pre-trained weights from the remote repository / URL with a progress bar.
- Custom download URLs can be provided via the `--checkpoint_url` CLI flag or environment variables:
  - `ROBERTA_CHECKPOINT_URL`
  - `MULTIQA_CHECKPOINT_URL`
  - `STCQA_CHECKPOINT_URL`

You can also pre-download all weights in batch using the standalone CLI script:
```bash
# Download all model weights
python -m src.utils.download_checkpoints --model all

# Download a specific model weight
python -m src.utils.download_checkpoints --model stcqa

# Download with a custom URL
python -m src.utils.download_checkpoints --model stcqa --url <CUSTOM_URL>
```

---

## Quickstart

### 1. Environment Setup
```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Running Evaluation (Auto-downloads .pt if missing)

#### Evaluate RoBERTa-base:
```bash
python baselines/roberta/run.py --eval_only
```

#### Evaluate MultiQA:
```bash
python baselines/multiqa/run.py --eval_only
```

#### Evaluate Proposed STCQA:
```bash
python baselines/stcqa/run.py --eval_only
```

### 3. Training from Scratch

To train any model from scratch (exact paper configuration: 60 epochs, batch size 150, lr 2e-5):
```bash
# RoBERTa-base
python baselines/roberta/run.py

# MultiQA
python baselines/multiqa/run.py

# STCQA
python baselines/stcqa/run.py
```
*(All hyperparameter defaults now match Appendix B of the paper: `--epochs 60 --batch_size 150 --lr 2e-5`)*.

Tip for quick debugging: Add `--max_samples 50 --epochs 1` to test the pipeline in seconds.

### 4. Generate Comparative Benchmark Table (Table 5)
```bash
python -m src.evaluation.benchmark
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
2. Reuse `src.data.STQADataset` and `src.data.get_vocabularies` for data loading.
3. Call `src.evaluation.evaluate_benchmark(predictions, dataset_items)` to compute metrics.
4. Save your predictions to `experiments/predictions/my_mutation_test_preds.json`.
5. Run `python -m src.evaluation.benchmark` to see your new method added to the leaderboard.
