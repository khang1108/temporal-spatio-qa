# AGENTS.md — Research Charter & Core Architecture
## Project: Spatial Intelligence in MLLMs — Explicit 3D Coordinate Frame Transformation
### Target Benchmark: SpatialMQA (ACL 2025 Long) | Baselines: LLaVA-1.5 & SpaceLLaVA

---

## 🎯 1. Mission Statement & Research Problem

This repository is dedicated to advancing **3D Spatial Reasoning in Multimodal Large Language Models (MLLMs)**. 

Standard MLLMs (LLaVA, GPT-4V, Gemini) suffer from a fundamental **"spatial intelligence bottleneck"**: their vision backbones (CLIP ViT) flatten 3D physical reality into 2D pixel grids, rendering them blind to 3D geometry, metric depth, and observer perspective transformations.

### The Anchor Benchmark: SpatialMQA (ACL 2025)
* **Dataset:** 5,392 questions over 5,063 uncalibrated, in-the-wild MS-COCO images (no bounding boxes provided).
* **Axes Evaluated:** Horizontal ($A_x$), Metric Depth ($A_y$), Vertical ($A_z$).
* **Perspective Types:** 
  - **Q1 (Out-of-image / Camera Viewpoint):** Observer looks into the scene.
  - **Q2 (First-person / Ego-centric Viewpoint):** *"If you were the giraffe in the picture, where is the sun relative to you?"*
  - **Q3 (Third-person / In-image Observer):** Spatial relation relative to an arbitrary landmark inside the scene.
* **Baseline Ceilings:** 
  - **LLaVA-1.5-7B (LoRA):** **46.56%** overall accuracy ($A_y = 29.64\%$, $\text{Q2} = 40.99\%$).
  - **SpaceLLaVA (LoRA):** **48.14%** overall accuracy ($A_z$ collapses to $31.41\%$).

---

## 🔬 2. The Core Research Gap & Driving Motivation (The "Why")

A thorough literature survey of 25 landmark papers (2020–2025) reveals a critical mathematical insight:

> ### ⚠️ The Depth-Injection Fallacy
> Monocular depth estimation models (e.g., Depth Anything V2, Metric3D) only measure scalar distance along the **camera optical axis** ($Z_{\text{camera}}$).
> 
> **Depth maps contain ZERO information about an object's internal 3D orientation (Heading Vector $\vec{h}$).**
> 
> Therefore, simply concatenating depth maps or early-fusing depth tokens (as in *Spatial-MLLM* [NeurIPS 2024] or *LLaVA-3D* [2024]) **completely fails to resolve Frame of Reference Shift (FRS)**, which is responsible for **38.2% of all spatial errors** on SpatialMQA.

### 🚀 Our Core Novelty & Unclaimed Research Gap:
**Analytical 3D Reference-Frame Transformation ($T_{\text{cam} \rightarrow \text{obj}}$)**

Instead of forcing the neural network to implicitly guess rotational transformations, we introduce a **Neuro-Symbolic Geometric Bridge**:
1. **Metric Depth Extraction:** Compute per-pixel depth $(X_{\text{cam}}, Y_{\text{cam}}, Z_{\text{cam}})$ using Depth Anything V2 directly on monocular 2D images.
2. **Object Heading Estimation:** Estimate the 3D heading/facing vector $\vec{h} = (\theta, \phi)$ of the reference subject.
3. **Analytical Coordinate Rotation:** Compute the exact 3D rotation matrix $R_{\text{cam} \rightarrow \text{obj}}$:
   $$P_{\text{obj}} = R_{\text{cam} \rightarrow \text{obj}} \cdot (P_{\text{target}} - P_{\text{subject}})$$
4. **Spatial-CoT Conditioning:** Feed the explicit relative 3D coordinate vector $(\Delta x', \Delta y', \Delta z')$ into LLaVA / SpaceLLaVA via structured Chain-of-Thought (Spatial-CoT), transforming a 40% guessing game into rigorous mathematical deduction.

---

## 🏛️ 3. Target Performance Metrics (Hypothesis & Goals)

| Metric | LLaVA-1.5 Baseline | SpaceLLaVA Baseline | **Our Target Goal** |
|---|---|---|---|
| **Overall Accuracy** | 46.56% | 48.14% | **$\ge$ 56.0% (+8 to 10%)** |
| **Q1 (Camera-centric)** | 53.14% | 54.87% | **$\ge$ 62.0%** |
| **Q2 (Ego-centric / FRS)** | **40.99%** | **42.37%** | **$\ge$ 65.0% (+23% boost)** |
| **Q3 (In-image landmark)** | 64.71% | 58.82% | **$\ge$ 70.0%** |
| **$A_x$ (Horizontal)** | 55.71% | 56.00% | **$\ge$ 65.0%** |
| **$A_y$ (Metric Depth)** | **29.64%** | 51.85% | **$\ge$ 60.0% (+30% boost)** |
| **$A_z$ (Vertical)** | 48.13% | 31.41% | **$\ge$ 55.0%** |

---

## 📂 4. Repository Structure & Artifact Layout

```
Spatial-Temporal-KG/
├── AGENTS.md                               # Project Charter, Research Hypothesis & Guidelines
├── docs/
│   ├── TRAINING_GUIDE_A100.md             # A100 execution guide (ZeRO-2, LoRA r=128, BF16)
│   └── survey/
│       ├── SURVEY_MASTER.md               # 25-paper deep literature survey (4 paradigms)
│       ├── survey_papers.xlsx             # Live reading tracker & limitations workbook
│       ├── survey_papers.csv              # Downloaded Google Sheet raw export
│       └── papers_pdf/                    # Curated local library of foundational PDFs
├── visualizations/
│   ├── baselines_workflow.html            # Standalone visualizer of baseline LoRA architectures
│   └── spatial_mllm_survey_limitations.html # Interactive proof (SVG & Slider) of why depth alone fails
├── data/
│   └── spatial_mqa/
│       ├── train.jsonl (3,780 QAs)        # Raw train split
│       ├── dev.jsonl (536 QAs)            # Raw validation split
│       ├── test.jsonl (1,076 QAs)         # Official test benchmark
│       ├── invisible.jsonl (811 QAs)      # Challenging invisible split
│       ├── train_3780.json                # LLaVA-formatted instruction dataset
│       ├── dev_536.json                   # LLaVA-formatted dev dataset
│       └── images/ (5,063 images)         # Exact 800MB image set from Hugging Face CDN
├── scripts/
│   ├── download_images.py                 # Fast HF CDN downloader (skips 6.2GB throttled COCO zip)
│   ├── train_mem.py                       # Standalone entrypoint wrapper with graceful SDPA fallback
│   ├── train_llava_lora.sh                # LLaVA-1.5 LoRA training launcher (DeepSpeed ZeRO-2)
│   ├── train_spacellava_lora.sh           # SpaceLLaVA LoRA training launcher
│   ├── run_eval.py                        # Standalone evaluation & metric computation runner
│   └── zero2.json                         # DeepSpeed ZeRO-2 configuration
├── src/
│   ├── data/format_llava.py               # JSONL to LLaVA conversational format converter
│   └── evaluation/metrics.py              # Macro-F1, Precision, Recall, Per-axis, Per-Q evaluators
└── experiments/
    ├── checkpoints/                       # Trained LoRA adapter weights
    └── predictions/                       # Output JSONL predictions for error analysis
```

---

## 🤖 5. Agent Behavioral Rules & Operational Directives

Any AI agent interacting with this workspace must adhere strictly to the following directives:

1. **Protect the Scope:** 
   - **Do NOT** re-introduce old Knowledge Graph (KGQA / STCQA) files, datasets, or libraries.
   - **Do NOT** add unrelated baseline models (BLIP, IDEFICS, mPLUG, closed APIs). The project is strictly focused on **LLaVA-1.5-7B** and **SpaceLLaVA**.
2. **Respect the Research Gap:**
   - Every proposed improvement must be benchmarked against the **38.2% FRS error** and the **29.64% Depth error ($A_y$)**.
   - Do not propose superficial prompt engineering without geometric grounding.
3. **Hardware & Environment Compatibility:**
   - All code must run on **1x NVIDIA A100 (40GB)** under Linux with **PyTorch 2.5.1 + CUDA 12.1**.
   - Always leverage native PyTorch SDPA (Scaled Dot-Product Attention) to avoid source-compilation issues with `flash-attn`.
   - Always set `TRITON_CACHE_DIR=/tmp/triton_${USER}` to avoid hanging on NFS network mounts.
4. **Data Integrity:**
   - The official image dataset consists of **exactly 5,063 images (~800 MB)** located in `data/spatial_mqa/images/`.
   - Never re-attempt downloading the full 6.2GB `test2017.zip` from throttled COCO servers.
5. **Git Synchronization:**
   - Always verify git status and commit clean, semantic commits (`feat:`, `fix:`, `docs:`, `refactor:`) before and after major modifications.
   - Keep remote `origin/main` synchronized.
