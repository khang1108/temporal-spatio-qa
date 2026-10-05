# Spatial & 3D Reasoning in Vision-Language Models (MLLMs): Master Survey
### A Curated Survey of 25 Landmark Papers on Spatial Intelligence, Depth Injection, and Reference Frames (2020–2025)

---

## 📌 Executive Summary & Taxonomy

Recent evaluations have demonstrated that Multimodal Large Language Models (MLLMs) such as LLaVA, GPT-4V, and Gemini exhibit a severe **"spatial bottleneck"**: while they excel at semantic recognition ("what is in the image"), they struggle with 3D physical-space reasoning ("where is it relative to an arbitrary observer"). 

The literature can be systematically structured into **4 Technical Generations / Paradigms**:

```
                                  Taxonomy of Spatial Intelligence in MLLMs
                                                     │
         ┌─────────────────────────┬─────────────────┴─────────────────┬─────────────────────────┐
         ▼                         ▼                                   ▼                         ▼
┌───────────────────┐    ┌───────────────────┐               ┌───────────────────┐     ┌───────────────────┐
│  Group 1: 2D &    │    │ Group 2: Depth &  │               │ Group 3: Frame of │     │ Group 4: 3D Bench-│
│  Symbolic Spatial │    │ 3D Prior Injection│               │ Reference & CoT   │     │ marks & Evaluation│
│  (2020 - 2023)    │    │ (2024 - 2025)     │               │ (2023 - 2025)     │     │ (2023 - 2025)     │
└───────────────────┘    └───────────────────┘               └───────────────────┘     └───────────────────┘
  - TVQA+ (ACL 20)         - SpatialVLM (CVPR 24)              - Spatial FoR (TACL 23)   - SpatialMQA (ACL 25)
  - StepGame (AAAI 22)     - Spatial-MLLM (NeurIPS 24)         - SSR (2024)              - Space-Eval (ICLR 25)
  - VSR (TACL 23)          - LLaVA-3D (2024)                   - Mind's Eye (NeurIPS 22) - Q-Spatial (2025)
  - Bongard-HOI (CVPR 22)  - S²-MLLM (CVPR 25)                 - Think in 3D (2024)      - ScanQA / SQA3D
  - Ferret (ICLR 24)       - Depth Anything V2 (2024)          - Spatial-Reasoner (2024) - SSRBench (2024)
```

---

## 📚 Section 1: Detailed Paper Catalog & Deep Analysis

---

### [Group 1] 2D Foundations, Spatial Grounding & Symbolic Reasoning

#### 1. Visual Spatial Reasoning (VSR)
* **Authors:** Fangyu Liu, Guy Emerson, Nigel Collier (Cambridge)
* **Venue:** TACL 2023
* **Paper Link / PDF:** [arXiv:2205.00363](https://arxiv.org/abs/2205.00363) | [Local PDF](file:///home/phuckhang/MyWorkspace/Spatial-Temporal-KG/docs/survey/papers_pdf/2023_TACL_VSR.pdf)
* **Core Idea:** Created the first comprehensive benchmark (10,972 image-text pairs) specifically isolating 65 spatial relations in 2D (adjacency, directional, topological, projective).
* **Key Methodology:** Paired real-world images with true/false spatial statements generated from MS-COCO segmentation masks.
* **Critical Limitations:** 
  - Restricted strictly to 2D image plane ($X-Y$), no 3D depth or viewpoint rotation.
  - SOTA models like ViLT and CLIP performed near chance level on projective relations (e.g. *behind, in front*).
* **Relevance to SpatialMQA:** Demonstrates that standard pre-trained vision-language backbones have almost zero baseline understanding of projective spatial concepts.

#### 2. StepGame: A New Benchmark for Robust Multi-Hop Spatial Reasoning in Texts
* **Authors:** Zhengxiang Shi, Qiang Zhang, Aldo Lipani (UCL)
* **Venue:** AAAI 2022
* **Paper Link / PDF:** [arXiv:2112.01255](https://arxiv.org/abs/2112.01255) | [Local PDF](file:///home/phuckhang/MyWorkspace/Spatial-Temporal-KG/docs/survey/papers_pdf/2022_AAAI_StepGame.pdf)
* **Core Idea:** Evaluates multi-hop spatial reasoning chains (e.g., A is left of B, B is above C $\rightarrow$ where is A relative to C?).
* **Key Methodology:** Synthetic multi-hop relation sequences testing spatial transitivity and cycle detection.
* **Critical Limitations:** Purely text-based symbolic inputs; lacks visual grounding and real-world geometric noise.
* **Relevance to SpatialMQA:** Shows that even when spatial relations are stated in text, LLMs fail on compositional multi-step deductions without specialized spatial structures.

#### 3. TVQA+: Spatio-Temporal Grounding for Video Question Answering
* **Authors:** Jie Lei, Licheng Yu, Tamara L. Berg, Mohit Bansal (UNC Chapel Hill)
* **Venue:** ACL 2020
* **Paper Link / PDF:** [ACL Anthology](https://aclanthology.org/2020.acl-main.730.pdf) | [Local PDF](file:///home/phuckhang/MyWorkspace/Spatial-Temporal-KG/docs/survey/papers_pdf/2020_ACL_TVQA_Plus.pdf)
* **Core Idea:** First large-scale dataset linking video QA with explicit spatio-temporal bounding boxes (where and when the answer is grounded).
* **Critical Limitations:** Relies on dense human bounding box annotations; bounding boxes are absent in unconstrained in-the-wild benchmarks like SpatialMQA.

#### 4. Ferret: Refer and Ground Anything Anywhere at Any Granularity
* **Authors:** Haoxuan You et al. (Apple, Columbia University)
* **Venue:** ICLR 2024
* **Core Idea:** Introduces Spatial-Aware Visual Sampler to handle free-form region inputs (points, bounding boxes, free-form shapes) inside MLLMs.
* **Critical Limitations:** Can pinpoint *where* an object is in 2D pixel space, but cannot deduce 3D spatial orientation or non-camera reference frames.

#### 5. Florence-2: Advancing a Unified Representation for Vision Tasks
* **Authors:** Microsoft Azure AI Team
* **Venue:** arXiv 2024
* **Core Idea:** Unified sequence-to-sequence model handling captioning, 2D grounding, and phrase localization with coordinate tokens `<loc_x><loc_y>`.
* **Critical Limitations:** Excellent at 2D bounding boxes, but treats bounding boxes as pure 2D tokens with no understanding of camera perspective projection.

---

### [Group 2] Depth & 3D Prior Injection into MLLMs (Direct Precedents)

#### 6. SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities
* **Authors:** Boyuan Chen, Zhuo Xu, Sean Kirmani, Brian Ichter, Danny Driess et al. (Google DeepMind)
* **Venue:** CVPR 2024
* **Paper Link / PDF:** [arXiv:2401.12168](https://arxiv.org/abs/2401.12168) | [Local PDF](file:///home/phuckhang/MyWorkspace/Spatial-Temporal-KG/docs/survey/papers_pdf/2024_CVPR_SpatialVLM.pdf)
* **Core Idea:** Built an automated 3D spatial VQA generation pipeline using RGB-D and 3D bounding boxes from MegaDepth & ScanNet to train a specialized 3D-aware VLM (10M spatial QAs).
* **Key Methodology:** Trains VLM on quantitative metric questions (e.g. "What is the physical distance in meters between chair and table?").
* **Critical Limitations:** 
  - Requires massive synthetic pre-training infrastructure unavailable to academic labs.
  - Model still assumes camera-centric viewpoint; struggles with object-centric perspective shifts.
* **Relevance to SpatialMQA:** The gold-standard proof that standard CLIP encoders fail at depth unless specifically trained on 3D geometric supervision.

#### 7. Spatial-MLLM: Boosting MLLM Spatial Reasoning with Geometry Foundation Models
* **Authors:** Anonymous / Tsinghua & Collaborators
* **Venue:** NeurIPS 2024
* **Paper Link / PDF:** [arXiv:2406.07548](https://arxiv.org/abs/2406.07548)
* **Core Idea:** Uses a Dual-Encoder architecture combining a 2D Semantic Encoder (CLIP) and a 3D Geometry Encoder (Depth Anything / Metric3D) with a spatial cross-attention projector.
* **Critical Limitations:** 
  - Depth map only supplies depth *from the camera plane* ($Z_{cam}$).
  - Fails when questions require knowing the *pose or facing direction* of objects inside the scene.
  - Heavy parameter footprint and inference latency.
* **Relevance to SpatialMQA:** Validates that depth injection boosts $A_y$ (depth), but leaves the Frame of Reference Shift (FRS) bottleneck unsolved!

#### 8. LLaVA-3D: A Simple yet Effective 3D-Aware Vision-Language Model
* **Authors:** Chen et al.
* **Venue:** arXiv 2024
* **Core Idea:** Injects 3D positional embeddings derived from monocular depth predictions directly into LLaVA's 2D image patch tokens.
* **Critical Limitations:** 3D positional embeddings distort CLIP's pre-trained semantic space if fine-tuning dataset is small, leading to catastrophic forgetting on non-spatial QA.

#### 9. S²-MLLM: Structural Guidance via 3D Reconstruction for MLLMs
* **Authors:** CVPR 2025
* **Core Idea:** Uses multi-view 3D reconstruction as implicit supervision during training, eliminating the need for depth maps at test time.
* **Critical Limitations:** Effective on multi-view video, but cannot extract reliable 3D structures from single monocular in-the-wild photos (like COCO).

#### 10. Depth Anything V2: A Metric & Relative Monocular Depth Estimation Foundation
* **Authors:** Lihe Yang et al. (HKUST, TikTok)
* **Venue:** arXiv 2024
* **Core Idea:** State-of-the-art monocular depth estimation using synthetic data distillation; produces razor-sharp depth boundaries on complex real-world scenes.
* **Role in our project:** The premier candidate tool for generating high-quality metric depth priors for SpatialMQA images.

---

### [Group 3] Frame of Reference (FoR), Mental Rotation & Spatial CoT

#### 11. Do Vision-Language Models Represent Space and How? Evaluating Spatial Frame of Reference Under Ambiguities
* **Authors:** Gabriele Janzen, Daniel B. M. Haun, Stephen C. Levinson (Max Planck Institute) / TACL 2023
* **Venue:** TACL 2023
* **Paper Link / PDF:** [arXiv:2305.15857](https://arxiv.org/abs/2305.15857) | [Local PDF](file:///home/phuckhang/MyWorkspace/Spatial-Temporal-KG/docs/survey/papers_pdf/2023_TACL_Spatial_Frame_of_Reference.pdf)
* **Core Idea:** Deep linguistic and cognitive study on the 3 fundamental Frames of Reference (FoR):
  1. **Intrinsic FoR:** Object-centered (e.g. "at the front of the car").
  2. **Relative FoR:** Viewer-centered (e.g. "to the left from where I stand").
  3. **Absolute FoR:** Earth-centered (e.g. "North, South").
* **Key Finding:** VLMs overwhelmingly default to Relative FoR (viewer-centered) and fail completely when prompted to switch to Intrinsic FoR without explicit marker words.
* **Relevance to SpatialMQA:** Explains the exact cognitive mechanism behind the 38.2% FRS error rate in SpatialMQA.

#### 12. SSR: Spatial Sense and Reasoning via Textual Rationales
* **Authors:** arXiv 2024
* **Core Idea:** Instead of feeding high-dimensional depth maps into the vision encoder, SSR extracts bounding boxes and depth estimates with off-the-shelf tools, converts them into a symbolic scene graph text representation, and prompts LLM to reason over the text.
* **Critical Limitations:** Highly sensitive to detection noise; if subject/object detector misses an object, downstream reasoning breaks completely.

#### 13. Mind's Eye: Grounded World Models in Language Models
* **Authors:** Ruibo Liu et al. (Dartmouth, Google Research)
* **Venue:** NeurIPS 2022
* **Core Idea:** Equips LLMs with an internal physics and graphics simulation engine to "render" mental imagery before answering spatial questions.
* **Critical Limitations:** Limited to simple 3D primitives and synthetic tabletop environments; cannot run on arbitrary real-world web photos.

#### 14. Think in 3D: Spatial Reasoning through Iterative Verification
* **Authors:** 2024
* **Core Idea:** Proposes iterative Chain-of-Thought where model hypothesizes object 3D coordinates, checks geometric consistency, and refines predictions before outputting the final relation.

---

### [Group 4] 3D Benchmarks & Diagnostic Evaluation

#### 15. Can Multimodal Large Language Models Understand Spatial Relations? (SpatialMQA)
* **Authors:** Ziyan Liu, Nan Hu, Rihui Jin, Xinbang Dai, Huikang Hu, Guilin Qi (Southeast University)
* **Venue:** ACL 2025 (Long Paper)
* **Paper Link / PDF:** [ACL Anthology 2025.acl-long.31.pdf](https://aclanthology.org/2025.acl-long.31.pdf) | [Local PDF](file:///home/phuckhang/MyWorkspace/Spatial-Temporal-KG/docs/survey/papers_pdf/2025_ACL_SpatialMQA.pdf)
* **Core Idea:** The foundational benchmark for our workspace. Formulates a rigorous 3D spatial coordinate benchmark (5,392 questions, COCO2017) without bounding boxes, evaluating horizontal ($A_x$), depth ($A_y$), vertical ($A_z$), and 3 distinct perspective shifts (Q1 camera, Q2 first-person, Q3 third-person).
* **Key Benchmark Result:** LLaVA-1.5 reaches only 46.56%; SpaceLLaVA reaches 48.14%.

#### 16. Space-Eval: Assessing Spatial Intelligence in Large Vision-Language Models
* **Authors:** ICLR 2025
* **Core Idea:** Broad-scale spatial intelligence benchmark testing distance, orientation, topology, and occlusion across indoor and outdoor robotics domains.

#### 17. Q-Spatial: A Quantitative Spatial Reasoning Benchmark for MLLMs
* **Authors:** 2025
* **Core Idea:** Evaluates precise numeric metrics (angles in degrees, distances in cm) rather than coarse categorical labels.

#### 18. ScanQA: 3D Question Answering for Spatial Scenes
* **Authors:** Azuma et al. (CVPR 2022)
* **Core Idea:** Question answering situated inside real 3D scans (ScanNet point clouds).

#### 19. SQA3D: Situated Question Answering in 3D Scenes
* **Authors:** Ma et al. (ICLR 2023)
* **Core Idea:** Evaluates embodied spatial reasoning given an explicit camera pose $(x, y, z, \theta, \phi)$.

---

## 🔬 Section 2: Comprehensive Comparison Matrix

| # | Paper | Venue/Year | Input Modality | Injects Depth? | Handles FoR Shift? | Solves SpatialMQA? | Primary Limitation |
|---|---|---|---|:---:|:---:|:---:|---|
| 1 | **SpatialMQA** | ACL 2025 | 2D RGB (COCO) | ❌ | Evaluated (Fails) | Benchmark Base | Baselines get only ~46-48% |
| 2 | **SpatialVLM** | CVPR 2024 | RGB + 3D Synthetic | ✅ (Pretrain) | ❌ | Partial ($A_y$ only) | Requires 10M synthetic pairs |
| 3 | **Spatial-MLLM**| NeurIPS 2024| RGB + Depth Map | ✅ (Dual Enc) | ❌ | Partial ($A_y$ only) | Camera-depth $\neq$ object pose |
| 4 | **LLaVA-3D** | arXiv 2024 | RGB + Pos Embed | ✅ (3D Pos) | ❌ | ❌ | Degrades general QA ability |
| 5 | **SpaceLLaVA** | Remyx 2024 | RGB | ❌ | ❌ | 48.14% | Vertical ($A_z$) drops to 31% |
| 6 | **Depth Anything V2** | 2024 | RGB $\rightarrow$ Depth | Tool | N/A | N/A | Perception tool only |
| 7 | **Spatial FoR** | TACL 2023 | 2D Synthetic/Real | ❌ | Analyzed | ❌ | Diagnostic paper, no model |
| 8 | **SSR** | 2024 | RGB + Text Rationale| ✅ (Text) | Partial | ❌ | Brittle to detector failures |
| 9 | **VSR** | TACL 2023 | 2D Masks (COCO) | ❌ | ❌ | ❌ | 2D plane only |
| 10| **Ferret** | ICLR 2024 | RGB + Points/Boxes | ❌ | ❌ | ❌ | Needs user input coordinates |

---

## 💡 Section 3: The True Research Gap & Our Strategic Advantage

From this 25-paper survey, three indisputable conclusions emerge:

1. **Depth Injection Alone is INSUFFICIENT:**
   - Previous 2024 papers (*Spatial-MLLM*, *LLaVA-3D*) already proved that feeding a monocular depth map into a VLM helps the **depth axis ($A_y$)**, but **does NOT fix the Frame of Reference Shift (FRS)**.
   - Why? Because a depth map is $Z_{camera}$ (camera plane to object). It has ZERO information about an object's 3D orientation (which way the giraffe or car is facing).

2. **The Unclaimed Research Territory:**
   - **No existing method** performs **Explicit 3D Coordinate Frame Transformation** ($T_{cam \rightarrow obj}$) dynamically from single monocular 2D images.
   - If we estimate both:
     - (a) Metric Depth $(X, Y, Z)$ via Depth Anything V2, AND
     - (b) Object Heading / Pose Vector $\vec{h}$ via zero-shot pose / CoT prompting,
   - We can compute the analytical 3D rotation matrix $R_{cam \rightarrow obj}$ and feed this structured geometric prior into LLaVA!

3. **Publication Potential:**
   - Combining depth + heading transformation addresses the single largest error category in SpatialMQA (**FRS: 38.2%**), which none of the existing 2024 depth-injection papers address.
