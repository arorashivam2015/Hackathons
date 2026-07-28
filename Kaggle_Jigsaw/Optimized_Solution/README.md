# Final Ensemble — Model Comparison

`Master_Ensemble_v1.ipynb` combines all seven approaches below into a single pipeline: a job allocator schedules each approach's training and inference across Kaggle's 2×T4 GPUs (either **automatically**, for minimum wall time, or via a **manual** GPU assignment), then blends every approach's per-rule ranked predictions using a configurable **per-approach weight**. All run times listed below are measured wall-clock times on that 2×T4 environment.

## Best Submission

The highest-scoring configuration ran **all seven approaches**, blended with the per-approach seed counts and weights below:

| Approach | Seeds | Weight (top submission) |
|---|:---:|:---:|
| lama3-8b-INSTRU | 3 | 0.20 |
| QWEN3-8b | 3 | 0.20 |
| QWEN3-4b-INSTRU | 1 | 0.10 |
| QWEN2.5-7b-INSTRU | 1 | 0.10 |
| llama3.2-3b-INSTRU | 1 | 0.10 |
| Embedding | 3 | 0.15 |
| Cross-Encoder | 3 | 0.15 |
| **Total** | | **1.00** |

**PBL: 0.93303 · PVT: 0.92853** — higher than any individual approach (best single model is 0.926 PBL). The **seed count** controls how stable each approach's own averaged prediction is; the **weight** controls how much that approach counts in the final blend. In shorthand, `3 (0.20)` means 3 seeds at a blend weight of 0.20.

## Per-Approach Comparison

| Approach | PBL Score | PVT Score | GPUs (Training) | GPUs (Inference) | Training Time (per seed) | Inference Time (per seed) |
|---|---|---|---|---|---|---|
| lama3-8b-INSTRU | 0.926 | 0.921 | 1 | 2 | ~48 min/seed | ~25 min/seed |
| QWEN3-8b | 0.924 | 0.921 | 1 | 2 | ~48 min/seed | ~25 min/seed |
| QWEN3-4b-INSTRU | 0.924 | 0.918 | 1 | 1 | ~26 min/seed | ~24 min/seed |
| QWEN2.5-7b-INSTRU | 0.923 | 0.919 | 1 | 2 | ~48 min/seed | ~25 min/seed |
| llama3.2-3b-INSTRU | 0.923 | 0.917 | 1 | 1 | ~26 min/seed | ~24 min/seed |
| Embedding | 0.916 | 0.910 | 1 | 1 | ~7.5 min/model | ~7.5 min/model |
| Cross-Encoder | 0.912 | 0.904 | 1 | 1 | ~5 min/model | ~5 min/model |

## How the ensemble is combined

- Each approach first averages its own seeds/members into one per-rule ranked score, then those per-approach scores are combined using the weights above — so adding a seed improves an approach's stability without changing its influence in the blend unless you also change its weight.
- Scoring is per-rule AUC: every approach ranks rows **within each rule** and rescales the rank to `[0, 1]` before combining, so the ensemble averages calibrated per-rule ranks rather than raw probabilities that may be scaled differently model to model.

## Notes

- **GPUs (Training)** is 1 for every approach — none use multi-GPU training (e.g. DeepSpeed); each seed/model trains on a single pinned GPU.
- **GPUs (Inference) = 2** (lama3-8b-INSTRU, QWEN3-8b, QWEN2.5-7b-INSTRU): these use vLLM tensor parallelism (`tensor_parallel_size=num_gpus`) — the unquantized model is split across both GPUs per seed, run sequentially seed by seed (train on GPU 0, then infer split across both GPUs).
- **GPUs (Inference) = 1** (QWEN3-4b-INSTRU, llama3.2-3b-INSTRU): these use the per-seed-per-GPU scheduler (`tensor_parallel_size=1`) — each seed trains and infers independently on its own single GPU, with seeds cycling through the 2 available GPUs in parallel batches.
- **Embedding**: a different (non-LLM) pipeline — a sentence-embedding model fine-tuned per member (m1/m2/m3), each job pinned to one GPU for both training and inference (true independent parallelism, not DataParallel — DataParallel was tried and found slower due to cross-GPU sync overhead for a workload this small), with the three members' predictions blended (simple average) at the end.
- **Cross-Encoder**: three different base cross-encoder models (m1/m2/m3), each fine-tuned once with its own seed, one GPU each, run in parallel and blended by per-rule rank.
- The LLM approaches support up to 3 seeds (`SEEDS = [1010, 2020, 3030]`); the top submission used 3 seeds for the two strongest models and 1 for the rest. Embedding and Cross-Encoder use 3 independently trained members (m1, m2, m3) as their ensemble members.
