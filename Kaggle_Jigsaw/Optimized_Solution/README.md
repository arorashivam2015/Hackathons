# Final Ensemble — Model Comparison

Run times were provided as one combined figure per seed/model; **Training Time** and **Inference Time** below are that figure split evenly in half (not independently measured).

| Approach | PBL Score | PVT Score | GPUs (Training) | GPUs (Inference) | Training Time (per seed) | Inference Time (per seed) |
|---|---|---|---|---|---|---|
| lama3-8b-INSTRU | 0.926 | 0.921 | 1 | 2 | ~48 min/seed | ~25 min/seed |
| QWEN3-8b | 0.924 | 0.921 | 1 | 2 | ~48 min/seed | ~25 min/seed |
| QWEN3-4b-INSTRU | 0.924 | 0.918 | 1 | 1 | ~26 min/seed | ~24 min/seed |
| QWEN2.5-7b-INSTRU | 0.923 | 0.919 | 1 | 2 | ~48 min/seed | ~25 min/seed |
| llama3.2-3b-INSTRU | 0.923 | 0.917 | 1 | 1 | ~26 min/seed | ~24 min/seed |
| Embedding | 0.916 | 0.910 | 1 | 1 | ~7.5 min/model | ~7.5 min/model |
| Cross-Encoder | 0.912 | 0.904 | 1 | 1 | ~5 min/model | ~5 min/model |

## Notes

- **GPUs (Training)** is 1 for every approach — none of these use multi-GPU training (e.g. DeepSpeed); each seed/model trains on a single pinned GPU.
- **GPUs (Inference) = 2** (lama3-8b-INSTRU, QWEN3-8b, QWEN2.5-7b-INSTRU): these use vLLM tensor parallelism (`tensor_parallel_size=num_gpus`) — the unquantized model is split across both GPUs per seed, run sequentially seed by seed (train on GPU 0, then infer split across both GPUs).
- **GPUs (Inference) = 1** (QWEN3-4b-INSTRU, llama3.2-3b-INSTRU): these use the per-seed-per-GPU scheduler (`tensor_parallel_size=1`) — each of the 3 seeds trains and infers independently on its own single GPU, with seeds cycling through the 2 available GPUs in parallel batches.
- **Embedding**: a different (non-LLM) pipeline — a sentence-embedding model fine-tuned per model (m1/m2/m3), each job pinned to one GPU for both training and inference (true independent parallelism, not DataParallel — DataParallel was tried and found slower due to cross-GPU sync overhead for a workload this small), with the three models' predictions blended (simple average) at the end.
- All score-bearing approaches use 3 seeds (`SEEDS = [1010, 2020, 3030]`); Embedding uses 3 independently trained models (m1, m2, m3) as its ensemble members instead.
