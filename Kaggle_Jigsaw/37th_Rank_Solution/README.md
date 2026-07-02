# Jigsaw — Agile Community Rules Classification

[Competition link](https://www.kaggle.com/competitions/jigsaw-agile-community-rules)

---

## Competition Overview

**Task:** Given a Reddit comment and a community rule, predict whether the comment violates that rule. This is a binary classification problem where the output is a continuous violation score (higher = more likely to violate).

**Dataset:**

| File | Key columns | Description |
|---|---|---|
| `train.csv` | `body`, `rule`, `subreddit`, `rule_violation` | Labeled Reddit comments (1 = violates, 0 = does not violate) |
| `test.csv` | `row_id`, `rule`, `subreddit`, `positive_example_1/2`, `negative_example_1/2` | Comments to score, plus two labeled examples per rule as context |

A notable feature of the test set is that it includes `positive_example_1/2` (comments that **do** violate the rule) and `negative_example_1/2` (comments that **do not**) for each rule. These labeled examples can be incorporated into training or used as retrieval anchors.

**Evaluation metric:** ROC-AUC (Area Under the Receiver Operating Characteristic Curve).

---

## Our Solution

We built three independent systems and ensembled them. Each notebook in this folder corresponds to one approach.

### 1. Embedding Ensemble (`embedding_ensemble.ipynb`)

Fine-tuned three bi-encoder models on the rule-violation corpus using **triplet loss**:
- `thenlper/gte-large`
- `BAAI/bge-large-en-v1.5`
- `intfloat/e5-large-v2`

**Training:** For each violating comment, 2 triplets were constructed — `(rule text, violating comment, non-violating comment)` — teaching the model that rule-violating comments embed close to the rule they break.

**Scoring:** For each test comment, cosine similarity is computed against the labeled corpus filtered to the same rule. The final score is `Σ(similarity × label)` where labels are `+1` (violating) and `−1` (non-violating). This signed sum is positive when the comment is more similar to violations than to compliant posts.

**Ensemble:** The three models' scores are rank-normalised to `(0, 1)` and averaged equally.

---

### 2. Cross-Encoder Ensemble (`cross_encoder.ipynb`)

Fine-tuned three cross-encoder models that jointly attend to both the rule and the comment in a single forward pass, enabling richer interaction than bi-encoders:
- `google/electra-base-discriminator` (seed 42)
- `FacebookAI/roberta-base` (seed 123)
- `microsoft/deberta-v3-base` (seed 456)

**Training:** Binary `(Rule: ..., Comment: ...)` sentence pairs with `CrossEntropyLoss`. Added **cross-rule negatives** — comments that violate a *different* rule are relabelled as non-violating for the current rule, providing hard negatives that improve rule discrimination. A held-out stratified validation set (50 samples/rule) was used for stacking.

**Ensemble:** 13 stacking strategies were evaluated on the validation set (rank average, optimised weights, logistic regression variants, random forest, gradient boosting, neural network). The best-performing strategy by validation ROC-AUC was selected automatically.

---

### 3. LLM Fine-tuning with LoRA (`llm_finetuning.ipynb`)

Fine-tuned `Qwen2.5-7B-Instruct` (GPTQ INT4) with **LoRA** on a sampled 15% of the corpus, then ran constrained inference via vLLM.

**Training:** SFT on `(prompt, "Yes"/"No")` pairs using `completion_only_loss`. Distributed across 2 GPUs with DeepSpeed ZeRO-2 for memory efficiency.

**Inference:** `MultipleChoiceLogitsProcessor` constrains generation to exactly `"Yes"` or `"No"`, and the log-probability of `"Yes"` is used as the continuous violation score — no threshold tuning required.

**Key insight explored:** The test set's `positive_example_1/2` and `negative_example_1/2` columns contain *labeled* examples that can be added to training data, effectively mining supervision signal from the test set itself. The notebook discusses the grey area this creates with respect to competition rules.

---

## Learnings from Top Submissions

*To be filled.*
