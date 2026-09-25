# Natural Language → SQL Analytics Agent

A fine-tuned language model that translates natural-language questions into executable SQL queries, with query validation, caching, and a deployable API service.

## Overview

This project fine-tunes CodeT5-base using LoRA (parameter-efficient fine-tuning) to translate natural-language questions into SQL, evaluates it with multiple accuracy metrics, adds production-style guardrails and caching, and packages it as a containerized API service with a live demo.

Built under real constraints: free-tier compute only (Kaggle T4 GPU), limited development time, and a goal of achieving solid — not necessarily state-of-the-art — accuracy.

## Architecture

```
Question + Table Schema
        │
        ▼
┌────────────────────┐
│  BentoML Service    │ ← fine-tuned CodeT5-base + LoRA adapter
│  (service.py)       │ ← query validation / guardrails
│                      │ ← in-memory cache
└────────────────────┘
        │
        ▼
   Generated SQL
        │
        ├──► Gradio Demo (executes against sample tables, shows results)
        └──► FastAPI client (demonstrates a web app consuming the model API)
```

## Tech Stack

- **Model:** CodeT5-base, fine-tuned with Hugging Face Transformers + PEFT (LoRA)
- **Dataset:** WikiSQL
- **Experiment tracking:** Weights & Biases
- **Model registry:** Hugging Face Hub
- **Serving:** BentoML, containerized with Docker
- **Demo:** Gradio
- **Compute:** Kaggle Notebooks (free-tier T4 GPU)

## Dataset Choice: WikiSQL

Considered Spider, BIRD, ATIS, and WikiSQL. Given free-tier compute and limited development time, WikiSQL (single-table queries, no joins) was chosen over the harder multi-table benchmarks (Spider, BIRD) to keep training tractable while still yielding strong, reportable accuracy. ATIS was ruled out since it was the dataset used by a comparable reference project, and WikiSQL offered better differentiation plus a larger example count.

WikiSQL's SQL data required reconstruction: the dataset's `human_readable` SQL field is inconsistently formatted (unquoted multi-word columns, unquoted string literals). A custom `reconstruct_sql` function was built to generate properly quoted, type-aware SQL from the dataset's structured fields (`sel`, `agg`, `conds`), using each column's declared type (`text` vs `real`) rather than guessing from value appearance.

## Model & Fine-Tuning

**Base model:** CodeT5-base (~220M parameters) — chosen over plain T5-base for its code-pretraining, a closer match to SQL generation.

**Method:** LoRA (rank 16, alpha 32, targeting attention query/value projections) — trains ~0.79% of total parameters, making fine-tuning feasible on a free-tier GPU.

**Two training runs were conducted and compared:**

| | Run 1 | Run 2 |
|---|---|---|
| Epochs | 3 | 7 |
| Batch size | 8 | 8 |
| Learning rate | 2e-4 | 2e-4 |
| Final validation loss | 0.1018 | 0.0859 |

Run 2 was trained after observing validation loss was still decreasing at epoch 3 (not plateaued). Extending to 7 epochs — where loss genuinely flattened — improved every downstream metric, confirming the hypothesis.

## Evaluation Results (Run 2, final model)

| Metric | Score |
|---|---|
| Strict exact-match accuracy | 63.47% |
| Case-insensitive exact-match accuracy | 68.76% |
| **Execution accuracy** | **78.39%** |

Three metrics are reported deliberately: exact-match string comparison understates true performance, since semantically correct queries can differ in casing, quoting, or clause order. Execution accuracy — actually running generated and reference SQL against real SQLite tables built from each example — captures functional correctness directly, and is the most reliable number.

## Guardrails & Self-Correction

Generated SQL is validated before execution:
- Must be a `SELECT` statement (blocks `INSERT`/`UPDATE`/`DELETE`/`DROP`/etc.)
- No statement chaining
- All referenced columns must exist in the table's actual schema

A self-correction retry loop was also built: on validation failure, the failure reason is appended to the prompt and the model retries. Testing showed the base model does not reliably use this feedback (it was never trained on feedback-conditioned prompts, and can echo words from the retry prompt into the output), so **the retry loop is not used in the production service** — it remains a documented, demonstrated capability with a known limitation, not a deployed feature.

## Caching

An exact-match cache (keyed on normalized question + column set) avoids redundant model calls for repeated queries. Semantic/fuzzy caching was considered but scoped out as a documented future extension.

## Quantization — Scoped Out

8-bit quantization via bitsandbytes was attempted but abandoned after repeated, unresolvable version incompatibilities between `bitsandbytes`, `peft`, and Triton on the Kaggle environment. This was a deliberate engineering trade-off given time constraints — deprioritized after the issue was diagnosed, rather than continuing to chase environment-level conflicts unrelated to model quality.

## Known Limitation: Case Sensitivity

The model is given only column names as schema context, not sample cell values — so it cannot reliably reproduce the exact casing of string literals it was never shown (e.g. generating `'mario volarevic'` instead of `'Mario Volarevic'`). This is a primary driver of the gap between exact-match and execution accuracy. Documented fixes for a future iteration:
- Include sample column values in the training input (requires retraining)
- Post-generation fuzzy-match correction against real column values (no retraining needed)
- Case-insensitive execution (`COLLATE NOCASE`) — implemented in the demo as a pragmatic mitigation

## Project Structure

```
nl-to-sql-project/
├── notebook.ipynb          # data prep, fine-tuning, evaluation, guardrails, caching
├── requirements.txt
├── service/
│   ├── service.py          # BentoML service definition
│   └── bentofile.yaml
└── webapp/
    ├── app.py               # FastAPI client demonstrating model API consumption
    └── requirements.txt
```

## Running the Service

```bash
cd service
pip install bentoml
bentoml build
bentoml containerize nl_to_sql_service:latest
docker run -p 3000:3000 nl_to_sql_service:latest
```
Swagger UI: `http://localhost:3000`

## Running the Demo Web App (FastAPI)

```bash
cd webapp
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```
Swagger UI: `http://localhost:8000/docs`

## Live Demo

[Gradio demo link — pending Hugging Face Spaces deployment]

## Model

Fine-tuned weights: [huggingface.co/siddharth57/codet5-wikisql-run2](https://huggingface.co/siddharth57/codet5-wikisql-run2)