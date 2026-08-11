# IDK Cascade Meta-Router Study

## Overview

This project studies **IDK (I-Don't-Know) cascades** — a technique for making inference cheaper by
running a small, fast model first (Model A) and only escalating to a large, accurate model (Model C)
when Model A signals low confidence. The escalation decision is made by a small **meta-router** classifier
trained on Model A's own output telemetry (confidence, entropy, margin between top-2 class probabilities).

The core question this project investigates: **does the choice of meta-router matter for the cascade's
end-to-end latency, or is the router "free" compared to the cost of the underlying models?**

## Setup

- **Base cascade**: ResNet20 (Model A, fast/cheap) → ResNet56 (Model C, slow/accurate), pretrained on
  CIFAR-10 and CIFAR-100 (`chenyaofo/pytorch-cifar-models`).
- **Router features**: `confidence`, `entropy`, `margin` — all derived from Model A's softmax output on
  each sample, no extra forward pass required.
- **Router families compared**: XGBoost, Random Forest, Logistic Regression, and a single shallow
  Decision Tree — all trained with the same cost-sensitive class weighting so no router is unfairly
  tuned relative to the others.
- **Evaluation**: 8 random seeds × 2 datasets (CIFAR-10, CIFAR-100), 4,000 samples per seed, holding a
  fixed escalation threshold (`theta = 0.20`) and early-exit threshold (`0.90`) constant across all
  router families for a fair comparison.
- **Baselines**: "Pure Expert" (always run Model C, no cascade) and a "Naive Threshold" cascade
  (escalate purely on a raw confidence cutoff, no learned router at all).

## Key Finding: router overhead can erase the entire benefit of cascading

Early results only measured cascade latency as `LATENCY_A + LATENCY_C`, which **excluded the router's
own inference cost** entirely. Once that overhead is correctly folded into the end-to-end latency:

| Metric | XGBoost | Random Forest |
|---|---|---|
| Router overhead alone | 0.465 ± 0.059 ms/sample | 3.258 ± 0.227 ms/sample |
| End-to-end cascade latency | 4.635 ± 0.769 ms | 7.563 ± 0.866 ms |
| Accuracy | 83.48 ± 10.76% | 83.59 ± 10.65% |

For reference, the "Pure Expert" baseline (always running Model C, no cascade at all) costs **6.831 ms**.

**The Random Forest cascade (7.563 ms) ends up slower than never cascading at all.** Its own router
overhead is large enough that the "cheap path" (Model A + router, before ever considering Model C) already
costs nearly as much as the full expert model — so once you add the cost of the samples that *do* escalate,
the total exceeds the no-cascade baseline. XGBoost's much smaller overhead (7x less) means its cheap path
stays meaningfully below the expert model's cost, so the cascade's savings survive.

Accuracy between XGBoost and Random Forest is statistically indistinguishable (paired t-test / Wilcoxon,
not significant) — this is a **pure latency effect**, not an accuracy/latency tradeoff. The latency gap
itself is highly significant: **p = 0.0078** (Wilcoxon signed-rank, the smallest achievable p-value at
n=8 seeds) on both CIFAR-10 and CIFAR-100 — Random Forest was slower than XGBoost on every single seed,
with zero exceptions.

## Why the overhead gap exists

Profiling `predict_proba` on both router types (same hyperparameters: `n_estimators=50, max_depth=3`)
shows the gap is **not** primarily about tree-evaluation cost. XGBoost's booster evaluates all 50 trees in
one compiled call. Random Forest's overhead is dominated by sklearn's ensemble dispatch machinery —
`joblib.Parallel` call overhead plus a `check_is_fitted` / `warnings.filterwarnings` check run on *every
individual tree, every single call*. This is an implementation-level inefficiency in scikit-learn's
`RandomForestClassifier`, not an inherent property of Random Forest as an algorithm.

## Generalizing beyond XGBoost vs. Random Forest

The notebook extends the comparison to a full **router family registry** (`ROUTER_REGISTRY`), so any new
router type can be added by writing one `train_*_router` function and adding it to the dict — every
downstream step (overhead timing, multi-seed trial, pairwise significance tests, Pareto plot) picks it up
automatically. This run adds:

- **Logistic Regression** — the simplest possible router, useful for checking whether router complexity
  buys anything at all.
- **Decision Tree** (single tree, no ensembling) — isolates whether *ensembling itself* (bagging or
  boosting over many trees) is the source of overhead, independent of tree count.

*(Fill in this section with the actual Logistic Regression / Decision Tree numbers once you've run the
updated notebook — the registry-based structure means the Summary Performance and Pareto sections will
report them automatically alongside XGBoost and Random Forest.)*

## Total Computation Time

The notebook reports both:
1. **Total pipeline wall-clock time** — the full time to train and evaluate every router family, across
   all seeds and both datasets (Section 10 of the notebook).
2. **Per-router-family cumulative compute time** — how much of that total each router family's
   training+evaluation consumed, useful for distinguishing "cheap to run at inference time" (per-sample
   latency) from "cheap to train and evaluate at experiment time" (total compute).

*(Fill in the actual numbers here after running — e.g. "Total pipeline wall-clock: X minutes; XGBoost:
Y% of router compute time, Random Forest: Z%, ...")*

## Repository Structure

- `idk_cascade_full_study.ipynb` — main notebook: dataset/model setup, router training, overhead
  profiling, multi-seed evaluation across all router families, significance testing, Pareto plot, and
  total compute time reporting.

## How to Run

1. Open `idk_cascade_full_study.ipynb` in Colab (GPU runtime recommended — CIFAR-100 with ResNet56 across
   8 seeds × 2 datasets × 4 router families is compute-heavy).
2. `Runtime → Run all`. The notebook is ordered so every function is defined before it's used — if you
   edit cell order, keep `measure_router_overhead_per_sample` (Section 4) before `evaluate_router_cascade`
   (Section 5), which depends on it.
3. Results land in `trial_df`; the Pareto plot saves to `router_family_pareto.png`.

## Status

Work in progress. Advisor feedback (Aug 2026) requested: (1) reporting total computation time alongside
per-sample latency — added in Section 10; (2) broadening the comparison beyond a single router swap to a
set of router families across datasets — added via `ROUTER_REGISTRY` and Logistic Regression/Decision
Tree baselines. Next step: rerun the full notebook to get final numbers for all four router families and
update this README's placeholders above.
