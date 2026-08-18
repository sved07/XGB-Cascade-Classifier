# XGB-Cascade-Classifier: Quantifying Router Overhead in IDK Cascades

Companion notebook for the work-in-progress paper *"Quantifying Router Overhead in IDK Cascade Skip
Decisions"* (Vedula, Cheng, Carroll — University of Houston Real-Time Systems Lab).

## What this is

IDK (I-Don't-Know) cascades reduce average inference cost by routing an input through a fast model first
and escalating to an expensive model only when the fast model's confidence is low. Prior work in this line
of research ([Nguyen, Cheng, Carroll, RTSS 2024]; [Katikaneni, Cheng, Carroll, RTSS 2025]) evaluated a
static confidence threshold and a Random Forest classifier for making that escalation decision, but neither
measured how much the *router itself* costs to run — both treated the routing decision as effectively free
and reported only end-to-end cascade latency.

This notebook measures router inference overhead directly, isolates it from the cost of the models it's
choosing between, and asks whether that overhead can be large enough to matter — across five different
router implementations rather than one.

## Architecture

A two-stage cascade: a lightweight ResNet20 (Model A) and a high-capacity ResNet56 (Model C), evaluated on
CIFAR-10 and CIFAR-100 using pretrained checkpoints from `chenyaofo/pytorch-cifar-models`. For every input,
three telemetry features are extracted from Model A's output — confidence, entropy, and the margin between
its top two predicted probabilities — and passed to a router that decides whether to accept Model A's
answer or escalate to Model C.

## Routers compared

All five are trained identically on the same three features with the same cost-sensitive weighting, so no
router receives preferential tuning:

- **XGBoost**
- **Random Forest**
- **Logistic Regression**
- **Decision Tree** (single shallow tree, no ensembling)
- **GA Heuristic** — a genetic-algorithm-optimized linear decision rule (4 scalar weights) that requires no
  call into scikit-learn or XGBoost at inference time, included specifically to test whether library-call
  overhead can be avoided entirely.

## The corrected latency metric

Prior evaluations report cascade latency as approximately:

```
Latency_naive = Latency_A + p(escalate) × Latency_C
```

which implicitly treats the router's own decision as instantaneous. This notebook instead measures each
router's actual `predict_proba` (or direct evaluation, for the GA Heuristic) wall-clock cost and includes
it explicitly:

```
Latency_corrected = Latency_A + Overhead_router + p(escalate) × Latency_C
```

## Key results (8 seeds × CIFAR-10 + CIFAR-100)

**Router overhead alone (ms/sample):**

| Router | Mean | Std |
|---|---|---|
| GA Heuristic | 0.0125 | 0.0009 |
| Decision Tree | 0.1475 | 0.0118 |
| Logistic Regression | 0.2017 | 0.0085 |
| XGBoost | 0.4000 | 0.0063 |
| Random Forest | 3.2369 | 0.0980 |

**Overhead-corrected cascade latency (ms):**

| Method | Mean | Std |
|---|---|---|
| GA Heuristic | 3.0311 | 0.5698 |
| Logistic Regression | 3.9458 | 0.4835 |
| Decision Tree | 4.1654 | 0.7616 |
| XGBoost | 4.4795 | 0.7661 |
| Random Forest | 7.4282 | 0.8587 |
| Pure Expert (no cascade) | 6.7495 | — |

**Random Forest's cascade is slower on average than never cascading at all.** This is invisible under the
naive latency formula used in prior work — it only appears once router overhead is measured and included.

**The GA Heuristic achieves the lowest overall cascade latency of any method tested**, consistent with its
near-zero overhead — but this is not a free win: it shows a statistically significant accuracy penalty
(3.5–4.4 percentage points lower than every other router on CIFAR-100; a smaller but still significant gap
against XGBoost and Random Forest on CIFAR-10). It's a genuine speed/accuracy trade-off, not a strictly
dominant option.

Accuracy is otherwise statistically indistinguishable between XGBoost and Random Forest on both datasets.
Logistic Regression shows a small but significant accuracy penalty relative to the tree-based routers
specifically on CIFAR-100, not on CIFAR-10.

## Why Random Forest's overhead is so much larger

Profiling `predict_proba` under matched hyperparameters (`n_estimators=50, max_depth=3`) shows XGBoost
evaluates all 50 trees in one compiled call, while Random Forest's overhead is dominated by scikit-learn's
`joblib.Parallel` dispatch machinery and a per-tree `check_is_fitted` / `warnings.filterwarnings` check
executed once for every one of the 50 trees, on every call. This points to a specific implementation detail
in scikit-learn's `RandomForestClassifier`, not to ensembling in general — the GA Heuristic's near-zero
overhead, achieved by skipping library calls entirely, is direct supporting evidence for this.

## Statistical validation

All headline comparisons use paired t-tests and Wilcoxon signed-rank tests across the 8 seeds. The
XGBoost/Random Forest latency gap is significant on both datasets (p = 0.0078, Wilcoxon — the minimum
attainable p-value at n = 8). The near-identical accuracy between XGBoost and Random Forest was verified as
non-coincidental: on a representative seed, the two routers agree on the escalation decision for 99.12% of
samples, and every disagreement occurs on a sample where Model A and Model C agree on correctness anyway.

## Total compute cost

Full pipeline wall-clock time (training and evaluating all five routers, all seeds, both datasets):
**694.26 seconds (11.57 minutes)**. Of the router-specific train+eval time, Random Forest accounts for
76.9%, versus 9.8% (XGBoost), 4.8% (Logistic Regression), 4.6% (GA Heuristic), and 4.0% (Decision Tree) —
the same implementation-level overhead affects training cost, not just inference cost.

## Repository structure

- `idk_cascade_full_study.ipynb` — the complete pipeline: dataset/model setup, router training (all five
  families via a shared `ROUTER_REGISTRY`), overhead profiling, multi-seed evaluation, significance
  testing, and total compute-time reporting.

## Running it

Open in Colab (GPU runtime recommended) and `Runtime → Run all`. Cell order matters:
`measure_router_overhead_per_sample` must execute before `evaluate_router_cascade`, which depends on it —
keep this ordering if you rearrange cells.

## Status

Work in progress. Current limitations, addressed as future work: the cascade studied here is two-stage
(A → C); extending to the three-stage A → B → C architecture used in prior work, and validating results on
embedded hardware (e.g., an NVIDIA Jetson Nano, matching Katikaneni et al.), are the next steps.
