# Empirical Evaluation of Router Overhead in Multi-Tier IDK Cascades on Edge Devices

This repository contains the complete experimental workflow, notebook, and Python evaluation scripts for benchmarking **"I Don't Know" (IDK) cascade architectures** on resource-constrained embedded edge hardware.

IDK cascades defer ambiguous samples from an efficient base model to a larger expert model to balance accuracy and latency. While traditional cascade research frequently treats the routing step as zero-cost, this benchmark measures **router execution overhead** directly on an embedded deployment target and evaluates the trade-offs between two-tier and three-tier cascade topologies across five distinct router families.

---

## Methodology

### 1. Model Hierarchy
The cascade leverages three ResNet variants pre-trained on CIFAR-10 and CIFAR-100 (via `chenyaofo/pytorch-cifar-models`):
* **Model A (Fast):** ResNet-20 ($15.12 \pm 1.94$ ms)
* **Model B (Intermediate):** ResNet-32 ($23.60 \pm 4.05$ ms)
* **Model C (Expert):** ResNet-56 ($39.35 \pm 2.67$ ms)

### 2. Telemetry Feature Extraction
Routers make escalation decisions using 3 lightweight telemetry features computed directly from the model output probabilities:
1. **Confidence:** $\max_k P(y=k \mid x)$
2. **Entropy:** $-\sum_k P(y=k \mid x) \log(P(y=k \mid x) + \epsilon)$
3. **Margin:** Difference between the top-1 and top-2 softmax probabilities

An **Early Exit Threshold** ($\tau_{\text{early}} = 0.90$) allows highly confident predictions to exit immediately after Model A without invoking downstream routers or fallback models.

### 3. Evaluated Router Families
Five router algorithms were trained using cost-sensitive positive class weights scaled by the latency penalty of escalating to downstream models:
* **XGBoost (XGB):** Gradient boosted decision trees ($n_{\text{estimators}}=50$, $\text{max\_depth}=3$).
* **Random Forest (RF):** Ensemble of bagged trees ($n_{\text{estimators}}=50$, $\text{max\_depth}=3$).
* **Logistic Regression (LogReg):** Linear classification model.
* **Decision Tree (DTree):** Single decision tree ($\text{max\_depth}=3$).
* **Genetic Algorithm (GA) Heuristic:** 4-parameter linear threshold router optimized via genetic search over 25 restarts.

### 4. Cascade Topologies
* **Two-Tier ($A \to C$):** Model A infers first; if confidence $< 0.90$, the router decides whether to escalate directly to Model C.
* **Three-Tier ($A \to B \to C$):** Sub-router $R_{AB}$ determines whether to escalate from Model A to Model B. If escalated, sub-router $R_{BC}$ decides whether further escalation to Model C is necessary.

---

## Experimental Results

All inference latencies and per-sample router overhead timings were profiled under real execution conditions on the embedded hardware.

### 1. Isolated Router Execution Overhead
Per-sample isolated `predict_proba` latencies measured on-device using 200 telemetry probe samples:

| Router Architecture | Overhead (Mean $\pm$ Std ms) | Architectural Profile |
| :--- | :--- | :--- |
| **GA Heuristic** | $0.0579 \pm 0.0066$ ms | Pure vector dot-product; negligible cost. |
| **Decision Tree** | $0.7549 \pm 0.0655$ ms | Ultra-low depth tree traversal. |
| **Logistic Regression** | $0.8418 \pm 0.0371$ ms | Single linear boundary computation. |
| **XGBoost** | $1.1845 \pm 0.1962$ ms | Fast sequential tree inference. |
| **Random Forest** | $38.5730 \pm 14.1894$ ms | Heavy parallel CPU tree evaluation. |

> **Key Finding:** Random Forest routers completely negate cascade benefits on edge hardware due to their CPU tree ensemble traversal overhead ($38.57$ ms), which nearly matches the inference cost of Model C itself ($39.35$ ms).

### 2. Two-Tier Cascade Performance (8 Seeds, CIFAR-10 & CIFAR-100)
Averaged across 16 experimental runs per method:

| Method | Accuracy (%) | End-to-End Latency (ms) | Router Overhead (ms) |
| :--- | :--- | :--- | :--- |
| **Pure Expert (Model C)** | $83.51 \pm 10.84$ | $39.35 \pm 0.00$ | — |
| **Naive Baseline ($\tau=0.80$)** | $83.80 \pm 10.08$ | $24.24 \pm 6.68$ | — |
| **GA Heuristic** | $81.29 \pm 11.30$ | **$15.36 \pm 0.28$** | $0.06$ |
| **Logistic Regression** | $83.08 \pm 10.67$ | $21.03 \pm 3.25$ | $0.84$ |
| **XGBoost** | **$83.94 \pm 10.39$** | $23.40 \pm 4.06$ | $1.18$ |
| **Decision Tree** | $83.90 \pm 10.28$ | $23.46 \pm 4.09$ | $0.75$ |
| **Random Forest** | $84.00 \pm 10.19$ | $61.58 \pm 4.75$ | $38.57$ |

### 3. Extension: Two-Tier ($A \to C$) vs. Three-Tier ($A \to B \to C$)

| Method Family | Topology | Accuracy (%) | Latency (ms) |
| :--- | :--- | :--- | :--- |
| **XGBoost** | 2-Tier ($A \to C$) | $84.08 \pm 10.27$ | $23.59 \pm 3.97$ |
| | 3-Tier ($A \to B \to C$) | **$84.29 \pm 9.93$** | **$21.34 \pm 3.08$** |
| **Decision Tree** | 2-Tier ($A \to C$) | $83.95 \pm 10.42$ | $23.75 \pm 4.66$ |
| | 3-Tier ($A \to B \to C$) | $84.22 \pm 9.92$ | **$21.22 \pm 4.03$** |
| **Logistic Regression** | 2-Tier ($A \to C$) | $83.49 \pm 10.39$ | $20.94 \pm 3.42$ |
| | 3-Tier ($A \to B \to C$) | $83.45 \pm 10.15$ | **$19.71 \pm 2.70$** |
| **Pure Threshold** | 2-Tier ($A \to C$) | $84.19 \pm 10.29$ | $26.77 \pm 7.79$ |
| | 3-Tier ($A \to B \to C$) | $84.49 \pm 9.75$ | $27.08 \pm 9.49$ |

Integrating an intermediate model (Model B) creates an additional step-down tier that reduces overall latency for efficient routers (e.g., XGBoost latency decreases from $23.59$ ms to $21.34$ ms while boosting accuracy) because many ambiguous samples resolve successfully at Model B without escalating to Model C.

---

## Repository Structure

```text
├── train_routers.ipynb                 # Notebook for training & exporting router weights
├── idk_cascade_jetson_nano_edition.py  # On-device hardware evaluation and profiling script
└── README.md
