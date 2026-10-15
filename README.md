# Empirical Evaluation of Router Overhead in Multi-Tier IDK Cascades on Edge Devices

This repository contains the complete experimental workflow, training notebooks, and hardware profiling scripts for benchmarking **"I Don't Know" (IDK) cascade architectures** on resource-constrained embedded edge hardware (NVIDIA Jetson Nano)[cite: 6, 8].

IDK cascades defer ambiguous samples from an efficient base model to larger expert models to balance accuracy, average latency, and worst-case execution time (WCET)[cite: 6, 7]. While traditional cascade research frequently treats the routing step as zero-cost, this benchmark measures **per-sample router execution overhead** directly on-device across **2-stage**, **3-stage**, and **4-stage** cascade topologies across five distinct router families[cite: 6, 7].

---

## Methodology

### 1. Model Hierarchies
Models are pre-trained on CIFAR-10/100 and ImageNet-1K (`IMAGENET1K_V1`), evaluated on CIFAR and ImageNetV2 (`TopImages`)[cite: 2, 8]:

* **CIFAR Benchmark (2-Tier & 3-Tier)[cite: 8]:**
  * **Model A (Fast):** ResNet-20 ($15.12 \pm 1.94$ ms)
  * **Model B (Intermediate):** ResNet-32 ($23.60 \pm 4.05$ ms)
  * **Model C (Expert):** ResNet-56 ($39.35 \pm 2.67$ ms)

* **ImageNetV2 Benchmark (2-, 3-, & 4-Stage Cascades)[cite: 2, 7]:**
  * **Stage A (Base):** ResNet-18[cite: 2, 7]
  * **Stage B (Intermediate-1):** ResNet-34[cite: 2, 7]
  * **Stage C (Intermediate-2):** ResNet-50[cite: 2, 7]
  * **Stage D (Terminal Expert):** ResNet-152[cite: 2, 7]

### 2. Telemetry Feature Extraction & Policy
Routers make escalation decisions using 3 lightweight telemetry features computed directly from model output probabilities[cite: 6, 7]:
1. **Confidence:** $\max_k P(y=k \mid x)$[cite: 7]
2. **Entropy:** $-\sum_k P(y=k \mid x) \log(P(y=k \mid x) + \epsilon)$[cite: 7]
3. **Margin:** Difference between the top-1 and top-2 softmax probabilities[cite: 7]

* **Early Exit Threshold ($\tau = 0.90$):** Allows predictions with confidence $\ge 0.90$ to exit immediately without invoking downstream routers or fallback models[cite: 2, 3, 7].
* **Escalation Threshold ($\theta = 0.20$):** Samples with confidence $< 0.90$ invoke the router; escalation to the next model occurs only when the router's escalation probability exceeds $\theta = 0.20$[cite: 2, 3, 7].
* **Terminal Classifier:** Stage C (for 2/3-stage) or Stage D (for 4-stage) acts as the final terminal classifier yielding unthresholded $\text{argmax}$[cite: 2, 3, 7].

### 3. Evaluated Router Families
Five router algorithms are trained using cost-sensitive positive class weights scaled by the latency penalty of escalating to downstream models[cite: 2, 8]:
* **XGBoost (XGB):** Gradient boosted decision trees (`n_estimators=50`, `max_depth=3`, 2 threads)[cite: 2, 7].
* **Random Forest (RF):** Ensemble of bagged trees (`n_estimators=50`, `max_depth=3`, 2 threads)[cite: 2, 8].
* **Logistic Regression (LogReg):** Linear classification model (`max_iter=1000`)[cite: 2, 8].
* **Decision Tree (DTree):** Single decision tree (`max_depth=3`)[cite: 2, 8].
* **GA Heuristic:** 4-parameter linear threshold router optimized via genetic search over 25 restarts with pure array dot-product execution (no third-party library dispatch)[cite: 2, 8].

### 4. Cascade Topologies
* **2-Stage ($A \to C$):** Model A infers first; if confidence $< 0.90$, sub-router $R_{AC}$ decides whether to escalate directly to Model C[cite: 4, 7].
* **3-Stage ($A \to B \to C$):** Sub-router $R_{AB}$ decides escalation from A to B; if escalated and confidence remains $< 0.90$, $R_{BC}$ decides escalation to C[cite: 4, 7].
* **4-Stage ($A \to B \to C \to D$):** Sequential cascade using sub-routers $R_{AB}$, $R_{BC}$, and $R_{CD}$ to gate escalation across all four tiers[cite: 2, 3, 7].

---

## Experimental Results

All inference latencies and per-sample router overhead timings were profiled on an **NVIDIA Jetson Nano (4GB)**[cite: 6, 8].

### 1. Isolated Router Execution Overhead
Per-sample isolated `predict_proba` latencies measured on-device[cite: 8]:

| Router Architecture | Overhead (Mean $\pm$ Std ms) | Architectural Profile |
| :--- | :--- | :--- |
| **GA Heuristic** | **$0.0579 \pm 0.0066$ ms** | Pure vector dot-product; negligible cost. |
| **Decision Tree** | $0.7549 \pm 0.0655$ ms | Ultra-low depth tree traversal. |
| **Logistic Regression** | $0.8418 \pm 0.0371$ ms | Single linear boundary computation. |
| **XGBoost** | $1.1845 \pm 0.1962$ ms | Fast sequential tree inference; single C++ dispatch[cite: 9]. |
| **Random Forest** | $38.5730 \pm 14.1894$ ms[cite: 8] | Heavy parallel CPU tree evaluation via `joblib.Parallel`[cite: 9]. |

> **Key Finding:** Random Forest routers completely negate cascade benefits on edge hardware due to their CPU tree ensemble traversal overhead ($38.57$ ms), which nearly matches the inference cost of Model C itself on CIFAR ($39.35$ ms)[cite: 8].

### 2. Two-Tier ($A \to C$) vs. Three-Tier ($A \to B \to C$) Benchmark (CIFAR)
Averaged across 16 experimental runs per method (8 seeds, CIFAR-10 & CIFAR-100)[cite: 8]:

| Method Family | Topology | Accuracy (%) | Latency (ms) |
| :--- | :--- | :--- | :--- |
| **Pure Expert (Model C)** | Baseline | $83.51 \pm 10.84$ | $39.35 \pm 0.00$ |
| **Pure Threshold** | 2-Tier ($A \to C$)[cite: 8] | $84.19 \pm 10.29$ | $26.77 \pm 7.79$ |
| | 3-Tier ($A \to B \to C$)[cite: 8] | $84.49 \pm 9.75$[cite: 9] | $27.08 \pm 9.49$ |
| **GA Heuristic** | 2-Tier ($A \to C$)[cite: 8] | $81.29 \pm 11.30$ | **$15.03 \pm 0.28$**[cite: 9] |
| | 3-Tier ($A \to B \to C$)[cite: 8] | $81.35 \pm 11.12$ | **$14.99 \pm 0.25$**[cite: 9] |
| **Logistic Regression** | 2-Tier ($A \to C$)[cite: 8] | $83.49 \pm 10.39$ | $20.94 \pm 3.42$ |
| | 3-Tier ($A \to B \to C$)[cite: 8] | $83.45 \pm 10.15$ | **$19.71 \pm 2.70$** |
| **Decision Tree** | 2-Tier ($A \to C$)[cite: 8] | $83.95 \pm 10.42$ | $23.75 \pm 4.66$ |
| | 3-Tier ($A \to B \to C$)[cite: 8] | $84.22 \pm 9.92$ | **$21.22 \pm 4.03$** |
| **XGBoost** | 2-Tier ($A \to C$)[cite: 8] | $84.08 \pm 10.27$[cite: 9] | $23.59 \pm 3.97$ |
| | 3-Tier ($A \to B \to C$)[cite: 8] | **$84.29 \pm 9.93$**[cite: 9] | **$21.34 \pm 3.08$** |
| **Random Forest** | 2-Tier ($A \to C$)[cite: 8] | $84.08 \pm 10.35$[cite: 9] | $61.94 \pm 4.75$[cite: 9] |
| | 3-Tier ($A \to B \to C$)[cite: 8] | $84.48 \pm 9.73$[cite: 9] | $65.19 \pm 4.88$[cite: 9] |

### 3. Four-Stage Cascade Performance (ImageNetV2-TopImages)
Evaluated across 8 seeds on ResNet-18 $\to$ ResNet-34 $\to$ ResNet-50 $\to$ ResNet-152[cite: 2, 3]:

* **Accuracy vs. Mean Latency:** 
  * **GA Heuristic:** Delivers the fastest average inference latency ($\approx 576$ ms) due to negligible router overhead and early exiting at Stage A for $\sim 63.5\%$ of samples[cite: 9].
  * **Tree Ensembles (XGBoost / Random Forest):** Maintain top-tier accuracy ($\approx 80.0\%$), answering $\sim 47.5\%$ at Stage A, $\sim 14.8\%$ at Stage B, $\sim 9.5\%$ at Stage C, and delegating $\sim 28.2\%$ to Stage D[cite: 9].
* **Worst-Case Execution Time (WCET):**
  * When ambiguous inputs cascade through the full escalation path ($A \to B \to C \to D$), observed WCET reaches $\approx 2619$ ms ($\sim 1.7\times$ the latency of executing Stage D alone)[cite: 3, 9].
  * Analytical WCET bounds strictly account for stage latencies plus cumulative router overhead: $\text{WCET}_{\text{bound}} = \sum_{k} L_{\max}(k) + 3 \times O_{\text{router}}^{\max}$[cite: 3].

---

## Repository Structure

```text
├── train_routers (3).ipynb               # Colab/GPU notebook to train & export 4-stage routers
├── idk_cascade_4stage_jetson_nano.py     # On-device hardware evaluation script for 4-stage cascades
├── idk_cascade_2_3stage.py               # On-device benchmark script for 2- and 3-stage cascades
└── README.md
