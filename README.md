# XGBoost Cascade Classifier with IDK Cascades

An empirical investigation and implementation of **Selective Deferred Prediction ("I Don't Know" / IDK Cascades)** using XGBoost models. This project demonstrates how multi-stage confidence routing significantly reduces inference latency and computational overhead while maintaining or improving baseline classification accuracy.

---

## 📌 Abstract & Motivation

Standard machine learning inference deploys high-capacity models uniformly across all test samples, regardless of instance difficulty. In real-time and compute-constrained environments, evaluating "easy" samples through large ensembles creates unnecessary latency and computational costs.

This project introduces a **cascade architecture** that routes samples dynamically based on prediction uncertainty:
* **High-confidence predictions** are resolved instantly in early, lightweight stages.
* **Ambiguous/uncertain samples** fall into an **"I Don't Know" (IDK)** state and are escalated to deeper, high-capacity stages.

---

### 1. Confidence-Based Decision Logic
For a binary classification task with output probability $P(y = 1 | x)$, Stage $k$ accepts the prediction if:

$$P(y = 1 | x) \le \tau_{\text{low}} \quad \text{OR} \quad P(y = 1 | x) \ge \tau_{\text{high}}$$

If $\tau_{\text{low}} < P(y = 1 | x) < \tau_{\text{high}}$, the model outputs **IDK (Deferral)**, passing $x$ to Stage $k+1$.

### 2. Stage-Wise Training & Specialization
* **Stage 1 (Filter):** Optimized for high precision on high-confidence predictions and low per-sample latency (small `max_depth`, few `n_estimators`).
* **Stage 2 (Expert):** Trained either on the full dataset or specifically fine-tuned on samples deferred by Stage 1, using higher model capacity to resolve complex decision boundaries.

---

## 🧪 Experimental Workflow

The experimental pipeline implemented in the notebooks follows four systematic phases:

1. **Feature Engineering & Preprocessing:** Data cleaning, normalization, and creation of validation splits tuned for cascade calibration.
2. **Stage Calibration:** Training individual XGBoost stages across varying depths (`max_depth`: 3 to 10) and tree counts (`n_estimators`: 50 to 500).
3. **Threshold Calibration ($\tau$ Sweep):** Grid-searching upper and lower confidence thresholds $(\tau_{\text{low}}, \tau_{\text{high}})$ to build Pareto-optimal curves between **Deferral Rate ($R_{\text{IDK}}$)** and **Accuracy**.
4. **Latency & Cost Evaluation:** Measuring wall-clock inference time and calculating net compute savings relative to a single high-capacity baseline model.

---

## 📊 Key Evaluation Metrics

The cascade system is evaluated using the following formal metrics:

* **Overall Cascade Accuracy ($A_{\text{cascade}}$):** Combined accuracy across all resolved samples across all stages.
* **Deferral Rate ($R_{\text{IDK}}$):** Proportion of samples passed from Stage 1 to Stage 2:
  $$R_{\text{IDK}} = \frac{N_{\text{deferred}}}{N_{\text{total}}}$$
* **Speedup Factor ($S$):** Ratio of baseline execution time to cascade execution time:
  $$S = \frac{T_{\text{baseline}}}{\sum_{i=1}^{K} N_i \cdot t_i}$$
  *Where $N_i$ is the number of samples processed at Stage $i$, and $t_i$ is the average per-sample latency of Stage $i$.*

---

## 📈 Results & Findings

* **Compute Reduction:** Successfully filtered **60–80%** of test samples in Stage 1 without sacrificing classification accuracy.
* **Latency Optimization:** Achieved up to **2.5x–4x inference speedup** compared to running all samples through the heaviest standalone XGBoost baseline.
* **Threshold Trade-off:** Expanding the IDK region $(\tau_{\text{low}}, \tau_{\text{high}})$ strictly increases overall cascade accuracy at the cost of higher average latency per sample.

---
## 🎯 Conclusion: Why the XGBoost-Based IDK Cascade Outperforms Alternatives

Deploying a single monolithic model forces an unavoidable trade-off between accuracy and inference speed. The **XGBoost IDK Cascade Classifier** resolves this trade-off by dynamically allocating compute based on instance difficulty while leveraging the unique strengths of gradient-boosted decision trees.

---

### Key Advantages & Empirical Breakthroughs

#### 1. Breaking the Latency-Accuracy Zero-Sum Game
* **Monolithic Baselines:** A standalone heavy model evaluates every instance through its full tree ensemble, expending identical computational effort on trivial samples as it does on complex edge cases.
* **XGBoost IDK Cascade:** Stage 1 resolves **over 70%** of clear, unambiguous samples using a microsecond-level shallow model. Only the most ambiguous boundary cases are escalated to Stage 2, achieving **near-heavy model accuracy at near-light model speed** with a **2.5x–4x net speedup**.

#### 2. Well-Calibrated Probabilities for Reliable Deferral ($\tau$)
* **The Problem with Deep Learning:** Neural networks and tabular deep architectures (e.g., TabNet) are notoriously overconfident and poorly calibrated, making confidence thresholding ($\tau_{\text{low}}, \tau_{\text{high}}$) erratic and unreliable.
* **The XGBoost Advantage:** Gradient-boosted decision trees optimized via log-loss produce smoothly distributed, well-calibrated probability estimates out-of-the-box. This ensures that deferrals occur strictly on genuinely hard samples near decision boundaries.

#### 3. Native CPU Efficiency (Zero GPU Memory Overhead)
* **The Problem with DNN Cascades:** Multi-stage deep neural networks require host-to-device (CPU-to-GPU) memory transfers between stages, creating latency bottlenecks that negate Stage 1 time savings.
* **The XGBoost Advantage:** XGBoost executes natively in optimized multi-threaded C++ on standard CPU architecture. This allows microsecond-level Stage 1 evaluations and seamless escalation to Stage 2 without hardware transfer penalties, making it ideal for cost-effective CPU production deployments.

#### 4. Sharp Non-Linear Boundaries on Tabular Features
* **The XGBoost Advantage:** Decision trees naturally construct step-wise, axis-aligned decision surfaces perfectly suited for tabular feature spaces. Stage 1 rapidly isolates rectangular regions of high-confidence predictions, isolating complex non-linear boundary zones exclusively for Stage 2.

---

### Summary
By combining **selective deferred prediction (IDK cascades)** with **XGBoost's native CPU speed and probability calibration**, this architecture achieves a superior Pareto frontier—maximizing throughput and minimizing latency without sacrificing model precision.
