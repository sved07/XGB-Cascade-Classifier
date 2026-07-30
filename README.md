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

## 📓 Notebook Summary

* **`01_data_preprocessing.ipynb`**: Loads, cleans, and structures dataset splits for multi-stage training.
* **`02_model_training.ipynb`**: Fits stage-wise XGBoost models and evaluates standalone baselines.
* **`03_threshold_calibration.ipynb`**: Sweeps confidence bounds ($\tau_1, \tau_2$) to construct IDK vs. Accuracy trade-off curves.
* **`04_benchmark_eval.ipynb`**: Performs latency benchmarks, compute cost analysis, and visualizations comparing Single Model vs. Cascade System.
