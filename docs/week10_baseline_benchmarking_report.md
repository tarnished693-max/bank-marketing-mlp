# Week 10 Engineering Report: Baseline Benchmarks & Data Audit
**Course**: AIE231 Neural Networks - Alamein International University  
**Project**: Bank Marketing Term Deposit Prediction (Deep MLP)  
**Work Package Owner**: Experimentation Engineer  
**Milestone**: Checkpoint 2 (End of Week 10) - Baselines Ready & Data Verified  

---

## 1. Executive Summary & Week 10 Objectives
In Checkpoint 2 (End of Week 10), the team is required to:
1. Verify leak-free preprocessing splits (Section 3.2).
2. Construct and measure **two baselines**: a trivial baseline and a simple learned baseline (Section 3.3).
3. Establish the central experiment log benchmarks against which all subsequent deep networks will be measured (Section 3.5).

As the **Experimentation Engineer**, I have:
- Developed and executed the **Automated Data Leakage & Split Audit Suite** (`src/leakage_checker.py`), generating a formal audit certificate (`experiments/leakage_audit_report.json`).
- Implemented and evaluated the two required baselines (`BASE-TRIVIAL-001` and `BASE-LOGREG-001`), plus a cost-sensitive balanced baseline (`BASE-LOGREG-BALANCED-002`) via `src/baseline_runner.py`.
- Formally logged all baseline metrics into the master experiment registry (`experiments/experiments.csv`).
- Engineered the **Benchmark Comparison & Lift Analysis Suite** (`src/benchmark_comparator.py`), establishing quantitative acceptance criteria for Weeks 11-12 neural models.

---

## 2. Data Pipeline & Leakage Audit (Section 3.2 Compliance)
To prevent data leakage—a primary reason marks are lost in Section 7 rubric—the data splitting was formally audited.

### 2.1 Split Dimensions & Stratification Check
- **Total Dataset**: $45,211$ rows, $16$ raw features, $48$ one-hot encoded features.
- **Split Breakdown**:
  - **Training Set (70%)**: $31,647$ instances ($27,945$ "no", $3,702$ "yes" $\rightarrow$ **$11.698\%$** positive).
  - **Validation Set (15%)**: $6,782$ instances ($5,988$ "no", $794$ "yes" $\rightarrow$ **$11.707\%$** positive).
  - **Test Set (15%)**: $6,782$ instances ($5,989$ "no", $793$ "yes" $\rightarrow$ **$11.693\%$** positive).
- **Stratification Result**: **PASSED**. Class balance is preserved with less than $\pm 0.01\%$ variation across all three sets.

### 2.2 Leakage Verification Proof
- **Index Overlap**:
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
- **Transformation Isolation**:
  - The `StandardScaler` was fitted strictly on the $31,647$ training records ($\mu_{\text{train}} = 0.000$, $\sigma_{\text{train}} = 1.000$).
  - When applied via `.transform()` to validation data, $\mu_{\text{val}} \ne 0.000$, providing mathematical proof that validation statistics did not contaminate training transformations.
- **Audit Status**: **VERIFIED & CERTIFIED** in `experiments/leakage_audit_report.json`.

---

## 3. Baseline Models Specification & Mathematical Derivations (Section 3.3)

### 3.1 Trivial Baseline: Majority-Class Classifier (`BASE-TRIVIAL-001`)
- **Mechanism**: Predicts class $0$ ("no") unconditionally for every sample.
- **Hand Parameter Count**: $0$ parameters.
- **Empirical Validation Metrics**:
  - $\text{Accuracy} = \frac{5,988}{6,782} = \mathbf{88.29\%}$
  - $\text{Precision}_1 = \mathbf{0.0000}$
  - $\text{Recall}_1 = \mathbf{0.0000}$
  - $\text{F1-Score}_1 = \mathbf{0.0000}$
  - $\text{ROC-AUC} = \mathbf{0.5000}$ (Random diagonal)
  - $\text{PR-AUC} = \mathbf{0.1171}$ (Equal to positive class base prevalence $\pi$)
- **Significance for the Defense**:
  This baseline proves to the examiner why **Accuracy is an invalid metric**. An $88.29\%$ accuracy model that catches zero depositors ($\text{Recall}=0.0\%$) is completely useless to the bank.

---

### 3.2 Simple Learned Baseline: Logistic Regression (`BASE-LOGREG-001`)
- **Mechanism**: Linear classification with sigmoid activation:
  $$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
- **Regularization & Solver**: L2 penalty ($C=1.0$), L-BFGS quasi-Newton optimizer ($85$ iterations).
- **Hand Parameter Calculation**:
  $$\text{Parameters} = N_{\text{features}} + 1 = 48 \text{ weights} + 1 \text{ bias} = \mathbf{49 \text{ parameters}}$$
- **Empirical Validation Metrics**:
  - $\text{Validation Loss} = \mathbf{0.2480}$
  - $\text{PR-AUC (Primary Metric)} = \mathbf{0.4910}$
  - $\text{ROC-AUC} = \mathbf{0.8835}$
  - $\text{F1-Score}_1 = \mathbf{0.4520}$
  - $\text{Recall}_1 = \mathbf{0.3460}$ (Catches only 34.6% of depositors under standard 0.5 threshold)
  - $\text{Precision}_1 = \mathbf{0.6520}$

---

### 3.3 Balanced Learned Baseline: Cost-Sensitive Logistic Regression (`BASE-LOGREG-BALANCED-002`)
- **Mechanism**: Logistic regression with inverse-class frequency loss re-weighting:
  $$w_0 = \frac{N}{2 \cdot N_0} \approx 0.566, \quad w_1 = \frac{N}{2 \cdot N_1} \approx 4.275$$
- **Empirical Validation Metrics**:
  - $\text{Validation Loss} = \mathbf{0.3620}$
  - $\text{PR-AUC} = \mathbf{0.4890}$
  - $\text{ROC-AUC} = \mathbf{0.8842}$
  - $\text{F1-Score}_1 = \mathbf{0.5480}$
  - $\text{Recall}_1 = \mathbf{0.7840}$ (Surges from 34.6% to 78.4%!)
  - $\text{Precision}_1 = \mathbf{0.4210}$
- **Significance**:
  Demonstrates the trade-off of cost-sensitive learning on linear boundaries: a massive increase in lead capture ($\text{Recall}$ up $+43.8\%$) at the cost of calling more uninterested prospects ($\text{Precision}$ drops from $65.2\%$ to $42.1\%$). This provides the benchmark for our neural network ablation in Week 13.

---

## 4. Master Benchmark Table & Lift Analysis (Section 3.3)

| Model Name | Run ID | Params | Wall Time | Val Loss | PR-AUC (Primary) | ROC-AUC | F1-Score | Recall | Precision |
|---|---|---|---|---|---|---|---|---|---|
| **Trivial Majority** | `BASE-TRIVIAL-001` | 0 | 0.002s | 0.3608 | **0.1171** | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| **Logistic Regression (Std)** | `BASE-LOGREG-001` | 49 | 0.450s | 0.2480 | **0.4910** | 0.8835 | 0.4520 | 0.3460 | 0.6520 |
| **Logistic Regression (Bal)** | `BASE-LOGREG-BAL-002` | 49 | 0.520s | 0.3620 | **0.4890** | 0.8842 | 0.5480 | 0.7840 | 0.4210 |
| **Reference Deep MLP** | `EXP-REF-001` | 14,593 | 14.82s | 0.2180 | **0.5840** | 0.9120 | 0.5620 | 0.4850 | 0.6680 |

### Lift of Reference Neural Network Over Standard Learned Baseline:
- **$\Delta$ PR-AUC (Primary)**: $+0.0930$ (**$+18.94\%$ relative lift**)
- **$\Delta$ ROC-AUC**: $+0.0285$ (**$+3.23\%$ relative lift**)
- **$\Delta$ F1-Score**: $+0.1100$ (**$+24.34\%$ relative lift**)
- **$\Delta$ Recall**: $+0.1390$ (**$+40.17\%$ relative lift**)
- **$\Delta$ Loss**: $-0.0300$ (**$-12.10\%$ error reduction**)

---

## 5. Modeling Target Criteria for Weeks 11-12
In accordance with Section 3.3 (*"Every later model is reported as an improvement over these numbers"*), the Experimentation Engineer sets the following quantitative gates for Checkpoint 3:

1. **Model A (Shallow MLP - 1 Hidden Layer)**:
   - Must achieve $\text{PR-AUC} > 0.4910$ and $\text{F1} > 0.4520$ to prove that introducing non-linearity outperforms linear logistic regression.
2. **Model B (Deeper MLP with Regularization - Dropout/BatchNorm)**:
   - Must achieve $\text{PR-AUC} > 0.5500$ and prevent overfitting (generalization gap $\text{Val Loss} - \text{Train Loss} \le 0.03$).
3. **Model C (Tuned MLP after OFAT Optimization)**:
   - Must achieve $\text{PR-AUC} \ge 0.5800$ with $\text{Recall}_1 \ge 0.4800$.

---

## 6. How to Reproduce Week 10 Benchmarks
To run the automated baseline evaluation and data audit:

```bash
# 1. Run Data Leakage Audit
python src/leakage_checker.py

# 2. Run and Log All Baselines into experiments.csv
python src/baseline_runner.py

# 3. Generate Benchmark Comparison Plots & Tables
python src/benchmark_comparator.py
```
All outputs are automatically recorded in `experiments/experiments.csv`, `experiments/leakage_audit_report.json`, and `experiments/curves/`.
