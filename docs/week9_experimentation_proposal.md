# Week 9 Project Proposal: Experimentation & Evaluation Strategy
**Course**: AIE231 Neural Networks - Alamein International University  
**Project**: Bank Marketing Term Deposit Subscription Prediction using Deep MLP  
**Work Package Owner**: Experimentation Engineer  
**Milestone**: Checkpoint 1 (End of Week 9) - Topic Settled & Proposal  

---

## 1. Executive Summary & Problem Formulation
In direct marketing campaigns of banking institutions (telemarketing), contacting every single customer is both economically inefficient and causes customer fatigue. The objective is to deploy a neural network classifier (Multi-Layer Perceptron) that predicts whether a client will subscribe to a long-term deposit (`y` = `yes` [1] or `no` [0]) based on 16 client and campaign features.

### Cost of Errors (Asymmetry):
- **False Negative (FN)**: The model predicts "no", but the client was willing to subscribe. The bank loses a high-value customer deposit and substantial lifetime customer value. **This is a high-cost error.**
- **False Positive (FP)**: The model predicts "yes", but the client declines. The cost is limited to the operational cost of a short phone call (~2-3 minutes of agent time). **This is a low-to-moderate cost error.**
- **Business Directive**: The system must prioritize **high recall** on the positive class (`yes`) while maintaining acceptable precision, ensuring the sales team focuses their calls on high-probability leads.

---

## 2. Dataset Criteria Check (Section 1 & 2 Verification)
| Criterion | Project Specification Requirement | UCI Bank Marketing Reality | Compliant? |
|---|---|---|---|
| **Size** | Tabular: $\ge$ 10,000 rows & 10 features | **45,211 rows**, **16 input features** | Yes (4.5x minimum) |
| **Task Difficulty** | Binary with clear class imbalance | **Imbalanced target**: 39,922 "no" (88.3%) vs 5,289 "yes" (11.7%) | Yes (Severe imbalance ~88:12) |
| **Real-world Mess** | Missing values, noisy records, mixed types | Contains numeric and categorical features, frequent `"unknown"` entries | Yes |
| **Not a Toy** | No MNIST, Iris, Titanic | Official UCI Benchmark (Moro et al., 2014) | Yes |
| **Feasibility** | $\le$ 30 minutes training on free Colab | MLP training takes $\sim$ 15-30 seconds per run | Yes (High feasibility) |

---

## 3. Justification of Evaluation Metrics

### Why Accuracy Fails:
A trivial dummy classifier predicting "no" for every single client achieves:
$$\text{Accuracy} = \frac{39,922}{45,211} = 88.30\%$$
Despite achieving $88.30\%$ accuracy, this model has a Recall of **0.0%** and captures zero new depositors, making it completely useless for the bank. According to the course rubric, using accuracy alone on imbalanced data results in lost marks.

### Primary Metric: Area Under Precision-Recall Curve (PR-AUC)
- **Mathematical Justification**:
  $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$
  Unlike ROC curves—which incorporate True Negatives ($TN$) in the False Positive Rate ($\frac{FP}{FP+TN}$) and can become overly optimistic when $TN$ is huge ($39,922$)—the Precision-Recall curve evaluates the direct trade-off exclusively on the minority positive class.
- **Decision Rule**: All hyperparameter selection, checkpointing, and model comparisons on the validation set will prioritize **Validation PR-AUC**.

### Secondary Monitored Metrics:
1. **ROC-AUC**: Evaluates overall ranking capability across discrimination thresholds.
2. **Positive Class Recall ($\text{Recall}_1$)**: Ensures high capture rate of prospective depositors.
3. **F1-Score ($\text{F1}_1$)**: Harmonic mean between precision and recall ($2 \cdot \frac{P \cdot R}{P + R}$).

---

## 4. Experimentation Protocol & Infrastructure

### 4.1 Strict Test Set Discipline & Anti-Leakage Rules
1. **Held-Out Test Set (15%)**: Kept isolated and evaluated exactly once at the end of the project (Week 14).
2. **Validation Set (15%)**: Used for all early stopping, hyperparameter tuning, and ablation studies.
3. **Training Set (70%)**: Used exclusively to fit model parameters, scalers, and encoders.

### 4.2 Fixed Reference Configuration (`configs/base_reference.json`)
To enable rigorous **One-Factor-At-A-Time (OFAT)** analysis, all studies deviate from a fixed control model:
- **Architecture**: 2 Hidden Dense Layers (`[128, 64]`), ReLU activation, Dropout rate $0.3$, Sigmoid output.
- **Optimizer**: Adam ($\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$).
- **Batch Size**: $64$.
- **Loss**: Binary Cross-Entropy.
- **Early Stopping**: Patience = $10$ epochs monitoring `val_loss`.

### 4.3 Systematic Hyperparameter Tuning Grid
In accordance with Section 3.5, three distinct factors will be tuned independently against the reference:
1. **Learning Rate Study** (`configs/tuning_learning_rate.json`):
   - $10^{-4}$ (0.0001): Tests convergence rate under small step sizes.
   - $10^{-3}$ (0.001): Reference Adam configuration.
   - $10^{-2}$ (0.01): Tests gradient stability and overshoot risk.
2. **Batch Size Study** (`configs/tuning_batch_size.json`):
   - Batch 32: High stochastic gradient variance, potential implicit regularization.
   - Batch 64: Reference configuration.
   - Batch 128: Smoothed gradient estimation, faster wall-clock execution.
3. **Optimizer Study** (`configs/tuning_optimizer.json`):
   - **Adam** ($\eta=0.001$): Adaptive first and second moment estimation.
   - **Vanilla SGD** ($\eta=0.01$, momentum=0.0): Baseline first-order stochastic descent.
   - **SGD with Momentum** ($\eta=0.01$, momentum=0.9): Momentum dampens oscillations in high-curvature valleys.

---

## 5. Planned Ablation Studies (4 Controlled Experiments)
Each ablation tests a specific hypothesis by toggling **one variable only** against the reference baseline:

| Ablation Study | Question & Hypothesis | Control | Variant | Expected Impact |
|---|---|---|---|---|
| **1. Dropout Regularization** | Does inverted dropout ($p=0.3$) reduce co-adaptation of neurons and close the generalization gap? | Dropout = $0.0$ | Dropout = $0.3$ | With dropout, validation loss remains lower and closer to training loss. |
| **2. Class Weighting** | Does cost-sensitive loss weighting improve minority class recall under 88:12 imbalance? | Unweighted BCE | Balanced Class Weights ($w_j = \frac{N}{2 \cdot N_j}$) | Recall for class 1 increases substantially with slight precision trade-off. |
| **3. Activation Function** | How do non-linearities influence loss surface geometry and gradient propagation? | ReLU ($\max(0, x)$) | Tanh & LeakyReLU ($\alpha=0.1$) | LeakyReLU prevents dying neuron phenomena; Tanh tests zero-centered activations. |
| **4. Batch Normalization** | Does normalizing layer inputs stabilize covariate shift in tabular MLPs? | Without BatchNorm | With BatchNorm before activation | Accelerates initial convergence and enables smoother loss decrease. |

---

## 6. Multi-Seed Statistical Validation Protocol
To avoid reporting a "lucky run" (Section 3.5 rubric violation):
- The final chosen model will be trained across **3 distinct random seeds**: `42`, `123`, and `999`.
- Metrics will be reported as:
  $$\mu \pm \sigma = \text{Mean} \pm \text{Standard Deviation}$$
- Plotted using `src/plot_curves.py:plot_multi_seed_spread` with a shaded $\pm 1\sigma$ confidence band.

---

## 7. Deliverables Implemented by Experimentation Engineer
1. `src/experiment_tracker.py`: Complete CSV logging framework and Keras `ExperimentTrackingCallback`.
2. `src/plot_curves.py`: Learning curve generator with automated overfitting/underfitting diagnostic annotations.
3. `src/experiment_runner.py`: End-to-end CLI execution engine for single runs and automated sweeps.
4. `configs/`: 8 JSON experiment configurations covering reference, tuning sweeps, and ablations.
5. `experiments/experiments.csv`: Master experiment registry with complete audit schema.
6. `notebooks/00_experimentation_framework_demo.ipynb`: Runnable demonstration notebook.
