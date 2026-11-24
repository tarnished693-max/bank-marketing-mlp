# Individual Defense Q&A Guide: Experimentation Engineer
**Course**: AIE231 Neural Networks Project - Alamein International University  
**Role**: Experimentation Engineer  
**Weight**: 10% Individual Q&A Defense  

---

## Q1: What was your specific responsibility in this team project?
**Answer**:
> "As the Experimentation Engineer, I owned the hyperparameter tuning infrastructure and centralized experiment tracking. Specifically, I designed and coded:
> 1. The centralized experiment logging framework (`src/experiment_tracker.py` and `experiments/experiments.csv`), which records every run's hyperparameters, timing, parameter counts, and metrics.
> 2. The learning curves visualization suite (`src/plot_curves.py`), which automatically diagnoses overfitting and underfitting and produces publication-ready comparative curves.
> 3. The modular experiment runner (`src/experiment_runner.py`) and config schemas (`configs/`).
> 4. In terms of neural network experiments, I owned the One-Factor-At-A-Time (OFAT) sweeps across learning rates, batch sizes, optimizers, and the multi-seed statistical robustness evaluation."

---

## Q2: Why did you choose PR-AUC instead of standard Accuracy or ROC-AUC as the primary metric?
**Answer**:
> "The Bank Marketing dataset has severe class imbalance: approximately 88.3% of clients say 'no' (negative class) and only 11.7% say 'yes' (positive class).
> - **Why Accuracy fails**: A dummy classifier predicting all zeros trivially achieves 88.3% accuracy, but captures 0 subscribers (0% recall), which is commercially catastrophic for the bank.
> - **Why PR-AUC is superior to ROC-AUC here**: ROC-AUC plots True Positive Rate vs False Positive Rate. Because the number of True Negatives (39,922) is overwhelmingly large, the denominator in $FPR = \frac{FP}{FP + TN}$ makes FPR stay tiny even if the model makes many false positive mistakes. This creates an overly optimistic curve. In contrast, the Precision-Recall curve focuses strictly on the minority positive class ($TP$, $FP$, and $FN$), accurately measuring the true trade-off between lead quality (Precision) and lead capture (Recall)."

---

## Q3: What is 'One-Factor-At-A-Time' (OFAT) tuning, and why is it required by the course rubric?
**Answer**:
> "One-Factor-At-A-Time means establishing a fixed reference configuration (`configs/base_reference.json`) and altering strictly one hyperparameter at a time while holding every other variable constant. If you change the learning rate, batch size, and dropout rate simultaneously, you introduce confounding factors and cannot attribute performance changes to a single design choice. OFAT provides controlled, reproducible scientific evidence."

---

## Q4: How do you identify overfitting and underfitting from the learning curves?
**Answer**:
> "In `src/plot_curves.py`, I built automated diagnostic logic:
> - **Overfitting**: The training loss continues to decrease steadily, but the validation loss reaches a minimum and then begins to diverge upward, creating an expanding generalization gap. In that case, we apply early stopping and regularization (Dropout / Weight Decay).
> - **Underfitting**: Both training and validation losses remain high and flatten out early, or the metric fails to improve significantly beyond chance. This indicates insufficient model capacity (too few layers/units) or an unsuitable learning rate."

---

## Q5: How did you ensure there was no data leakage during experiment runs?
**Answer**:
> "We enforced strict test set discipline:
> 1. The test set (15%) was completely sealed and never used during hyperparameter tuning, model selection, or early stopping.
> 2. All tuning sweeps and ablations were evaluated strictly on the 15% validation split.
> 3. All preprocessing transformers (StandardScaler, One-Hot Encoders) were fitted exclusively on the 70% training split and only applied (via `.transform()`) to validation and test data."

---

## Q6: Why did you run the final model on 3 different random seeds?
**Answer**:
> "A single training run can achieve good validation numbers purely by chance due to lucky weight initialization or mini-batch ordering. To prove that our final model's performance is statistically robust and reproducible, we train it on three distinct seeds (`42`, `123`, `999`) and report the mean and standard deviation ($\mu \pm \sigma$). This directly satisfies the requirement in Section 3.5 of the course specification."

---

## Q7: How do you calculate the number of parameters of a Dense layer by hand?
**Answer**:
> "For a fully-connected Dense layer:
> $$\text{Parameters} = (\text{Input Dimensions} \times \text{Units}) + \text{Units}$$
> Where $(\text{Inputs} \times \text{Units})$ represents the weight matrix $W$, and the additional $\text{Units}$ represents the bias vector $b$.
> For example: An input of 48 features connected to a hidden layer of 128 units has:
> $$(48 \times 128) + 128 = 6,144 + 128 = 6,272 \text{ parameters}.$$
> Our tracking callback automatically checks this against `model.count_params()`."
