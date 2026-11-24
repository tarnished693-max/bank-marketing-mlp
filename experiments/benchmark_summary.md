# Official Baseline Benchmark Summary Table (AIE231 Section 3.3)
**Author**: Experimentation Engineer  
**Milestone**: Checkpoint 2 (End of Week 10) - Baselines Ready  
**Primary Metric**: Validation PR-AUC (Precision-Recall Area Under Curve)  

---

## 1. Baseline Performance Table (Evaluated on Leak-Free Validation Split)

| Run ID | Model Name | Total Params | Training Time | Val Loss | Val PR-AUC (Primary) | Val ROC-AUC | Val F1-Score | Val Recall | Val Precision | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| **BASE-TRIVIAL-001** | Trivial (Majority Class) | 0 | 0.002s | 0.3608 | **0.1171** | 0.5000 | 0.0000 | 0.0000 | 0.0000 | Predicts all 'no' (class 0). Accuracy is 88.29% but Recall is 0.0%. |
| **BASE-LOGREG-001** | Logistic Regression (Standard) | 49 | 0.450s | 0.2480 | **0.4910** | 0.8835 | 0.4520 | 0.3460 | 0.6520 | L2-regularized linear baseline. High precision but low recall under default 0.5 cutoff. |
| **BASE-LOGREG-BALANCED-002** | Logistic Regression (Balanced) | 49 | 0.520s | 0.3620 | **0.4890** | 0.8842 | 0.5480 | 0.7840 | 0.4210 | Cost-sensitive loss weighting ($w_1 \approx 4.28$). Recall jumps to 78.4%. |
| **EXP-REF-001** | Reference Deep MLP | 14,593 | 14.820s | 0.2180 | **0.5840** | 0.9120 | 0.5620 | 0.4850 | 0.6680 | 2-hidden layer MLP ([128, 64], Dropout 0.3). Beats all baselines across all non-linear metrics. |

---

## 2. Quantified Lift Over Baselines (Section 3.3 Compliance)
> *"Every later model is reported as an improvement over these numbers."*

### Lift of Reference Deep MLP over Simple Learned Baseline (`BASE-LOGREG-001`):
- **PR-AUC Lift (Primary Metric)**:  
  $$\Delta \text{PR-AUC} = 0.5840 - 0.4910 = \mathbf{+0.0930} \quad (\mathbf{+18.94\% \text{ relative gain}})$$
- **ROC-AUC Lift**:  
  $$\Delta \text{ROC-AUC} = 0.9120 - 0.8835 = \mathbf{+0.0285} \quad (\mathbf{+3.23\% \text{ relative gain}})$$
- **F1-Score Lift**:  
  $$\Delta \text{F1} = 0.5620 - 0.4520 = \mathbf{+0.1100} \quad (\mathbf{+24.34\% \text{ relative gain}})$$
- **Recall Lift**:  
  $$\Delta \text{Recall} = 0.4850 - 0.3460 = \mathbf{+0.1390} \quad (\mathbf{+40.17\% \text{ relative gain}})$$
- **Loss Reduction**:  
  $$\Delta \text{Loss} = 0.2180 - 0.2480 = \mathbf{-0.0300} \quad (\mathbf{-12.10\% \text{ loss drop}})$$

---

## 3. Modeling Acceptance Threshold for Weeks 11-12
To qualify as an acceptable neural network candidate in Checkpoint 3 (Weeks 11-12):
1. **Model A (Shallow MLP)** must achieve $\text{PR-AUC} > 0.4910$ (beating Logistic Regression).
2. **Model B (Deeper Regularized MLP)** must achieve $\text{PR-AUC} > 0.5500$.
3. **Model C (Best MLP after Tuning)** must achieve $\text{PR-AUC} \ge 0.5800$ with $\text{Recall}_1 \ge 0.4800$.
