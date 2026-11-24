# Bank Marketing Term Deposit Prediction (Deep MLP)
**Course**: AIE231 - Neural Networks  
**Institution**: Alamein International University, Faculty of Computer Science & Engineering  
**Track**: Tabular Deep Learning (Multi-Layer Perceptron)  

---

## 1. Problem Statement & Overview
This project addresses direct telemarketing campaign optimization for a Portuguese retail bank. The goal is to predict whether a prospective client will subscribe to a long-term deposit (`y` = `yes` / `no`) based on 16 attributes covering demographic, economic, and campaign history features.

Because marketing campaigns suffer from severe class imbalance (~88.3% "no" vs ~11.7% "yes"), trivial accuracy is an invalid objective. The project employs a systematically tuned and regularized Multi-Layer Perceptron (MLP) evaluated primarily on **Precision-Recall AUC (PR-AUC)** and **F1-Score**.

- **Dataset**: [UCI Machine Learning Repository - Bank Marketing (2014)](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Instances**: 45,211 rows, 16 features.

---

## 2. Repository Structure (Modular Architecture)
```
bank-marketing-mlp/
├── README.md                  # Project overview, installation, and instructions
├── requirements.txt           # Pinned dependencies
├── .gitignore                 # Excludes raw data and virtual environment
├── configs/                   # Experiment JSON configs (base reference, tuning, ablations)
│   ├── base_reference.json
│   ├── tuning_learning_rate.json
│   ├── tuning_batch_size.json
│   ├── tuning_optimizer.json
│   ├── ablation_dropout.json
│   ├── ablation_class_weights.json
│   ├── ablation_activation.json
│   ├── ablation_batch_norm.json
│   └── multi_seed.json
├── data/                      # Dataset documentation and download script
│   ├── README.md
│   └── download_data.py
├── notebooks/                 # Demonstration & EDA notebooks
│   └── 00_experimentation_framework_demo.ipynb
├── src/                       # Modular source code
│   ├── __init__.py
│   ├── experiment_tracker.py  # Central CSV logger & Keras tracking callback
│   ├── plot_curves.py         # Learning curve generator with overfitting detection
│   └── experiment_runner.py   # CLI runner for configs and sweeps
├── experiments/               # Experiment logs and artifact figures
│   ├── README.md              # Logging conventions and schema guide
│   ├── experiments.csv        # Centralized audit log of all runs
│   └── curves/                # Exported publication-ready curve plots
├── models/                    # Saved model artifacts (.keras)
├── app/                       # Live inference demo application
├── report/                    # Final PDF technical report and presentation slides
└── docs/                      # Work package proposals & defense guides
    ├── week9_experimentation_proposal.md
    └── defense_qa_guide.md
```

---

## 3. Installation & Quickstart

### 3.1 Clone and Setup Environment
```bash
git clone https://github.com/tarnished693-max/bank-marketing-mlp.git
cd bank-marketing-mlp
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3.2 Download Dataset
```bash
python data/download_data.py
```

---

## 4. Experimentation & Reproduction Commands

### 4.1 Run the Base Reference Model
```bash
python src/experiment_runner.py --config configs/base_reference.json --run-id EXP-REF-001
```

### 4.2 Run Hyperparameter Sweeps (One-Factor-At-A-Time)
- **Learning Rate Study** (1e-4 vs 1e-3 vs 1e-2):
  ```bash
  python src/experiment_runner.py --sweep lr
  ```
- **Batch Size Study** (32 vs 64 vs 128):
  ```bash
  python src/experiment_runner.py --sweep batch_size
  ```
- **Optimizer Study** (Adam vs Vanilla SGD vs SGD with Momentum):
  ```bash
  python src/experiment_runner.py --sweep optimizer
  ```

### 4.3 Run Ablation Studies
Executes the 4 ablation studies (Dropout, Class Weights, Activation Functions, Batch Normalization):
```bash
python src/experiment_runner.py --sweep ablations
```

### 4.4 Run Multi-Seed Verification (3 Seeds)
```bash
python src/experiment_runner.py --sweep multi_seed
```

---

## 5. Experiment Tracking Log Sample (`experiments/experiments.csv`)
All runs automatically log to `experiments/experiments.csv` with the following schema:
- `run_id`, `date`, `author`, `model_name`, `learning_rate`, `batch_size`, `optimizer`, `hidden_units`, `activation`, `dropout_rate`, `use_batch_norm`, `use_class_weights`, `random_seed`, `epochs_trained`, `training_time_sec`, `total_params`, `trainable_params`, `train_loss`, `val_loss`, `train_pr_auc`, `val_pr_auc`, `val_roc_auc`, `val_f1_score`, `val_recall`, `val_precision`, `notes`, `curve_path`.

---

## 6. Team Role Allocation (AIE231 Specification)
| Role | Member / Work Package |
|---|---|
| **Tech Lead** | Repository structure, integration, code review, 3-seed final model |
| **Data Engineer** | Data cleaning, leak-free preprocessing, class imbalance handling |
| **Baseline Engineer** | EDA, trivial baseline, logistic regression baseline |
| **Model Architect** | MLP models (Shallow, Deep, Tuned), hand parameter counts, GradientTape |
| **Experimentation Engineer** | **Hyperparameter tuning, experiment tracker, learning-curve plots, OFAT sweeps** |
| **Deployment Engineer** | Evaluation metrics, error analysis, predict.py, demo application |
