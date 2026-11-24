# Experiment Tracking Protocol & Guidelines

**Owner**: Experimentation Engineer  
**Scope**: Central experiment repository log and artifacts  
**Course Code**: AIE231 - Neural Networks  

---

## 1. Golden Rules of Experimentation
1. **Never Tune on the Test Set**:
   - The test set (15%) is strictly sealed until final submission (Week 14).
   - All hyperparameter decisions, early stopping, learning rate tuning, and ablation comparisons **must** be evaluated strictly on the **Validation Set (15%)**.
2. **Every Run Must Be Logged**:
   - No unrecorded "trial and error".
   - Every single execution of a neural model must append a row to `experiments.csv`.
3. **One-Factor-At-A-Time (OFAT)**:
   - When tuning or running ablations, change **only one variable** relative to `configs/base_reference.json` while keeping all other hyperparameters identical.
4. **Statistical Reproducibility**:
   - Always set and log `random_seed` (NumPy, Python random, TensorFlow).
   - Candidate models must be verified across 3 distinct seeds (`42`, `123`, `999`), and results reported as `mean ± std`.

---

## 2. Experiment Table Schema (`experiments.csv`)

| Column | Description | Example |
|---|---|---|
| `run_id` | Unique ID for the experiment run | `TUNE-LR-001`, `ABL-DROP-002` |
| `date` | Timestamp of completion (YYYY-MM-DD) | `2026-11-24` |
| `author` | Team member running the experiment | `ExperimentationEngineer` |
| `model_name` | Name of neural architecture | `Reference_MLP`, `Deep_MLP_ModelB` |
| `learning_rate` | Optimizer learning rate | `0.001`, `0.01` |
| `batch_size` | Training mini-batch size | `32`, `64`, `128` |
| `optimizer` | Optimizer used | `adam`, `sgd`, `sgd_momentum` |
| `hidden_units` | Hidden layer architecture | `[128, 64]` |
| `activation` | Non-linearity function | `relu`, `tanh`, `leaky_relu` |
| `dropout_rate` | Dropout probability | `0.3`, `0.0` |
| `use_batch_norm`| Boolean flag for BatchNorm | `True`, `False` |
| `use_class_weights` | Boolean flag for balanced loss | `True`, `False` |
| `random_seed` | Seed used for reproducibility | `42` |
| `epochs_trained`| Actual epochs before early stop | `35` |
| `training_time_sec` | Total wall-clock training time | `14.82` |
| `total_params` | Hand-calculated & summary total params | `14337` |
| `trainable_params` | Trainable parameters count | `14337` |
| `train_loss` | Final training loss | `0.2014` |
| `val_loss` | Final validation loss | `0.2241` |
| `train_pr_auc` | Training Precision-Recall AUC | `0.5842` |
| `val_pr_auc` | **Primary Metric**: Validation PR-AUC | `0.5621` |
| `val_roc_auc` | Validation ROC-AUC | `0.8974` |
| `val_f1_score` | Validation F1-score for class 1 | `0.5312` |
| `val_recall` | Validation Recall for class 1 | `0.4851` |
| `val_precision` | Validation Precision for class 1 | `0.5872` |
| `notes` | Hypothesis, observations, or conclusions | `Learning rate 1e-3 converged smoothly` |
| `curve_path` | Relative path to saved loss/metric curve | `experiments/curves/TUNE-LR-001_curves.png` |

---

## 3. ID Naming Conventions

To keep logs clean and easily parseable:
- `EXP-REF-xxx`: Reference model runs
- `TUNE-LR-xxx`: Learning rate tuning sweep
- `TUNE-BS-xxx`: Batch size tuning sweep
- `TUNE-OPT-xxx`: Optimizer study sweep
- `ABL-DROP-xxx`: Dropout ablation
- `ABL-CW-xxx`: Class weights ablation
- `ABL-ACT-xxx`: Activation function ablation
- `ABL-BN-xxx`: Batch normalization ablation
- `SEED-xxx`: Multi-seed validation runs
