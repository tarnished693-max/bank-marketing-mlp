# Data

**Dataset:** Bank Marketing (UCI) – `bank-full.csv`, 45,211 rows, 16 input features,
binary target `y` (client subscribed to a term deposit).
**Link:** https://archive.ics.uci.edu/dataset/222/bank+marketing
**Licence:** CC BY 4.0 (academic use allowed).
**Citation:** S. Moro, P. Cortez, P. Rita. *A Data-Driven Approach to Predict the Success of
Bank Telemarketing.* Decision Support Systems 62 (2014).

## Get the data
```bash
python data/download_data.py     # writes data/raw/bank-full.csv (not tracked by git)
```

## What the pipeline does (`src/data.py`, config in `configs/data.yaml`)
| Step | Decision | Why |
|---|---|---|
| Duplicates | exact duplicates dropped (0 found) | avoid the same row in train and test |
| `"unknown"` values (job, education, contact, poutcome) | kept as their own category | not random; `poutcome=unknown` = no earlier campaign |
| `pdays = -1` | flag `previously_contacted` + set to 0 | −1 is a sentinel, not a number of days |
| `balance, campaign, previous, pdays` | signed log1p | heavy-tailed, outliers |
| `duration` | dropped (switchable in config) | only known after the call → target leakage |
| Split | stratified 70/15/15, seed 42 | target is 11.7 % positive; test used once at the end |
| Scaling / encoding | StandardScaler + one-hot, **fitted on train only** | no leakage |
| Imbalance (7.55 : 1) | class weights and random oversampling (train only) | effect on recall: `experiments/imbalance_logreg.csv` |

## Use it from other code
```python
from src.data import prepare_data, load_preprocessor, preprocess_raw
d = prepare_data()          # d.X_train, d.y_train, d.class_weights, ...
```
`preprocess_raw(df_raw, preprocessor, cfg)` is the single function that `predict.py` must reuse.
