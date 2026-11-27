"""Data loading, cleaning, leak-free splitting and preprocessing (Bank Marketing).

Design rules
------------
* Stateless cleaning (``clean_features``) is shared by training and inference,
  so preprocessing is never duplicated.
* Anything that *learns* from data (scaler, encoder) is fitted on the training
  split only, then applied to validation and test.
* No path is hard-coded: everything comes from ``configs/data.yaml`` and is
  resolved against the repository root.

Run ``python -m src.data`` to build the splits, fit/save the preprocessor,
run the leakage checks and write a small JSON report.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data.yaml"
PREVIOUSLY_CONTACTED = "previously_contacted"


# --------------------------------------------------------------------------- #
# Configuration and loading
# --------------------------------------------------------------------------- #
def load_config(path: str | Path | None = None) -> dict:
    """Read the YAML data configuration (defaults to ``configs/data.yaml``)."""
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path against the repository root (absolute paths kept)."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_raw(cfg: dict) -> pd.DataFrame:
    """Load the raw CSV. Raises a helpful error if the file was not downloaded."""
    csv_path = resolve_path(cfg["paths"]["raw_csv"])
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run: python data/download_data.py"
        )
    df = pd.read_csv(csv_path, sep=cfg["dataset"]["separator"])
    if cfg["dataset"]["target"] not in df.columns:
        raise ValueError("Target column missing; check dataset.separator in the config.")
    return df


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def drop_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate records. Returns the frame and the number removed."""
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, n_before - len(df)


def clean_features(X: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Stateless, row-wise cleaning shared by training and inference.

    * 'unknown' strings are kept as their own category: they are informative
      (e.g. poutcome='unknown' means "no earlier campaign") and are not random.
    * ``pdays == -1`` (never contacted) becomes a binary flag and 0.
    * Heavy-tailed columns get a signed log transform.
    * ``duration`` and the target are removed (duration is a post-call leak).
    """
    c = cfg["cleaning"]
    X = X.copy()
    X = X.drop(columns=[cfg["dataset"]["target"]], errors="ignore")
    if c["drop_duration"]:
        X = X.drop(columns=["duration"], errors="ignore")
    if "pdays" in X.columns:
        never = X["pdays"] == c["pdays_sentinel"]
        X[PREVIOUSLY_CONTACTED] = (~never).astype(int)
        X.loc[never, "pdays"] = 0
    for col in c["log_cols"]:
        if col in X.columns:
            X[col] = np.sign(X[col]) * np.log1p(np.abs(X[col]))
    return X


def encode_target(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Map the target to 0/1 (1 = subscribed). Matches sigmoid + binary CE."""
    d = cfg["dataset"]
    return (df[d["target"]] == d["positive_label"]).astype(int)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
@dataclass
class Splits:
    """Raw (unscaled) feature frames and 0/1 labels for each split."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def split_data(df: pd.DataFrame, cfg: dict) -> Splits:
    """Stratified train/val/test split (default 70/15/15), seeded.

    The rows carry no client id, so a stratified random split is used; the
    original row index is preserved so the split can be audited.
    """
    s = cfg["split"]
    if abs(s["train"] + s["val"] + s["test"] - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    X, y = df.drop(columns=[cfg["dataset"]["target"]]), encode_target(df, cfg)
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, test_size=s["val"] + s["test"], stratify=y, random_state=s["seed"]
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_rest, y_rest, test_size=s["test"] / (s["val"] + s["test"]),
        stratify=y_rest, random_state=s["seed"],
    )
    return Splits(X_tr, X_val, X_te, y_tr, y_val, y_te)


# --------------------------------------------------------------------------- #
# Scaling and encoding (fitted on train only)
# --------------------------------------------------------------------------- #
def build_preprocessor(X_clean: pd.DataFrame) -> ColumnTransformer:
    """Unfitted transformer: standard-scale numerics, one-hot categoricals.

    Column groups are derived from dtypes of the *cleaned* training frame.
    The binary flag is passed through unchanged. Unseen categories at
    inference time become all-zero rows instead of crashing.
    """
    passthrough = [PREVIOUSLY_CONTACTED] if PREVIOUSLY_CONTACTED in X_clean else []
    numeric = [c for c in X_clean.select_dtypes(include="number").columns
               if c not in passthrough]
    categorical = [c for c in X_clean.columns if c not in numeric + passthrough]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                  dtype=np.float32), categorical),
            ("flag", "passthrough", passthrough),
        ],
        verbose_feature_names_out=False,
    )


def preprocess_raw(df_raw: pd.DataFrame, preprocessor: ColumnTransformer,
                   cfg: dict) -> np.ndarray:
    """Raw rows -> model-ready float32 matrix. Single path for train AND inference."""
    cleaned = clean_features(df_raw, cfg)
    return preprocessor.transform(cleaned).astype(np.float32)


def load_preprocessor(cfg: dict | None = None) -> ColumnTransformer:
    """Load the fitted preprocessor saved by :func:`prepare_data`."""
    cfg = cfg or load_config()
    return joblib.load(resolve_path(cfg["paths"]["preprocessor"]))


# --------------------------------------------------------------------------- #
# Class imbalance
# --------------------------------------------------------------------------- #
def class_imbalance_summary(y: pd.Series | np.ndarray) -> dict:
    """Counts, proportions and imbalance ratio (majority / minority)."""
    counts = pd.Series(np.asarray(y)).value_counts().sort_index()
    return {
        "counts": {int(k): int(v) for k, v in counts.items()},
        "proportions": {int(k): round(float(v / counts.sum()), 4) for k, v in counts.items()},
        "imbalance_ratio": round(float(counts.max() / counts.min()), 2),
    }


def get_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """'Balanced' class weights from the TRAINING labels (Keras ``class_weight``)."""
    y_train = np.asarray(y_train)
    classes = np.unique(y_train)
    w = compute_class_weight("balanced", classes=classes, y=y_train)
    return {int(c): float(v) for c, v in zip(classes, w)}


def oversample_minority(X: np.ndarray, y: np.ndarray, seed: int = 42
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Random oversampling of the minority class (apply to TRAIN only)."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    minority, majority = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    extra = rng.choice(minority, size=len(majority) - len(minority), replace=True)
    idx = rng.permutation(np.concatenate([np.arange(len(y)), extra]))
    return X[idx], y[idx]


# --------------------------------------------------------------------------- #
# Leakage checks
# --------------------------------------------------------------------------- #
def run_leakage_checks(splits: Splits, preprocessor: ColumnTransformer,
                       X_train_clean: pd.DataFrame, n_total: int,
                       tol: float = 0.005) -> dict:
    """Assert the split/preprocessing is leak-free. Raises ``AssertionError``.

    ``splits`` must hold the *cleaned* feature frames (output of ``clean_features``).

    Checks: (1) splits are pairwise disjoint and cover every row, (2) class
    ratio is preserved in every split, (3) scaler mean/std equal the TRAIN
    statistics, (4) one-hot categories were learned from train only, and
    (5) transformed outputs contain no NaN/inf.
    """
    parts = {"train": splits.X_train.index, "val": splits.X_val.index,
             "test": splits.X_test.index}
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert len(parts[a].intersection(parts[b])) == 0, f"{a}/{b} overlap"
    assert sum(len(p) for p in parts.values()) == n_total, "rows lost or duplicated"

    overall = np.mean(np.concatenate([splits.y_train, splits.y_val, splits.y_test]))
    rates = {"train": splits.y_train.mean(), "val": splits.y_val.mean(),
             "test": splits.y_test.mean()}
    for name, r in rates.items():
        assert abs(r - overall) < tol, f"{name} positive rate {r:.4f} != {overall:.4f}"

    _, scaler, num_cols = preprocessor.transformers_[0]
    train_mean = X_train_clean[num_cols].mean().to_numpy()
    train_std = X_train_clean[num_cols].std(ddof=0).to_numpy()
    assert np.allclose(scaler.mean_, train_mean), "scaler mean not from train"
    assert np.allclose(scaler.scale_, train_std), "scaler std not from train"

    _, encoder, cat_cols = preprocessor.transformers_[1]
    for col, cats in zip(cat_cols, encoder.categories_):
        assert set(cats) == set(X_train_clean[col].unique()), f"{col}: categories not from train"

    for name, X in (("train", splits.X_train), ("val", splits.X_val), ("test", splits.X_test)):
        assert np.isfinite(preprocessor.transform(X)).all(), f"{name}: NaN/inf after transform"
    return {"disjoint": True, "covers_all_rows": True,
            "positive_rate": {k: round(float(v), 4) for k, v in rates.items()},
            "scaler_fitted_on_train_only": True, "encoder_fitted_on_train_only": True}


# --------------------------------------------------------------------------- #
# End-to-end entry point
# --------------------------------------------------------------------------- #
@dataclass
class PreparedData:
    """Everything the model code needs, already scaled/encoded (float32)."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    preprocessor: ColumnTransformer
    class_weights: dict[int, float]
    report: dict = field(default_factory=dict)


def prepare_data(cfg: dict | None = None, save: bool = True) -> PreparedData:
    """Load -> clean -> split -> fit on train -> transform all splits -> checks.

    With ``save=True`` the fitted preprocessor is written to
    ``paths.preprocessor`` and the split row-indices plus a report to
    ``paths.processed_dir`` (all small files, safe to commit).
    """
    cfg = cfg or load_config()
    raw = load_raw(cfg)
    n_raw = len(raw)
    df, n_dupes = drop_duplicate_rows(raw) if cfg["cleaning"]["drop_duplicates"] else (raw, 0)

    splits = split_data(df, cfg)
    clean = {k: clean_features(getattr(splits, f"X_{k}"), cfg) for k in ("train", "val", "test")}

    preprocessor = build_preprocessor(clean["train"])
    preprocessor.fit(clean["train"])  # TRAIN ONLY
    X = {k: preprocessor.transform(v).astype(np.float32) for k, v in clean.items()}

    splits_for_check = Splits(clean["train"], clean["val"], clean["test"],
                              splits.y_train, splits.y_val, splits.y_test)
    checks = run_leakage_checks(splits_for_check, preprocessor, clean["train"], len(df))

    report = {
        "rows_raw": n_raw, "duplicates_removed": n_dupes, "rows_used": len(df),
        "n_features_after_encoding": int(X["train"].shape[1]),
        "split_sizes": {k: int(len(v)) for k, v in X.items()},
        "imbalance_train": class_imbalance_summary(splits.y_train),
        "class_weights_train": get_class_weights(splits.y_train),
        "unknown_counts": {c: int((df[c] == "unknown").sum())
                           for c in df.columns if (df[c] == "unknown").any()},
        "leakage_checks": checks,
    }
    prepared = PreparedData(
        X["train"], X["val"], X["test"],
        splits.y_train.to_numpy(), splits.y_val.to_numpy(), splits.y_test.to_numpy(),
        list(preprocessor.get_feature_names_out()), preprocessor,
        report["class_weights_train"], report,
    )
    if save:
        out_dir = resolve_path(cfg["paths"]["processed_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        pre_path = resolve_path(cfg["paths"]["preprocessor"])
        pre_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, pre_path)
        index = {k: [int(i) for i in getattr(splits, f"X_{k}").index] for k in ("train", "val", "test")}
        (out_dir / "split_indices.json").write_text(json.dumps(index))
        (out_dir / "data_report.json").write_text(json.dumps(report, indent=2))
    return prepared


if __name__ == "__main__":
    print(json.dumps(prepare_data().report, indent=2))
