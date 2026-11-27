"""Single-sample / batch inference on RAW input.

Re-uses the preprocessing object fitted at training time (saved with joblib),
so no preprocessing logic is duplicated between training and inference.

    python -m src.predict --input configs/sample_client.json
    python -m src.predict --input clients.csv --model models/final_mlp.keras

Python:
    from src.predict import Predictor
    result = Predictor().predict({...raw client fields...})
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Union

import joblib
import numpy as np
import pandas as pd

from src.data import PREVIOUSLY_CONTACTED, clean_features, load_config

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DEFAULT_MODEL = MODELS_DIR / "final_mlp.keras"
DEFAULT_PREPROCESSOR = MODELS_DIR / "preprocessor.joblib"
THRESHOLD_FILE = MODELS_DIR / "threshold.json"  # {"threshold": 0.37}, chosen on validation
LEAKY_COLUMNS = ("duration", "y")  # unknown before a call / the target itself


class Predictor:
    """Loads model + preprocessor once; scores raw client records."""

    def __init__(self, model_path: Path = DEFAULT_MODEL,
                 preprocessor_path: Path = DEFAULT_PREPROCESSOR,
                 threshold_file: Path = THRESHOLD_FILE,
                 data_cfg: Dict = None):
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.preprocessor = joblib.load(self.preprocessor_path)
        self.required = list(self.preprocessor.feature_names_in_)
        self.uses_data_cleaning = PREVIOUSLY_CONTACTED in self.required
        self.data_cfg = data_cfg or load_config()
        self.threshold = 0.5
        if Path(threshold_file).exists():
            self.threshold = float(json.loads(Path(threshold_file).read_text())["threshold"])
        if self.model_path.suffix == ".joblib":
            self.model = joblib.load(self.model_path)
        else:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path)

    def _to_frame(self, raw: Union[Dict, List[Dict], pd.DataFrame]) -> pd.DataFrame:
        df = pd.DataFrame([raw]) if isinstance(raw, dict) else pd.DataFrame(raw)
        if self.uses_data_cleaning:
            df = clean_features(df, self.data_cfg)
        else:
            df = df.drop(columns=[c for c in LEAKY_COLUMNS if c in df.columns])
        missing = [c for c in self.required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        return df[self.required]

    def _proba(self, X) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(X))[:, 1]
        return np.asarray(self.model.predict(X, verbose=0)).ravel()

    def predict(self, raw: Union[Dict, List[Dict], pd.DataFrame]) -> List[Dict]:
        """Return one result dict per input row.

        Keys: ``prediction`` ("yes"/"no"), ``p_subscribe``, ``confidence``
        (probability of the predicted label), ``threshold``, ``latency_ms``
        (per-sample: preprocessing + forward pass).
        """
        t0 = time.perf_counter()
        df = self._to_frame(raw)
        X = self.preprocessor.transform(df)
        if hasattr(X, "toarray"):
            X = X.toarray()
        p = self._proba(X.astype("float32"))
        ms = (time.perf_counter() - t0) * 1000 / len(df)
        out = []
        for pi in p:
            yes = pi >= self.threshold
            out.append({"prediction": "yes" if yes else "no",
                        "p_subscribe": round(float(pi), 4),
                        "confidence": round(float(pi if yes else 1 - pi), 4),
                        "threshold": round(self.threshold, 4),
                        "latency_ms": round(ms, 3)})
        return out

    def benchmark(self, raw: pd.DataFrame, n: int = 200) -> Dict[str, float]:
        """Per-sample latency (single-row calls) and model size on disk."""
        rows = raw.sample(min(n, len(raw)), random_state=0).to_dict("records")
        self.predict(rows[0])  # warm-up
        t = [self.predict(r)[0]["latency_ms"] for r in rows]
        size_kb = sum(p.stat().st_size for p in (self.model_path, self.preprocessor_path)
                      if p.exists()) / 1024
        return {"latency_ms_mean": float(np.mean(t)),
                "latency_ms_p95": float(np.percentile(t, 95)),
                "size_on_disk_kb": round(size_kb, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict term-deposit subscription.")
    ap.add_argument("--input", required=True, type=Path, help=".json (one client) or .csv")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--preprocessor", type=Path, default=DEFAULT_PREPROCESSOR)
    a = ap.parse_args()
    if a.input.suffix == ".json":
        raw = json.loads(a.input.read_text())
    else:  # UCI csv is ';'-separated
        raw = pd.read_csv(a.input, sep=None, engine="python")
    res = Predictor(a.model, a.preprocessor).predict(raw)
    print(json.dumps(res[0] if isinstance(raw, dict) else res, indent=2))


if __name__ == "__main__":
    main()
