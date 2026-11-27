"""Gradio live demo on top of src.predict.Predictor.

    python app/demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr  # noqa: E402

from src.predict import Predictor  # noqa: E402

CATEGORICAL = {
    "job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
            "self-employed", "services", "student", "technician", "unemployed", "unknown"],
    "marital": ["divorced", "married", "single"],
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "default": ["no", "yes"], "housing": ["no", "yes"],
    "loan": ["no", "yes"], "contact": ["cellular", "telephone", "unknown"],
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "poutcome": ["failure", "other", "success", "unknown"],
}
NUMERIC = {  # name: (default, minimum, maximum)
    "age": (41, 18, 95), "balance": (1500, -8019, 102127),
    "day": (15, 1, 31), "campaign": (2, 1, 63),
    "pdays": (-1, -1, 871), "previous": (0, 0, 275),
}
FIELDS = list(CATEGORICAL) + list(NUMERIC)

predictor = Predictor()


def run(*values):
    """Gradio callback: values arrive in FIELDS order."""
    res = predictor.predict(dict(zip(FIELDS, values)))[0]
    label = {"Will subscribe": res["p_subscribe"], "Will not subscribe": 1 - res["p_subscribe"]}
    info = (f"Prediction: **{res['prediction'].upper()}** | confidence {res['confidence']:.1%} | "
            f"threshold {res['threshold']:.2f} | {res['latency_ms']} ms")
    return label, info


inputs = [gr.Dropdown(CATEGORICAL[f], value=CATEGORICAL[f][0], label=f) for f in CATEGORICAL] + [
    gr.Number(value=v[0], minimum=v[1], maximum=v[2], label=f) for f, v in NUMERIC.items()]

demo = gr.Interface(
    fn=run, inputs=inputs,
    outputs=[gr.Label(label="Subscription probability"), gr.Markdown()],
    title="Bank Marketing - Term Deposit Predictor",
    description="Course project (AIE231). Pre-call features only; not financial advice.",
)

if __name__ == "__main__":
    demo.launch()
