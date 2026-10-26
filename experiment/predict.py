"""Predict from the exact raw-input pipeline used by a frozen arm winner.

python predict.py --policy retained --family general20 --input bookings.csv --output predictions.csv
"""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiment import ROOT, read, load_model
from features import make_features

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", choices=["deduplicated", "retained"], required=True)
    p.add_argument("--family", choices=["general20", "historical24"], required=True)
    p.add_argument("--input", required=True); p.add_argument("--output", default="predictions.csv")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    r = read(ROOT / "selection_before_test.json")["winners"][a.policy + "_" + a.family]
    raw = pd.read_csv(a.input)
    x = make_features(raw, a.family)
    pre = joblib.load(ROOT / r["preprocessor_path"])
    model = load_model(r["slug"], ROOT / r["model_path"], a.threads)
    with threadpool_limits(limits=a.threads):
        prob = model.predict_proba(np.asarray(pre.transform(x), dtype=np.float32))[:, 1]
    out = x.copy(); out["cancellation_probability"] = prob
    out["predicted_cancelled"] = (prob >= .5).astype(int)
    out["model"] = r["model"]
    out.to_csv(a.output, index=False)
    print(f"Wrote {len(out)} predictions using {r['model']} to {Path(a.output).resolve()}")
