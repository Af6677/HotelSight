"""Repeat all 48 fits in a new directory without overwriting the frozen run.

python reproduce.py --out fresh_run --threads 4
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True); ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent; out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Choose a new empty output directory; frozen results are preserved.")
    out.mkdir(parents=True, exist_ok=True); (out / "data").mkdir(exist_ok=True)
    for name in ["features.py", "experiment.py", "report.py", "predict.py", "requirements.txt", "analysis_plan.json"]:
        shutil.copy2(root / name, out / name)
    shutil.copy2(root / "data/hotel_bookings.csv", out / "data/hotel_bookings.csv")
    for options in [["prepare"], ["train", "--policy", "deduplicated"], ["train", "--policy", "retained"], ["select"], ["evaluate"]]:
        subprocess.run([sys.executable, str(out / "experiment.py"), *options, "--threads", str(args.threads)], cwd=out, check=True)
    subprocess.run([sys.executable, str(out / "report.py")], cwd=out, check=True)
    print("Fresh experiment complete:", out)
