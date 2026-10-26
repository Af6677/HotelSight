"""Start the dashboard server, confirm it answers, then stop it.

Run:  python scripts/launch_check.py
Checks: the Streamlit health endpoint returns 'ok' and the app page is served.
(It cannot check how charts look in a browser - open the app for that.)
"""
import subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8765
proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py"),
                         "--server.port", str(PORT), "--server.headless", "true"],
                        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
ok = False
try:
    for _ in range(60):
        time.sleep(1)
        try:
            health = urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health", timeout=3).read().decode()
            page = urllib.request.urlopen(f"http://localhost:{PORT}/", timeout=5).read().decode()
            ok = health.strip() == "ok" and "<div id=\"root\">" in page
            break
        except OSError:
            if proc.poll() is not None:
                break
finally:
    proc.terminate()
    try:
        out = proc.communicate(timeout=10)[0]
    except subprocess.TimeoutExpired:
        proc.kill(); out = proc.communicate()[0]
print("LAUNCH CHECK PASSED: server healthy and app page served" if ok else "LAUNCH CHECK FAILED\n" + (out or "")[-2000:])
sys.exit(0 if ok else 1)
