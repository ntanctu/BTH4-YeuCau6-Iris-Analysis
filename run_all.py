"""Single entrypoint for recreating input data, analytics, charts and PDF."""
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
for script in ("src/prepare_data.py", "src/run_pipeline.py"):
    print(f"\n>>> Running {script}", flush=True)
    subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)
print("\nHoàn tất. Tiếp theo: streamlit run app.py")
