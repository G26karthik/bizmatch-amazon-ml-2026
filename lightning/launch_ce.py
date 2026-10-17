"""Start a Lightning studio (RTX PRO 6000), upload the fast cross-encoder job and run a shell job list in the background.
Credentials come from the environment (use lightning/acct2.py for account 2); never printed.

python lightning/acct2.py lightning/launch_ce.py <studio> <jobfile.sh>     # start + upload + launch
python lightning/acct2.py lightning/launch_ce.py <studio> --watch <dstdir>  # poll every 150 s, fetch outputs, stop
python lightning/acct2.py lightning/launch_ce.py <studio> --stop
"""
import sys
import time
from pathlib import Path

from lightning_sdk import Machine, Studio
from lightning_sdk.utils.resolve import _get_authed_user

name = sys.argv[1]
u = _get_authed_user()
s = Studio(name=name, teamspace=u.teamspaces[0].name, user=u.name, create_ok=True)


def fetch_all(dst):
    dst = Path(dst); dst.mkdir(parents=True, exist_ok=True)
    lst = s.run("cd ce && ls *.parquet *.log 2>/dev/null").split()
    for f in lst:
        try:
            s.download_file(f"ce/{f}", str(dst / f)); print("downloaded", f, flush=True)
        except Exception as e:  # noqa: BLE001
            print("download failed", f, str(e)[:150], flush=True)


if len(sys.argv) > 2 and sys.argv[2] == "--stop":
    s.stop(); print("stopped", s.status); sys.exit()
if len(sys.argv) > 2 and sys.argv[2] == "--watch":
    dst = sys.argv[3]
    for i in range(80):
        try:
            tail = s.run("tail -n 3 ce/job.log; ls ce | grep -c ALLDONE")
        except Exception as e:  # noqa: BLE001
            tail = f"ERR {e}"
        print(time.strftime("%H:%M:%S"), tail.replace("\n", " | ")[-400:], flush=True)
        if tail.strip().endswith("1") or "Traceback" in tail:
            fetch_all(dst)
            s.stop(); print("stopped", name, flush=True)
            break
        time.sleep(150)
    sys.exit()

job = sys.argv[2]
if len(sys.argv) > 3 and sys.argv[3] == "--relaunch":   # studio already up with data: refresh script + job only
    for src, dst in {"scripts/fast/ce_fast.py": "ce_fast.py", job: "job.sh"}.items():
        s.upload_file(src, "ce/" + dst)
    print(s.run("python - <<'EOF'\nimport os\nfor f in os.listdir('.'):\n    if f.startswith('ce\\\\'):\n        os.replace(f, os.path.join('ce', f.split('\\\\', 1)[1]))\nprint(sorted(os.listdir('ce')))\nEOF"), flush=True)
    s.run("cd ce && rm -f ALLDONE && (nohup bash job.sh > job.log 2>&1 &)")
    time.sleep(90)
    print(s.run("tail -n 5 ce/job.log; tail -n 3 ce/bench_fast.log"), flush=True)
    sys.exit()
if "Running" not in str(s.status):
    s.start(Machine.CPU)
try:
    s.auto_sleep = False
except Exception as e:  # noqa: BLE001
    print("auto_sleep not settable:", str(e)[:120], flush=True)
print("auto_sleep", getattr(s, "auto_sleep", "?"), flush=True)
t = time.time()
files = {"scripts/fast/ce_fast.py": "ce_fast.py", "local_data/fast/tl_train.parquet": "tl_train.parquet",
         "local_data/llm/unc_val_text.parquet": "unc_val_text.parquet", "local_data/llm/unc_test_text.parquet": "unc_test_text.parquet",
         "kaggle/ce_ds/ce_val.parquet": "ce_val.parquet", job: "job.sh"}
s.run("mkdir -p ce")
for src, dst in files.items():
    s.upload_file(src, "ce/" + dst)   # may land as a literal 'ce\<dst>' file (Windows path); fixed below
    print("uploaded", dst, round(time.time() - t), "s", flush=True)
print(s.run("python - <<'EOF'\nimport os\nfor f in os.listdir('.'):\n    if f.startswith('ce\\\\'):\n        os.replace(f, os.path.join('ce', f.split('\\\\', 1)[1]))\nprint(sorted(os.listdir('ce')))\nEOF"), flush=True)
if "RTXP_6000" not in str(s.machine):
    s.switch_machine(Machine.RTXP_6000)
print(s.status, s.machine, s.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader").strip(), round(time.time() - t), "s", flush=True)
print(s.run("pip install -q 'polars==1.44.2' 'transformers>=4.48' 2>&1 | tail -1; python -c 'import torch,transformers;print(torch.__version__,transformers.__version__)'"), flush=True)
s.run("cd ce && (nohup bash job.sh > job.log 2>&1 &)")
time.sleep(60)
print(s.run("tail -n 5 ce/job.log"), flush=True)
