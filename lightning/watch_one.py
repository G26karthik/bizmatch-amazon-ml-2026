"""Keep-alive watcher for one judge studio: polls the log every 180 s (activity keeps the studio awake), downloads
llm_val/llm_test/judge.log when the job prints DONE or fails, then stops the studio.

python lightning/watch_one.py <studio-name> <local-dst-dir>
"""
import sys
import time
from pathlib import Path

from lightning_sdk import Studio
from lightning_sdk.utils.resolve import _get_authed_user

name, dst = sys.argv[1], Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
u = _get_authed_user()
for i in range(200):
    s = Studio(name=name, teamspace=u.teamspaces[0].name, user=u.name)
    try:
        tail = s.run("tail -2 judge/judge.log")
    except Exception as e:  # noqa: BLE001
        tail = f"ERR {e}"
    print(time.strftime("%H:%M:%S"), name, tail.replace("\n", " | ")[-200:], flush=True)
    if "DONE" in tail or "Traceback" in tail or "Error" in tail:
        for f in ("llm_val.parquet", "llm_test.parquet", "judge.log"):
            try:
                s.download_file(f"judge/{f}", str(dst / f)); print("downloaded", name, f, flush=True)
            except Exception as e:  # noqa: BLE001
                print("download failed", name, f, str(e)[:150], flush=True)
        s.stop(); print("stopped", name, flush=True)
        break
    time.sleep(180)
