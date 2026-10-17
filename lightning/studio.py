"""Lightning studio control for the LoRA judge. Credentials: LIGHTNING_USER_ID / LIGHTNING_API_KEY in the environment.

python lightning/studio.py start H100|A100_80GB|L40S     start the studio on a machine
python lightning/studio.py upload                         upload data + script
python lightning/studio.py launch [TRAIN_MIN]             install deps and start the job in the background
python lightning/studio.py status                         tail the job log
python lightning/studio.py fetch                          download llm_val / llm_test parquet
python lightning/studio.py stop                           stop the studio (stops billing)
"""
import sys
from pathlib import Path

from lightning_sdk import Machine, Studio
from lightning_sdk.utils.resolve import _get_authed_user

ROOT = Path(__file__).resolve().parents[1]
u = _get_authed_user()
s = Studio(name="scratch-studio-devbox", teamspace=u.teamspaces[0].name, user=u.name)
cmd = sys.argv[1]
if cmd == "start":
    m = getattr(Machine, sys.argv[2] if len(sys.argv) > 2 else "H100")
    s.start(m)
    print("started", s.status, m)
elif cmd == "upload":
    files = {ROOT / "kaggle/ce_ds/ce_train.parquet": "judge/ce_train.parquet",
             ROOT / "local_data/llm/unc_val_text.parquet": "judge/unc_val_text.parquet",
             ROOT / "local_data/llm/unc_test_text.parquet": "judge/unc_test_text.parquet",
             ROOT / "lightning/lora_judge.py": "judge/lora_judge.py"}
    s.run("mkdir -p judge")
    for src, dst in files.items():
        s.upload_file(str(src), dst)
        print("uploaded", dst)
elif cmd == "launch":
    tm = sys.argv[2] if len(sys.argv) > 2 else "70"
    print(s.run("pip install -q -U 'transformers>=4.45,<4.52' 'peft>=0.12' polars accelerate 2>&1 | tail -2; nvidia-smi --query-gpu=name,memory.total --format=csv"))
    print(s.run(f"cd judge && (TRAIN_MIN={tm} DATA=. nohup python lora_judge.py > judge.log 2>&1 &) ; sleep 5; tail -3 judge.log"))
elif cmd == "status":
    print(s.status)
    print(s.run("tail -8 judge/judge.log; ls -la judge | tail -6"))
elif cmd == "fetch":
    for f in ("llm_val.parquet", "llm_test.parquet", "judge.log"):
        s.download_file(f"judge/{f}", str(ROOT / "local_data/llm" / f))
        print("downloaded", f)
elif cmd == "stop":
    s.stop()
    print("stopped", s.status)
