"""Start a Lightning studio on a GPU, disable auto-sleep, upload the judge job (trimmed scoring subsets) and launch it.
Credentials: LIGHTNING_USER_ID / LIGHTNING_API_KEY in the environment (never printed).

python lightning/launch_judge.py <studio-name> <MACHINE> <TRAIN_MIN> <SEED> [MODEL]
"""
import sys
import time

from lightning_sdk import Machine, Studio
from lightning_sdk.utils.resolve import _get_authed_user

name, machine, train_min, seed = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
model = sys.argv[5] if len(sys.argv) > 5 else "Qwen/Qwen2.5-7B"
u = _get_authed_user()
s = Studio(name=name, teamspace=u.teamspaces[0].name, user=u.name, create_ok=True)
if "Running" not in str(s.status):
    s.start(Machine.CPU)
try:
    s.auto_sleep = False
except Exception as e:  # noqa: BLE001
    print("auto_sleep not settable:", str(e)[:120], flush=True)
print("auto_sleep", getattr(s, "auto_sleep", "?"), flush=True)
if machine not in str(s.machine):
    s.switch_machine(getattr(Machine, machine))
print(s.status, s.machine, s.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader").strip(), flush=True)
files = {"kaggle/ce_ds/ce_train.parquet": "ce_train.parquet", "local_data/llm/unc_val_text_w.parquet": "unc_val_text.parquet",
         "local_data/llm/unc_test_text_ui.parquet": "unc_test_text.parquet", "lightning/lora_judge.py": "lora_judge.py"}
have = s.run("mkdir -p judge && ls judge")
for src, dst in files.items():
    if dst == "ce_train.parquet" and "ce_train.parquet" in have:
        continue
    s.upload_file(src, "judge/" + dst)   # lands as a file literally named 'judge\<dst>' (Windows path); fixed below
s.upload_file("lightning/fixnames.py", "fixnames.py")
print(s.run("ls fixnames.py* ; f=$(ls | grep -i fixnames | head -1); cp \"$f\" fixnames_run.py; python fixnames_run.py; ls -la judge"), flush=True)
print(s.run("pip install -q -U 'transformers>=4.45,<4.52' 'peft>=0.12' polars accelerate 2>&1 | tail -1"), flush=True)
s.run(f"cd judge && rm -rf ckpt && (MODEL={model} TRAIN_MIN={train_min} SEED={seed} MAXLEN=160 BS=32 CKPT_EVERY=1000 DATA=. "
      f"nohup python lora_judge.py > judge.log 2>&1 &)")
time.sleep(90)
print(s.run("tail -3 judge/judge.log"), flush=True)
