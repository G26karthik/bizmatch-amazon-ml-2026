#!/bin/sh
# after prep: wait for g5 US to finish, then rules (g9) and decoy samples (g8)
cd /work/dive
export PYTHONDONTWRITEBYTECODE=1
sleep 20
while pgrep -f g2_prep.py > /dev/null; do sleep 10; done
sleep 15
while pgrep -f "g5_decoys.py US" > /dev/null; do sleep 10; done
POLARS_MAX_THREADS=2 nice -n 19 /work/venv/bin/python g9_rules.py > g9.log 2>&1
POLARS_MAX_THREADS=1 nice -n 19 /work/venv/bin/python g8_decoy_samples.py > g8.log 2>&1
echo done > /work/dive/DONE4
