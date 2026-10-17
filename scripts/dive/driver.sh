#!/bin/sh
# chain the dive jobs on the worker (all niced, low thread counts)
cd /work/dive
export PYTHONDONTWRITEBYTECODE=1
PY="nice -n 19 /work/venv/bin/python"
while pgrep -f g2_prep.py > /dev/null; do sleep 10; done
( POLARS_MAX_THREADS=2 $PY g4_groups.py > g4.log 2>&1 ) &
( POLARS_MAX_THREADS=2 $PY g5_decoys.py US > g5_US.log 2>&1; POLARS_MAX_THREADS=2 $PY g5_decoys.py India > g5_India.log 2>&1 ) &
while pgrep -f "g1.py|g1_fetch.py" > /dev/null; do sleep 10; done
POLARS_MAX_THREADS=2 $PY g1_summary.py > g1_summary.log 2>&1
wait
echo alldone > /work/dive/DONE
