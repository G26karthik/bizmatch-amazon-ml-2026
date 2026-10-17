#!/bin/sh
cd /work/dive
export PYTHONDONTWRITEBYTECODE=1
PY="nice -n 19 /work/venv/bin/python"
while pgrep -f g4_groups.py > /dev/null; do sleep 5; done
POLARS_MAX_THREADS=2 $PY g2_prep.py > g2_prep.log 2>&1
( POLARS_MAX_THREADS=2 $PY g4_groups.py > g4.log 2>&1; POLARS_MAX_THREADS=1 $PY g7_decoy_clusters.py > g7.log 2>&1 ) &
( POLARS_MAX_THREADS=2 $PY g5_decoys.py US > g5_US.log 2>&1 ) &
( POLARS_MAX_THREADS=2 $PY g5_decoys.py India > g5_India.log 2>&1 ) &
wait
echo done > /work/dive/DONE2
