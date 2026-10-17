#!/usr/bin/env bash
# Laptop: fetch cross-encoder score files (Kaggle kernel output or Modal volume, never the model weights) and stage them
# in S3 under ce_out_<name>/ for the CPU worker to ingest.
# Usage: scripts/fetch_ce.sh kaggle <kernel slug> <name>   |   scripts/fetch_ce.sh modal <run> <name>
set -euo pipefail
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 MSYS_NO_PATHCONV=1
AWS="${AWS:-aws}"   # AWS CLI; BUCKET, AWS_PROFILE and AWS_REGION come from the environment
kind=$1; src=$2; name=$3
d=local_data/ce_in/$name
mkdir -p "$d"
if [ "$kind" = kaggle ]; then
  uvx kaggle kernels output "YOUR_KAGGLE_USERNAME/$src" -p "$d" --file-pattern '.*\.(parquet|log)$' -o
else
  for f in ce_scores_val.parquet ce_scores_test.parquet fr_new.parquet train.log; do
    modal volume get --force aml2026-ce "/out/$src/$f" "$d/$f"
  done
fi
ls -la "$d"
"$AWS" s3 sync "$d" "s3://$BUCKET/ce_out_$name/" --exclude "*.log" --only-show-errors
echo "staged s3://$BUCKET/ce_out_$name/"
