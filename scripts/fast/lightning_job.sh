#!/bin/bash
# Runs on the Lightning studio in ~/ce. Full test-like retrain of the best cross-encoder recipe (e5-base, 2 epochs,
# lr 3e-5, bs 128; H records dropped), plus an orphan-upweighted variant; each scores the uncertain zone.
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python ce_fast.py --base intfloat/multilingual-e5-base --train tl_train.parquet --eval_file ce_val.parquet --mode fast --bench 300 --n 60000 --drop_h --bs 128 > bench_fast.log 2>&1
grep -E "TRAIN DONE|Error" bench_fast.log
python ce_fast.py --base intfloat/multilingual-e5-base --train tl_train.parquet --mode fast --drop_h --epochs 2 --lr 3e-5 --bs 128 --seed 7 \
   --name tlce_b --score --data_dir . --out_dir . --eval_file ce_val.parquet > tlce_b.log 2>&1
grep -E "TRAIN DONE|eval logloss|wrote|Error" tlce_b.log
python ce_fast.py --base intfloat/multilingual-e5-base --train tl_train.parquet --mode fast --drop_h --orph_w 2 --epochs 2 --lr 3e-5 --bs 128 --seed 7 \
   --name tlce_bw --score --data_dir . --out_dir . --eval_file ce_val.parquet > tlce_bw.log 2>&1
grep -E "TRAIN DONE|eval logloss|wrote|Error" tlce_bw.log
touch ALLDONE
