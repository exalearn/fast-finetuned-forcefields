#!/bin/bash
# Test for the local system using molecular dynamics
python run.py \
    --training-set  ../../notebooks/initial-models/dft/initial-database.db \
    --search-space ../../notebooks/initial-database/initial-ttm.db \
    --starting-model ../../notebooks/initial-models/dft/starting-model \
    --calculator dft \
    --sampling-method md \
    --num-qc-workers 1 \
    --min-run-length 100 \
    --max-run-length 10000 \
    --num-frames 200 \
    --num-epochs 128 \
    --ensemble-size 4 \
    --huber-deltas 0.1 10 \
    --max-force 10 \
    --infer-chunk-size 2000 \
    --infer-pool-size 1 \
    --retrain-freq 25 \
    --num-to-run 250 \
    --parsl \
    --parsl-site local \
    --train-ps-backend file
