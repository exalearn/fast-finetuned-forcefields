#!/bin/bash

# Test for the local system
python run.py \
    --training-set  ../../notebooks/initial-database/initial-ttm.db \
    --search-space ../../notebooks/initial-database/initial-ttm.db \
    --starting-model ../../notebooks/psi4-integration/ttm/starting-psi4-model \
    --calculator ttm \
    --num-qc-workers 1 \
    --min-run-length 1 \
    --max-run-length 250 \
    --num-epochs 64 \
    --ensemble-size 4 \
    --huber-deltas 0.1 10 \
    --infer-chunk-size 1000 \
    --infer-pool-size 1 \
    --retrain-freq 10 \
    --num-to-run 50 \
    --parsl \
    --parsl-site local \
    --train-ps-backend file
