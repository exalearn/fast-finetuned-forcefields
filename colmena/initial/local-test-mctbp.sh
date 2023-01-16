#!/bin/bash

# Test for the local system
python run.py \
    --training-set  ../../notebooks/initial-database/initial-ttm.db \
    --search-space ../../notebooks/initial-database/initial-ttm.db \
    --starting-model ../../notebooks/initial-models/ttm/starting-model \
    --calculator ttm \
    --sampling-method mctbp \
    --num-qc-workers 1 \
    --min-run-length 2 \
    --max-run-length 200 \
    --num-frames 200 \
    --num-epochs 128 \
    --ensemble-size 4 \
    --huber-deltas 0.1 10 \
    --infer-chunk-size 2000 \
    --infer-pool-size 1 \
    --retrain-freq 25 \
    --num-to-run 250 \
    --parsl \
    --parsl-site local \
    --train-ps-backend file
