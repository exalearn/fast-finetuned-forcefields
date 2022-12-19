#!/bin/bash

# Test for the local system
python run.py \
    --training-set ../../notebooks/psi4-integration/initial-database.db \
    --search-space ../../notebooks/psi4-integration/initial-database.db \
    --starting-model ../../notebooks/psi4-integration/starting-psi4-model \
    --calculator dft \
    --num-qc-workers 1 \
    --min-run-length 1 \
    --max-run-length 100 \
    --num-epochs 64 \
    --ensemble-size 4 \
    --huber-deltas 0.1 10 \
    --infer-chunk-size 1000 \
    --infer-pool-size 4 \
    --retrain-freq 10 \
    --num-to-run 100 \
    --parsl \
    --parsl-site local \
    --train-ps-backend file
