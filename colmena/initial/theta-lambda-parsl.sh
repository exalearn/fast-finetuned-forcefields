#!/bin/bash

# Test for the local system
python run.py \
    --ml-endpoint ae6479e9-61e6-4582-a34a-c17af560e535 \
    --qc-endpoint 698fba9a-4b12-4e0b-b83a-be6ded509946 \
    --training-set ../../notebooks/initial-database/initial-psi4-631g.db \
    --search-space ../../notebooks/initial-database/initial-psi4-631g.db \
    --starting-model ../../notebooks/psi4-integration/starting-psi4-model \
    --num-qc-workers 8 \
    --min-run-length 100 \
    --max-run-length 5000 \
    --num-frames 100 \
    --num-epochs 1024 \
    --ensemble-size 8 \
    --infer-chunk-size 4000 \
    --infer-pool-size 1 \
    --retrain-freq 16 \
    --num-to-run 1000 \
    --parsl \
    --train-ps-backend redis
