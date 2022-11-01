#!/bin/bash

# Test for the local system
python run.py \
    --ml-endpoint ae6479e9-61e6-4582-a34a-c17af560e535 \
    --qc-endpoint 698fba9a-4b12-4e0b-b83a-be6ded509946 \
    --training-set ../../notebooks/initial-database/initial-psi4-631g.db \
    --search-space ../../notebooks/initial-database/initial-psi4-631g.db \
    --starting-model ../../notebooks/psi4-integration/starting-psi4-model \
    --num-qc-workers 8 \
    --run-length 100 \
    --num-epochs 128 \
    --ensemble-size 4 \
    --infer-chunk-size 1000 \
    --infer-pool-size 4 \
    --retrain-freq 10000 \
    --num-to-run 1000 \
    --parsl \
    --train-ps-backend redis
