#!/bin/bash

# Test for the local system
python run.py \
    --ml-endpoint db55e9cc-ec32-47d6-a6ff-ecd45776d276 \
    --qc-endpoint 698fba9a-4b12-4e0b-b83a-be6ded509946 \
    --training-set ../../notebooks/initial-database/initial-psi4-631g.db \
    --search-space ../../notebooks/initial-database/initial-psi4-631g.db \
    --starting-model ../../notebooks/psi4-integration/starting-psi4-model \
    --num-qc-workers 8 \
    --min-run-length 200 \
    --max-run-length 2000 \
    --num-frames 100 \
    --num-epochs 512 \
    --ensemble-size 8 \
    --infer-chunk-size 4000 \
    --infer-pool-size 1 \
    --retrain-freq 16 \
    --num-to-run 500 \
    --huber-deltas 1 10 \
    --train-ps-backend globus \
    --ps-globus-config globus_config.json
