#!/bin/bash

# Test for the local system
python run.py \
    --ml-endpoint 9fcf93c9-bfb3-459a-94de-d5114954229b \
    --qc-endpoint 3c80aa3f-702e-46e0-8707-ba307c85affa \
    --training-set ../../notebooks/initial-database/initial-ttm.db \
    --search-space ../../notebooks/initial-database/initial-ttm.db \
    --starting-model ../../notebooks/train-schnetpack/test/best_model \
    --num-simulators 1 \
    --run-length 10000 \
    --num-epochs 8 \
    --ensemble-size 4 \
    --infer-chunk-size 100 \
    --retrain-freq 10 \
    --num-to-run 1000

