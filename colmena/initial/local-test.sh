#!/bin/bash

# Test for the local system
python run.py \
    --ml-endpoint 9fcf93c9-bfb3-459a-94de-d5114954229b \
    --training-set ../../notebooks/initial-database/initial-ttm.db \
    --starting-model ../../notebooks/train-schnetpack/test/best_model \
    --num-qc-workers 1
