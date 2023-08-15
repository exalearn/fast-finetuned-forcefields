#!/bin/bash

# Reminders: 
# - Set the number of QC workers equal to number of nodes in config.py

# Define where to look for a model
init_dir=../../notebooks/hydronet/2_initial-models/mp2_post-dft-refine/
init_model=$init_dir/best-models/dft_mctbp-25000_ttm-100k/avtz-starting-model
#init_dir=../../notebooks/hydronet/initial-models/dft_post-ttm-refine/
#init_model=$init_dir/best-models/md-100k-large/dft-starting-model

# Define the structures to use for 
search_space=../../notebooks/hydronet/structures-to-sample/small-clusters.db

# Start from the database of the last run
training_set=runs/mp2-mctbp-23Jul27-133936-30280b/train.db

# Test for the local system
python run.py \
    --redisport 7485 \
    --training-set $training_set \
    --search-space $search_space \
    --starting-model $init_model \
    --calculator mp2 \
    --sampling-method mctbp \
    --dynamics-temp 300 \
    --num-qc-workers 128 \
    --num-sampling-workers 8 \
    --node-size-map 65 105 165 195 225 \
    --min-run-length 100 \
    --max-run-length 1000 \
    --queue-length 8 \
    --queue-tolerance 0. \
    --num-frames 200 \
    --num-epochs 512 \
    --ensemble-size 8 \
    --learning-rate 3e-5 \
    --patience 64 \
    --max-force 10 \
    --huber-deltas 0.1 10 \
    --infer-chunk-size 3200 \
    --infer-pool-size 1 \
    --retrain-freq 32 \
    --num-to-run 512 \
    --energy-tolerance 0.001 \
    --parsl \
    --parsl-site perlmutter_nwchem \
    --train-ps-backend file
