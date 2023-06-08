#! /bin/bash
# Trains a model and runs sampling using the settings we've selected for TTM models

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

for run in $@; do
    echo $run
    python $script_dir/train-model.py --learning-rate 1e-5 --num-epochs 128 --max-force 10 $run
    python $script_dir/run-sampling.py --num-steps 100 --device cuda $run
    python $script_dir/run-sampling.py --structure-source pure --num-steps 100 --md-log-frequency 10 --sampling-method md --device cuda $run
    python $script_dir/run-sampling.py --structure-source pure --num-steps 50000 --md-log-frequency 2500 --sampling-method md --device cuda $run
done
