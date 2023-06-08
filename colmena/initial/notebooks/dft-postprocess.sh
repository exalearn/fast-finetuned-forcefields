#! /bin/bash
# Trains a model and runs sampling using the settings we've selected for TTM models

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

ep=08d55756-0d6f-445e-bb7d-0efd039c5aa7
for run in $@; do
    echo $run
    python $script_dir/train-model.py --learning-rate 1e-4 --num-epochs 256 --max-force 10 $run
    python $script_dir/run-sampling.py --num-steps 32 --device cuda --endpoint $ep --psi4-threads 64 $run
    python $script_dir/run-sampling.py --structure-source pure --num-steps 100 --md-log-frequency 10 --sampling-method md --device cuda --endpoint $ep --psi4-threads 64 $run
    python $script_dir/run-sampling.py --structure-source pure --num-steps 50000 --md-log-frequency 2500 --sampling-method md --device cuda --endpoint $ep --psi4-threads 64 $run
done
