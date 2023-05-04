#! /bin/bash
# Trains a model and runs sampling using the settings we've selected for TTM models

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

for run in $@; do
    echo $run
    python $script_dir/train-model.py --learning-rate 1e-4 --num-epochs 256 --max-force 10 $run
    python $script_dir/run-sampling.py --num-steps 10 --device cuda --endpoint 698fba9a-4b12-4e0b-b83a-be6ded509946 --psi4-threads 64 $run
done
