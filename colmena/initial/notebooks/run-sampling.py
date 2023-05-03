"""Run sampling using the trained model and compute the energies using the ground truth"""
from argparse import ArgumentParser
from csv import DictWriter
from pathlib import Path
import shutil
import json

from ase.calculators.psi4 import Psi4
from tqdm import tqdm
import pandas as pd
import torch
from ttm.ase import TTMCalculator

from fff.sampling.mctbp import MCTBP
from fff.learning.gc.ase import SchnetCalculator
from fff.simulation.utils import read_from_string, write_to_string

structures_to_sample = Path(__file__).parent / '../../../notebooks/nwchem-evaluation/example_structures.csv'

if __name__ == "__main__":
    # Get input arguments
    parser = ArgumentParser()
    parser.add_argument('--num-steps', default=100, type=int, help='Number of sampling steps to run')
    parser.add_argument('--device', default='cpu', help='Device on which to run forcefield')
    parser.add_argument('--overwrite', action='store_true', help='Whether to re-do an existing computation')
    parser.add_argument('run_dir', help='Path for the run being evaluated')
    args = parser.parse_args()

    # Find the model directory
    run_path = Path(args.run_dir)
    if run_path.name != 'final-model':
        run_path = run_path / 'final-model'
    assert run_path.is_dir(), f'No such directory: {run_path}'

    # Load in the options
    with open(run_path.parent / 'runparams.json') as fp:
        run_params = json.load(fp)
    if run_params['calculator'] == 'ttm':
        target_calc = TTMCalculator()
    elif run_params['calculator']:
        target_calc = Psi4(method='pbe0', basis='aug-cc-pvdz', num_threads=12)
    else:
        raise ValueError('Calculator not supported')

    # Load in the model and structures to be sampled
    model = torch.load(run_path / 'model', map_location='cpu')
    calc = SchnetCalculator(model)
    structures = pd.read_csv(structures_to_sample).query('n_waters > 6')

    # Save our run settings
    output_dir = run_path / 'mctbp'
    if output_dir.is_dir():
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            print('Already done!')
    output_dir.mkdir()
    with open(output_dir / 'params.json', 'w') as fp:
        json.dump(args.__dict__, fp)

    # Sample all structures
    out_csv = output_dir / 'sampled.csv'
    mctbp = MCTBP(scratch_dir=output_dir, return_minima_only=True)
    with out_csv.open('w') as fp:
        writer = DictWriter(fp, fieldnames=['structure', 'n_waters', 'minimum_num', 'xyz', 'ml_energy', 'ml_forces', 'target_energy', 'target_forces'])
        writer.writeheader()

        for _, row in tqdm(structures.iterrows(), total=len(structures), desc='Structures'):
            # Run sampling for the prescribed number of steps
            atoms = read_from_string(row['xyz'], 'xyz')

            # Produce the minima
            _, structures = mctbp.run_sampling(atoms, args.num_steps, calc,
                                               device=args.device, random_seed=1)

            # Compute their energies
            for i, new_atoms in enumerate(structures):
                target_atoms = new_atoms.copy()
                target_atoms.calc = target_calc
                target_atoms.get_forces()  # Caches result
                writer.writerow({
                    'structure': row['id'],
                    'n_waters': row['n_waters'],
                    'minimum_num': i,
                    'xyz': write_to_string(new_atoms, 'xyz'),
                    'ml_energy': new_atoms.get_potential_energy(),
                    'ml_forces': json.dumps(new_atoms.get_forces().tolist()),
                    'target_energy': target_atoms.get_potential_energy(),
                    'target_forces': json.dumps(target_atoms.get_forces().tolist()),
                })
