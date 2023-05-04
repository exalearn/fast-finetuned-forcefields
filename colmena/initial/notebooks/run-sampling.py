"""Run sampling using the trained model and compute the energies using the ground truth"""
import os
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from csv import DictWriter
from functools import partial, update_wrapper
from pathlib import Path
import shutil
import json
from random import shuffle

import ase
from globus_compute_sdk import Executor
from tqdm import tqdm
import pandas as pd
import torch
from ttm.ase import TTMCalculator

from fff.sampling.mctbp import MCTBP
from fff.learning.gc.ase import SchnetCalculator
from fff.simulation import run_calculator
from fff.simulation.utils import read_from_string, write_to_string

structures_to_sample = Path(__file__).parent / '../../../notebooks/nwchem-evaluation/example_structures.csv'

if __name__ == "__main__":
    # Get input arguments
    parser = ArgumentParser()
    parser.add_argument('--num-steps', default=100, type=int, help='Number of sampling steps to run')
    parser.add_argument('--device', default='cpu', help='Device on which to run forcefield')
    parser.add_argument('--overwrite', action='store_true', help='Whether to re-do an existing computation')
    parser.add_argument('--endpoint', default=None, help='If provided, will execute computation on this endpoint')
    parser.add_argument('--psi4-threads', default=os.cpu_count(), type=int, help='Number of threads to use for each Psi4 invocation')
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
        target_calc = dict(calc='psi4', method='pbe0', basis='aug-cc-pvdz', num_threads=args.psi4_threads)
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
    all_structures: list[tuple[ase.Atoms, dict]] = []
    for _, row in tqdm(structures.iterrows(), total=len(structures), desc='Sampling'):
        # Run sampling for the prescribed number of steps
        atoms = read_from_string(row['xyz'], 'xyz')

        # Produce the minima
        _, structures = mctbp.run_sampling(atoms, args.num_steps, calc,
                                           device=args.device, random_seed=1)

        # Put in them a list of structures to be run
        for i, new_atoms in enumerate(structures):
            all_structures.append((
                new_atoms.copy(),
                {
                    'structure': row['id'],
                    'n_waters': row['n_waters'],
                    'minimum_num': i,
                    'xyz': write_to_string(new_atoms, 'xyz'),
                    'ml_energy': new_atoms.get_potential_energy(),
                    'ml_forces': json.dumps(new_atoms.get_forces().tolist()),
                }))

    # Shuffle them to get a mix of completion times
    shuffle(all_structures)

    # Run the structures
    @contextmanager
    def _make_exec():
        if args.endpoint is None:
            output = ThreadPoolExecutor(1)
        else:
            output = Executor(endpoint_id=args.endpoint)
        try:
            yield output
        finally:
            output.shutdown(cancel_futures=True, wait=False)


    with out_csv.open('w') as fp:
        writer = DictWriter(fp, fieldnames=['structure', 'n_waters', 'minimum_num', 'xyz', 'ml_energy', 'ml_forces', 'target_energy', 'target_forces'])
        writer.writeheader()

        # Submit everything
        func = partial(run_calculator)
        update_wrapper(func, run_calculator)

        with _make_exec() as executor:
            futures = []


            def _submit(my_atoms, my_metadata):
                xyz = write_to_string(my_atoms, 'xyz')
                output = executor.submit(func, xyz, target_calc)
                output.metadata = my_metadata
                output.atoms = my_atoms
                return output


            for atoms, metadata in all_structures:
                futures.append(_submit(atoms, metadata))

            # Wait until they complete
            pbar = tqdm(total=len(futures), desc='Auditing')
            while len(futures) > 0:
                # Wait until one finishes
                done, futures = wait(futures, return_when=FIRST_COMPLETED)

                for future in done:
                    if future.exception() is not None:
                        exc = future.exception()
                        if 'Task failure due to loss of manager' in str(exc):
                            print('Resubmitting a task that failed because of manager lost')
                            futures.add(_submit(future.atoms, future.metadata))
                        else:
                            executor.shutdown(cancel_futures=True)  # End ungracefully
                            raise future.exception()

                    else:
                        pbar.update(1)
                        metadata = future.metadata
                        atoms = read_from_string(future.result(), 'json')
                        metadata['target_energy'] = atoms.get_potential_energy()
                        metadata['target_forces'] = json.dumps(atoms.get_forces().tolist())

                        writer.writerow(metadata)
