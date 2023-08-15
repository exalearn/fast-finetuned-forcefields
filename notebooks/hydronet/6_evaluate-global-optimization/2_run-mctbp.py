"""Run MCTBP with a certain potential"""
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from fff.sampling.mctbp import MCTBP
from fff.simulation import read_from_string, write_to_string

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--calculator', default='ttm', help='Calculator to use for energies. Either TTM or the path to a SchNet model.')
    parser.add_argument('--num-steps', default=100000, help='Number of MCTBP steps to take', type=int)
    parser.add_argument('--num-to-keep', default=2, type=int,
                        help='Number of the lowest-energy structures to keep, and the number of randomly-selected minima')
    parser.add_argument('--num-per-size', default=8, type=int, help='Number of starting points to run per size')
    args = parser.parse_args()

    # Make the calculator
    if args.calculator == 'ttm':
        from ttm.ase import TTMCalculator

        calc = TTMCalculator()
        run_name = 'ttm'
    else:
        raise NotImplementedError()

    # Initialize the starting path
    out_directory = Path('mctbp-runs') / run_name / f'steps={args.num_steps}_starting={args.num_per_size}_count={args.num_to_keep}'
    out_directory.mkdir(parents=True, exist_ok=True)

    # Loop over all structures in the test set
    sampler = MCTBP(return_minima_only=True, progress_bar=True)
    example_structures = pd.read_csv('test-set-structures.csv')
    for size, group in example_structures.query('n_waters > 10').groupby('n_waters'):
        out_file = out_directory / f'waters={size}.csv'
        if out_file.is_file():
            continue

        for_target_size = []
        for ind, example_xyz in tqdm(enumerate(group.head(args.num_per_size)['xyz']), desc=f'size={size}', total=len(group)):
            atoms = read_from_string(example_xyz, 'xyz')

            # Run MCTBP
            _, all_strcs = sampler.run_sampling(atoms, args.num_steps, calc)
            for_target_size.extend(all_strcs)

        print(f'Produced {len(for_target_size)} of {size} waters')

        # Sort structures by energy and remove structures which agree within 0.01 meV/water
        top_structures = pd.DataFrame({'atoms': for_target_size})
        top_structures['energy'] = top_structures['atoms'].apply(lambda x: x.get_potential_energy())
        top_structures.sort_values('energy', ascending=True, inplace=True)

        is_unique = (top_structures['energy'].diff() / 3) > 0.01e-3
        is_unique[0] = True
        top_structures = top_structures[is_unique]
        print(f'Have a total of {len(top_structures)} energetically-unique structures for {size} waters')

        # Save the best structures and random from the remainder
        top_structures['source'] = 'all'
        best_structures = top_structures.head(args.num_per_size).copy()
        best_structures['source'] = 'best'

        random_structures = top_structures.iloc[args.num_per_size:].sample(args.num_per_size, replace=False).copy()
        random_structures['source'] = 'random'

        # Combine and save them including the XYZ
        to_save = pd.concat([best_structures, random_structures], ignore_index=True)
        to_save['xyz'] = to_save['atoms'].apply(lambda x: write_to_string(x, 'xyz'))

        to_save[['xyz', 'source', 'energy']].to_csv(out_file, index=False)
