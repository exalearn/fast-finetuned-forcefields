"""Helps find the optimal parameters for training the model on TTM"""
from argparse import ArgumentParser
from csv import DictWriter
from hashlib import sha512
from pathlib import Path
import shutil
import gzip
import json
from time import perf_counter

import torch
import numpy as np
import pandas as pd
from ase.db import connect
from ase.io import read
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader

from fff.learning.gc.data import AtomsDataset
from fff.learning.gc.functions import eval_batch, GCSchNetForcefield

if __name__ == "__main__":
    # Make the options parser
    parser = ArgumentParser()

    group = parser.add_argument_group(title='Training Settings', description='Options related to training routine')
    group.add_argument('--huber-deltas', default=(0.1, 1), nargs=2, type=float, help='Delta parameter used in loss function')
    group.add_argument('--num-epochs', default=8, type=int, help='Number of training epochs')
    group.add_argument('--batch-size', default=32, type=int, help='Number of training entries per batch')
    group.add_argument('--learning-rate', default=1e-4, type=float, help='Initial training rate')
    group.add_argument('--lr-patience', default=2, type=int, help='Number of epochs without improvement before we lower learning rate')
    group.add_argument('--energy-weight', default=0.1, type=float, help='Fractional weight to energy during training (0<x<1)')
    group.add_argument('--max-force', default=10, type=float, help='Maximum allowed force value')

    parser.add_argument('--starting-model', default='md-100k-large')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    # Make the run directory
    config = args.__dict__.copy()
    config.pop('overwrite')
    config.pop('starting_model')
    run_hash = sha512(json.dumps(config).encode()).hexdigest()[-8:]
    run_dir = Path('runs') / args.starting_model / f'n{args.num_epochs}-lr{args.learning_rate:.1e}-{run_hash}'
    print(f'Saving run to {run_dir}')
    if run_dir.exists():
        if not args.overwrite:
            raise ValueError(f'Directory exists. Run with --overwrite to redo')
        else:
            shutil.rmtree(run_dir)

    # Unpack some inputs
    huber_eng, huber_force = args.huber_deltas

    # Split the data into subsets
    starting_ttm_dir = Path('starting-models') / args.starting_model
    train_data_dir = (run_dir / '..' / 'data' / f'maxF={args.max_force:.1f}').resolve()
    if not train_data_dir.exists():
        print(f'No training data found in {train_data_dir}')
        train_data_dir.mkdir(parents=True, exist_ok=True)
        with connect(starting_ttm_dir / 'train.db') as db:
            all_atoms = [a.toatoms() for a in db.select(f'fmax<{args.max_force}')]
        train_atoms, test_atoms = train_test_split(all_atoms, test_size=0.1)
        train_atoms, valid_atoms = train_test_split(train_atoms, test_size=0.1)
        for name, atoms in zip(['train', 'valid', 'test'], [train_atoms, valid_atoms, test_atoms]):
            with connect(train_data_dir / f'{name}.db', append=False) as db:
                for a in atoms:
                    db.write(a)
        del all_atoms, train_atoms, test_atoms, valid_atoms

    # Load in the training and validation data
    train_atoms = read(train_data_dir / 'train.db', ':')
    valid_atoms = read(train_data_dir / 'valid.db', ':')

    # Start the training process
    schnet = GCSchNetForcefield()
    model, train_log = schnet.train(
        starting_ttm_dir / 'starting_model.pth',
        train_atoms,
        valid_atoms,
        learning_rate=args.learning_rate,
        huber_deltas=args.huber_deltas,
        num_epochs=args.num_epochs,
        patience=args.lr_patience,
        device='cuda',
    )
    del train_atoms, valid_atoms

    # Save the results
    run_dir.mkdir(parents=True)
    with open(run_dir / 'config.json', 'w') as fp:
        json.dump(config, fp)

    train_log.to_csv(run_dir / 'train_log.csv', index=False)
    torch.save(model.get_model('cpu'), run_dir / 'best_model')

    # Validate the model against test set
    test_atoms = read(train_data_dir / 'test.db', ':')
    test_start_time = perf_counter()
    eng_pred, force_pred = schnet.evaluate(model, test_atoms, device='cuda')
    test_runtime = perf_counter() - test_start_time

    test_records = []
    test_record_file = run_dir / 'test_records.csv.gz'
    with gzip.open(test_record_file, 'wt') as fp:
        writer = DictWriter(fp, fieldnames=['n_atoms', 'energy_true', 'force_true', 'energy_ml', 'force_ml'])
        writer.writeheader()

        for atoms, me, mf in zip(test_atoms, eng_pred, force_pred):
            record = {
                'n_atoms': len(atoms),
                'energy_true': atoms.get_potential_energy(),
                'force_true': json.dumps(atoms.get_forces().tolist()),
                'energy_ml': me,
                'force_ml': json.dumps(mf.tolist()),
            }
            writer.writerow(record)

    # Compute some statistics
    test_records = pd.read_csv(test_record_file)
    test_records['energy_err'] = test_records['energy_ml'] - test_records['energy_true']
    test_records['energy_err_per_atom'] = test_records['energy_err'] / test_records['n_atoms']
    test_records['force_rmse'] = [
        np.sqrt(np.power(
            np.subtract(json.loads(x), json.loads(y)),
            2,
        ).mean())
        for x, y in zip(test_records['force_true'], test_records['force_ml'])
    ]
    test_records.to_csv(test_record_file, index=False)

    with open(run_dir / 'test_summary.json', 'w') as fp:
        json.dump({
            'test_runtime': test_runtime,
            'energy_mae': test_records['energy_err_per_atom'].abs().mean(),
            'force_rmse': test_records['force_rmse'].mean()
        }, fp)
