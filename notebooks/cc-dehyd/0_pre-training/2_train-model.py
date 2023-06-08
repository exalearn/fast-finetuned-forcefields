from argparse import ArgumentParser
from csv import DictWriter
from tempfile import TemporaryDirectory
from contextlib import redirect_stderr
from collections import defaultdict
from hashlib import sha512
from pathlib import Path
import shutil
import gzip
import json
import time
import os
from time import perf_counter

import torch
import numpy as np
import pandas as pd
from ase import data
from torch.nn import functional as F
from torch_geometric.data import DataLoader

from fff.learning.gc.data import AtomsDataset
from fff.learning.gc.functions import eval_batch
from fff.learning.gc.models import SchNet

if __name__ == "__main__":
    # Make the options parser
    parser = ArgumentParser()

    group = parser.add_argument_group(title='Model Architecture', description='Options related to SchNet architecture')
    group.add_argument('--num-interactions', default=4, type=int, help='Number of message-passing steps')
    group.add_argument('--num-features', default=64, type=int, help='Number of features used to describe each node and edge')
    group.add_argument('--num-gaussians', default=32, type=int, help='Number of Gaussians describing the nearest neighbors')
    group.add_argument('--cutoff', default=6., type=float, help='Cutoff radius for the distance computer')

    group = parser.add_argument_group(title='Training Settings', description='Options related to training routine')
    group.add_argument('--train-size', default=None, type=int, help='Maximum number of entries to use for training')
    group.add_argument('--huber-deltas', default=(0.1, 1), nargs=2, type=float, help='Delta parameter used in loss function')
    group.add_argument('--num-epochs', default=8, type=int, help='Number of training epochs')
    group.add_argument('--batch-size', default=32, type=int, help='Number of training entries per batch')
    group.add_argument('--learning-rate', default=1e-4, type=float, help='Initial training rate')
    group.add_argument('--lr-patience', default=2, type=int, help='Number of epochs without improvement before we lower learning rate')
    group.add_argument('--energy-weight', default=0.1, type=float, help='Fractional weight to energy during training (0<x<1)')

    parser.add_argument('--method', default='wb97x_dz', help='Method used to create training data')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    # Make the run directory
    config = args.__dict__.copy()
    config.pop('overwrite')
    run_hash = sha512(json.dumps(config).encode()).hexdigest()[-8:]
    run_dir = Path('runs') / f'N{"full" if args.train_size is None else args.train_size}-n{args.num_epochs}-{run_hash}'
    if run_dir.exists():
        if not args.overwrite:
            raise ValueError(f'Directory exists. Run with --overwrite to redo')
        else:
            shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    with open(run_dir / 'config.json', 'w') as fp:
        json.dump(config, fp)

    # Make the model
    # TODO (wardlt): Allow users to configure hyperparams
    model = SchNet(
        num_features=args.num_features,
        num_gaussians=args.num_gaussians,
        num_interactions=args.num_interactions,
        cutoff=args.cutoff,
    )

    # Set the reference energies
    ref_energies = json.loads(Path('reference_energies.json').read_text())[args.method]
    for elem, eng in ref_energies['ref_energies'].items():
        model.set_reference_energy(elem, eng)
    model.mean = ref_energies['offsets']['mean']
    model.std = ref_energies['offsets']['std']

    # Unpack some inputs
    huber_eng, huber_force = args.huber_deltas

    # Start the training process
    train_data_dir = Path('data') / args.method
    assert train_data_dir.exists()
    with TemporaryDirectory(dir=run_dir, prefix='spk') as td:
        td = Path(td)
        # Save the batch to an ASE Atoms database
        with open(os.devnull, 'w') as fp, redirect_stderr(fp):
            train_dataset = AtomsDataset(train_data_dir / 'train.db', root=str(td / 'train'), max_size=args.train_size)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            print(f'Made a training dataset with {len(train_dataset)} entries')

            valid_dataset = AtomsDataset(train_data_dir / 'valid.db', root=str(td / 'valid'), max_size=args.train_size)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
            print(f'Made a validation dataset with {len(valid_dataset)} entries')

        # Make the trainer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_patience, factor=0.8, min_lr=1e-6)

        # Store the best loss
        best_loss = torch.inf

        # Loop over epochs
        log = []
        model.train()
        device = 'cuda'
        model.to(device)
        start_time = time.perf_counter()
        for epoch in range(args.num_epochs):
            # Iterate over all batches in the training set
            train_losses = defaultdict(list)
            for batch in train_loader:
                batch.to(device)

                optimizer.zero_grad()

                # Compute the energy and forces
                energy, force = eval_batch(model, batch)

                # Get the forces in energy and forces
                energy_loss = F.huber_loss(energy / batch.n_atoms, batch.y / batch.n_atoms, reduction='mean', delta=huber_eng)
                force_loss = F.huber_loss(force, batch.f, reduction='mean', delta=huber_force)
                total_loss = args.energy_weight * energy_loss + (1 - args.energy_weight) * force_loss

                # Iterate backwards
                total_loss.backward()
                optimizer.step()

                # Add the losses to a log
                with torch.no_grad():
                    train_losses['train_loss_force'].append(force_loss.item())
                    train_losses['train_loss_energy'].append(energy_loss.item())
                    train_losses['train_loss_total'].append(total_loss.item())

            # Compute the average loss for the batch
            train_losses = dict((k, np.mean(v)) for k, v in train_losses.items())

            # Get the validation loss
            valid_losses = defaultdict(list)
            for batch in valid_loader:
                batch.to(device)
                energy, force = eval_batch(model, batch)

                # Get the loss of this batch
                energy_loss = F.huber_loss(energy / batch.n_atoms, batch.y / batch.n_atoms, reduction='mean', delta=huber_eng)
                force_loss = F.huber_loss(force, batch.f, reduction='mean', delta=huber_force)
                total_loss = args.energy_weight * energy_loss + (1 - args.energy_weight) * force_loss

                with torch.no_grad():
                    valid_losses['valid_loss_force'].append(force_loss.item())
                    valid_losses['valid_loss_energy'].append(energy_loss.item())
                    valid_losses['valid_loss_total'].append(total_loss.item())

            valid_losses = dict((k, np.mean(v)) for k, v in valid_losses.items())

            # Reduce the learning rate
            scheduler.step(valid_losses['valid_loss_total'])

            # Save the best model if possible
            if valid_losses['valid_loss_total'] < best_loss:
                best_loss = valid_losses['valid_loss_total']
                torch.save(model, run_dir / 'best_model')

            # Store the log line
            log_line = {'epoch': epoch, 'time': time.perf_counter() - start_time, 'lr': optimizer.param_groups[0]['lr'],
                        **train_losses, **valid_losses}
            with open(run_dir / 'log.csv', 'a') as fp:
                writer = DictWriter(fp, fieldnames=list(log_line.keys()))
                if epoch == 0:
                    writer.writeheader()
                writer.writerow(log_line)

        # Done with the training and validation sets
        del train_dataset, train_loader, valid_dataset, valid_loader

        # Load the best model back in
        model = torch.load(run_dir / 'best_model', map_location=device)

        # Run the test set
        test_dataset = AtomsDataset(train_data_dir / 'test.db', root=str(td / 'test'))
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        test_records = []
        test_record_file = run_dir / 'test_records.csv.gz'
        with gzip.open(test_record_file, 'wt') as fp:
            test_start_time = perf_counter()
            writer = DictWriter(fp, fieldnames=['n_atoms', 'energy_true', 'force_true', 'energy_ml', 'force_ml'])
            writer.writeheader()

            for batch in test_loader:
                batch.to(device)
                energy, force = eval_batch(model, batch)

                # Split the forces into each training entry
                split = np.cumsum(batch.n_atoms.cpu())[:-1]  # Get the indices which mark the beginning of each new training entry
                force_true = np.split(batch.f.cpu(), split)
                force_ml = np.split(force.cpu(), split)

                for na, te, tf, me, mf in zip(batch.n_atoms, batch.y, force_true, energy, force_ml):
                    record = {
                        'n_atoms': na.item(),
                        'energy_true': te.item(),
                        'force_true': json.dumps(tf.detach().numpy().tolist()),
                        'energy_ml': me.item(),
                        'force_ml': json.dumps(mf.detach().numpy().tolist()),
                    }
                    writer.writerow(record)
            test_runtime = perf_counter() - test_start_time

        # Done with test loader
        del test_dataset, test_loader

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
