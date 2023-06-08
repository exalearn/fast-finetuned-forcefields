"""Train an intiial model"""
import json
import platform
import shutil
from hashlib import sha256

import schnetpack as spk
from schnetpack import train as trn
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


if __name__ == "__main__":

    # Get the path to the run
    parser = ArgumentParser()

    # Things about how the model is trained
    parser.add_argument("--small-database", action='store_true', help='Whether to test on a smaller subset')
    parser.add_argument("--device", default='cpu', help='On which device to train')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--to-memory', action='store_true', help='Store the database in the memory')
    group.add_argument('--to-tempdisk', action='store_true', help='Store the database in a temporary folder')

    # Things about the model itself
    parser.add_argument('--num-epochs', default=64, type=int, help='Number of training epochs')
    parser.add_argument('--num-features', default=32, type=int, help='Number of features for atoms')
    parser.add_argument('--num-interactions', default=3, type=int, help='Number of message passing steps')
    parser.add_argument('--cutoff-distance', default=8., type=float, help='Cutoff distance for neighbors')

    args = parser.parse_args()

    # Determine the location of a temporary path
    if args.to_memory:
        temp_path = '/dev/shm'
    else:
        temp_path = None

    # Determine which file we use as the training data
    train_name = 'test.db' if args.small_database else 'train.db'

    # Create a temporary directory, even if it is not used
    with TemporaryDirectory(prefix='spk', dir=temp_path) as temp_path:
        # Copy the data if needed
        temp_path = Path(temp_path)
        data_dir = Path('./data/')
        if args.to_memory or args.to_tempdisk:
            print('Copying data to temporary folder')
            for f in [train_name, 'valid.db']:
                shutil.copyfile(data_dir / f, temp_path / f)
            data_dir = temp_path

        # Make the data loaders
        train_path = data_dir / train_name
        print(f'Loading training data from {train_path}')
        train_data = spk.AtomsData(str(train_path))
        valid_data = spk.AtomsData(str(data_dir / 'valid.db'))

        train_loader = spk.AtomsLoader(train_data, batch_size=64, shuffle=True, num_workers=8,
                                       pin_memory=args.device != "cpu")
        valid_loader = spk.AtomsLoader(valid_data, batch_size=64, num_workers=8, pin_memory=args.device != "cpu")

        # Get the statistics (if not already done)
        stats_path = Path('data') / f'{train_name[:-3]}.stats.pth'
        if stats_path.exists():
            print(f'Loading stats from: {stats_path}')
            means, stddevs = torch.load(stats_path)
        else:
            print('Computing stats from scratch')
            means, stddevs = train_loader.get_statistics(
                'energy', divide_by_atoms=True
            )
            torch.save([means, stddevs], stats_path)

        # Make the model
        n_features = args.num_features
        schnet = spk.representation.SchNet(
            n_atom_basis=n_features,
            n_filters=n_features,
            n_gaussians=32,
            n_interactions=args.num_interactions,
            cutoff=args.cutoff_distance,
            cutoff_network=spk.nn.cutoff.CosineCutoff
        )
        energy_model = spk.atomistic.Atomwise(
            n_in=n_features,
            property='energy',
            mean=means['energy'],
            stddev=stddevs['energy'],
            derivative='forces',
            negative_dr=True
        )
        model = spk.AtomisticModel(representation=schnet, output_modules=energy_model)

        # Define the training system
        rho_tradeoff = 0.9

        def loss(batch, result):
            # compute the mean squared error on the energies
            diff_energy = batch['energy'] - result['energy']
            err_sq_energy = torch.mean(diff_energy ** 2)

            # compute the mean squared error on the forces
            diff_forces = batch['forces'] - result['forces']
            err_sq_forces = torch.mean(diff_forces ** 2)

            # build the combined loss function
            err_sq = rho_tradeoff * err_sq_energy + (1 - rho_tradeoff) * err_sq_forces

            return err_sq


        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

        # set up metrics
        metrics = [
            spk.metrics.MeanAbsoluteError('energy'),
            spk.metrics.MeanAbsoluteError('forces')
        ]

        # Make the run directory
        hasher = sha256()
        run_params = args.__dict__
        run_params['host'] = platform.node()
        run_hash = sha256(json.dumps(run_params).encode()).hexdigest()[:6]
        run_dir = (
                Path('runs') /
                f'{"small" if args.small_database else "full"}-N{args.num_features}-n{args.num_epochs}-{run_hash}'
        )
        run_dir.mkdir(parents=True)

        with open(run_dir / 'params.json', 'w') as fp:
            json.dump(run_params, fp)

        # construct hooks
        hooks = [
            trn.CSVHook(log_path=str(run_dir), metrics=metrics),
            trn.ReduceLROnPlateauHook(
                optimizer,
                patience=1, factor=0.8, min_lr=1e-6,
                stop_after_min=True
            )
        ]

        print(f'Start training in {run_dir}!')
        trainer = trn.Trainer(
            model_path=str(run_dir),
            model=model,
            hooks=hooks,
            loss_fn=loss,
            optimizer=optimizer,
            train_loader=train_loader,
            validation_loader=valid_loader,
        )

        trainer.train(device='cuda', n_epochs=args.num_epochs)
