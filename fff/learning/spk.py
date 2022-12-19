"""Utilities for using models based on SchNet"""
from tempfile import TemporaryDirectory
from typing import List, Dict
from time import monotonic
from pathlib import Path
import os

from schnetpack.data import AtomsData, AtomsLoader
from schnetpack import train as trn
import schnetpack as spk
from six import BytesIO
from torch import optim
import pandas as pd
import numpy as np
import torch
import ase

from fff.learning.util.messages import TorchMessage
from .base import BaseLearnableForcefield, ModelMsgType


class SpkCalculator(spk.interfaces.SpkCalculator):
    """Subclass of the SchNetPack calculator that is better for distributed computing"""

    def __getstate__(self):
        # Remove the model from the serialization dictionary
        state = self.__dict__.copy()
        net = state.pop('model')

        # Serialize it using Torch's "save" functionality
        fp = BytesIO()
        torch.save(net, fp)
        state['_net'] = fp.getvalue()
        return state

    def __setstate__(self, state):
        # Load it back in
        fp = BytesIO(state.pop('_net'))
        state['model'] = torch.load(fp, map_location='cpu')
        self.__dict__ = state

    def to(self, device: str):
        """Move to model to a specific device"""
        self.model.to(device)


def ase_to_spkdata(atoms: List[ase.Atoms], path: Path) -> AtomsData:
    """Add a list of Atoms objects to a SchNetPack database

    Args:
        atoms: List of Atoms objects
        path: Path to the database file
    Returns:
        A link to the database
    """

    assert not Path(path).exists(), 'Path already exists'
    db = AtomsData(str(path), available_properties=['energy', 'forces'])

    # Get the properties as dictionaries
    prop_lst = []
    for a in atoms:
        props = {}
        # If we have the property, store it
        if a.calc is not None:
            calc = a.calc
            for k in ['energy', 'forces']:
                if k in calc.results:
                    props[k] = np.atleast_1d(calc.results[k])
        else:
            # If not, store a placeholder
            props.update(dict((k, np.atleast_1d([])) for k in ['energy', 'forces']))
        prop_lst.append(props)
    db.add_systems(atoms, prop_lst)
    return db


class TimeoutHook(trn.Hook):
    """Hook that stop training if a timeout limit has been reached"""

    def __init__(self, timeout: float):
        """
        Args:
            timeout: Maximum allowed training time
        """
        super().__init__()
        self.timeout = timeout
        self._start_time = None

    def on_train_begin(self, trainer):
        self._start_time = monotonic()

    def on_batch_end(self, trainer, train_batch, result, loss):
        if monotonic() > self._start_time + self.timeout:
            trainer._stop = True


class SchnetPackForcefield(BaseLearnableForcefield):
    """Forcefield based on the SchNetPack implementation of SchNet"""

    def __init__(self, scratch_dir: Path | None = None, timeout: float = None):
        """

        Args:
            scratch_dir: Directory in which to cache converted data
            timeout: Maximum training time
        """
        super().__init__(scratch_dir)
        self.timeout = timeout

    def evaluate(self,
                 model_msg: ModelMsgType,
                 atoms: list[ase.Atoms],
                 batch_size: int = 64,
                 device: str = 'cpu') -> (list[float], list[np.ndarray]):

        # Get the message
        model_msg = self.get_model(model_msg)

        # Make the dataset
        with TemporaryDirectory(dir=self.scratch_dir, prefix='spk') as td:
            # Save the data to an ASE Atoms database
            run_file = os.path.join(td, 'run_data.db')
            db = AtomsData(run_file, available_properties=[])
            db.add_systems(atoms, [{} for _ in atoms])

            # Build the data loader
            loader = AtomsLoader(db, batch_size=batch_size)

            # Run the models
            energies = []
            forces = []
            model_msg.to(device)  # Move the model to the device
            for batch in loader:
                # Push the batch to the device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Run it and save results
                pred = model_msg(batch)
                energies.extend(pred['energy'].detach().cpu().numpy()[:, 0].tolist())
                forces.extend(pred['forces'].detach().cpu().numpy())

            return energies, forces

    def train(self,
              model_msg: ModelMsgType,
              train_data: list[ase.Atoms],
              valid_data: list[ase.Atoms],
              num_epochs: int,
              device: str = 'cpu',
              batch_size: int = 32,
              learning_rate: float = 1e-3,
              huber_deltas: (float, float) = (0.5, 1),
              energy_weight: float = 0.9,
              reset_weights: bool = False,
              patience: int = None) -> (TorchMessage, pd.DataFrame):

        # Make sure the models are converted to Torch models
        model_msg = self.get_model(model_msg)

        # If desired, re-initialize weights
        if reset_weights:
            for module in model_msg.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()

        # Start the training process
        with TemporaryDirectory(dir=self.scratch_dir, prefix='spk') as td:
            # Save the data to an ASE Atoms database
            train_file = Path(td) / 'train_data.db'
            train_db = ase_to_spkdata(train_data, train_file)
            train_loader = AtomsLoader(train_db, batch_size=batch_size, shuffle=True, num_workers=8,
                                       pin_memory=device != "cpu")

            valid_file = Path(td) / 'valid_data.db'
            valid_db = ase_to_spkdata(train_data, valid_file)
            valid_loader = AtomsLoader(valid_db, batch_size=batch_size, num_workers=8, pin_memory=device != "cpu")

            # Make the trainer
            opt = optim.Adam(model_msg.parameters(), lr=learning_rate)

            # tradeoff
            rho_tradeoff = 0.9

            # loss function
            if huber_deltas is None:
                # Use mean-squared loss
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
            else:
                # Use huber loss
                delta_energy, delta_force = huber_deltas

                def loss(batch: Dict[str, torch.Tensor], result):
                    # compute the mean squared error on the energies per atom
                    n_atoms = batch['_atom_mask'].sum(axis=1)
                    err_sq_energy = torch.nn.functional.huber_loss(batch['energy'] / n_atoms,
                                                                   result['energy'].float() / n_atoms,
                                                                   delta=delta_energy)

                    # compute the mean squared error on the forces
                    err_sq_forces = torch.nn.functional.huber_loss(batch['forces'], result['forces'], delta=delta_force)

                    # build the combined loss function
                    err_sq = rho_tradeoff * err_sq_energy + (1 - rho_tradeoff) * err_sq_forces

                    return err_sq

            metrics = [
                spk.metrics.MeanAbsoluteError('energy'),
                spk.metrics.MeanAbsoluteError('forces')
            ]

            if patience is None:
                patience = num_epochs // 8
            hooks = [
                trn.CSVHook(log_path=td, metrics=metrics),
                trn.ReduceLROnPlateauHook(
                    opt,
                    patience=patience, factor=0.8, min_lr=1e-6,
                    stop_after_min=True
                )
            ]

            if self.timeout is not None:
                hooks.append(TimeoutHook(self.timeout))

            trainer = trn.Trainer(
                model_path=td,
                model=model_msg,
                hooks=hooks,
                loss_fn=loss,
                optimizer=opt,
                train_loader=train_loader,
                validation_loader=valid_loader,
                checkpoint_interval=num_epochs + 1  # Turns off checkpointing
            )

            trainer.train(device, n_epochs=num_epochs)

            # Load in the best model
            model_msg = torch.load(os.path.join(td, 'best_model'), map_location='cpu')

            # Load in the training results
            train_results = pd.read_csv(os.path.join(td, 'log.csv'))

            return TorchMessage(model_msg), train_results
