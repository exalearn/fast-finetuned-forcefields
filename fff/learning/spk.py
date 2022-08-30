"""Utilities for using models based on SchNet"""
from tempfile import TemporaryDirectory
from typing import Union, List, Tuple
from time import monotonic
from pathlib import Path
from io import BytesIO
import pickle as pkl
import os

from schnetpack.data import AtomsData, AtomsLoader
from schnetpack import train as trn
import schnetpack as spk
from torch import optim
import pandas as pd
import numpy as np
import torch
import ase


class TorchMessage:
    """Send a PyTorch object via pickle, enable loading on to target hardware"""

    def __init__(self, model: torch.nn.Module):
        """
        Args:
            model: Model to be sent
        """
        self.model = model
        self._pickle = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # Save the model with pickle
        model_pkl = BytesIO()
        torch.save(self.model, model_pkl)

        # Store it
        state['model'] = None
        state['_pickle'] = model_pkl.getvalue()
        return state

    def get_model(self, map_location: Union[str, torch.device] = 'cpu'):
        """Load the cached model into memory

        Args:
            map_location: Where to copy the device
        Returns:
            Deserialized model, moved to the target resource
        """
        if self.model is None:
            self.model = torch.load(BytesIO(self._pickle), map_location=map_location)
            self.model.to(map_location)
            self._pickle = None
        else:
            self.model.to(map_location)
        return self.model


class SPKCalculatorMessage:
    """Tool for sending a SPK calculator to a remote system"""

    def __init__(self, model: torch.nn.Module):
        """
        Args:
            model: Model to be sent
        """
        self._model = pkl.dumps(TorchMessage(model))
        self.model = None  # Only loaded when deserialized

    def __getstate__(self):
        state = self.__dict__.copy()
        state['model'] = None  # The model never gets sent unserialized
        return state

    def load(self, device: str) -> spk.interfaces.SpkCalculator:
        """Unload a copy of the calculator

        Args:
            device: Device on which to load the calculator
        """
        # Load it in if we are not done already
        if self.model is None:
            self.model = pkl.loads(self._model)
            self._model = None

        # Return the object
        return spk.interfaces.SpkCalculator(
            self.model.get_model(device),
            energy='energy', forces='forces'
        )


def ase_to_spkdata(atoms: List[ase.Atoms], path: Path) -> AtomsData:
    """Add a list of Atoms objects to a SchNetPack database

    Args:
        atoms: List of Atoms objects
        path: Path to the database file
    Returns:
        A link to the database
    """

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


def evaluate_schnet(model: Union[TorchMessage, torch.nn.Module, Path],
                    atoms: List[ase.Atoms],
                    batch_size: int = 64,
                    device: str = 'cpu') -> Tuple[List[float], List[np.ndarray]]:
    """Run inference for a series of structures

    Args:
        model: Model to evaluate. Either a SchNet model or the bytes corresponding to a serialized model
        atoms: List of structures to evaluate
        batch_size: Number of molecules to evaluate per batch
        device: Device on which to run the computation
    Returns
    """

    # Make sure the models are converted to Torch models
    if isinstance(model, TorchMessage):
        model = model.get_model(map_location='cpu')
    elif isinstance(model, (Path, str)):
        model = torch.load(model, map_location='cpu')  # Load to main memory first

    # Make the dataset
    with TemporaryDirectory() as td:

        # Save the data to an ASE Atoms database
        run_file = os.path.join(td, 'run_data.db')
        db = AtomsData(run_file, available_properties=[])
        db.add_systems(atoms, [{} for _ in atoms])

        # Build the data loader
        loader = AtomsLoader(db, batch_size=batch_size)

        # Run the models
        energies = []
        forces = []
        model.to(device)  # Move the model to the device
        for batch in loader:
            # Push the batch to the device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Run it and save results
            pred = model(batch)
            energies.extend(pred['energy'].detach().cpu().numpy()[:, 0].tolist())
            forces.extend(pred['forces'].detach().cpu().numpy())

        return energies, forces


def train_schnet(model: Union[TorchMessage, torch.nn.Module, Path],
                 database: List[ase.Atoms],
                 num_epochs: int,
                 device: str = 'cpu',
                 batch_size: int = 32,
                 validation_split: float = 0.1,
                 random_state: int = 1,
                 learning_rate: float = 1e-3,
                 reset_weights: bool = True,
                 patience: int = None,
                 timeout: float = None) -> Union[Tuple[TorchMessage, pd.DataFrame],
                                                 Tuple[TorchMessage, pd.DataFrame, List[float]]]:
    """Train a SchNet model

    Args:
        model: Model to be retrained
        database: Mapping of XYZ format structure to property
        num_epochs: Number of training epochs
        device: Device (e.g., 'cuda', 'cpu') used for training
        batch_size: Batch size during training
        validation_split: Fraction to training set to use for the validation loss
        random_state: Random seed used for generating validation set and bootstrap sampling
        learning_rate: Initial learning rate for optimizer
        reset_weights: Whether to reset the weights before training
        patience: Patience until learning rate is lowered. Default: epochs / 8
        timeout: Maximum training time in seconds
    Returns:
        - model: Retrained model
        - history: Training history
        - test_pred: Predictions on ``test_set``, if provided
    """

    # Make sure the models are converted to Torch models
    if isinstance(model, TorchMessage):
        model = model.get_model(device)
    elif isinstance(model, (Path, str)):
        model = torch.load(model, map_location='cpu')  # Load to main memory first

    # If desired, re-initialize weights
    if reset_weights:
        for module in model.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    # Make the training and validation splits
    rng = np.random.RandomState(random_state)
    train_split = rng.rand(len(database)) > validation_split
    train = [a for a, x in zip(database, train_split) if x]
    valid = [a for a, x in zip(database, train_split) if not x]

    # Start the training process
    with TemporaryDirectory() as td:
        # Save the data to an ASE Atoms database
        train_file = Path(td) / 'train_data.db'
        train_db = ase_to_spkdata(train, train_file)
        train_loader = AtomsLoader(train_db, batch_size=batch_size, shuffle=True)

        valid_file = Path(td) / 'valid_data.db'
        valid_db = ase_to_spkdata(valid, valid_file)
        valid_loader = AtomsLoader(valid_db, batch_size=batch_size)

        # Make the trainer
        opt = optim.Adam(model.parameters(), lr=learning_rate)

        # Make a loss function based on force and energy
        rho_tradeoff = 0.1  # Between force and energy

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

        if timeout is not None:
            hooks.append(TimeoutHook(timeout))

        trainer = trn.Trainer(
            model_path=td,
            model=model,
            hooks=hooks,
            loss_fn=loss,
            optimizer=opt,
            train_loader=train_loader,
            validation_loader=valid_loader,
            checkpoint_interval=num_epochs + 1  # Turns off checkpointing
        )

        trainer.train(device, n_epochs=num_epochs)

        # Load in the best model
        model = torch.load(os.path.join(td, 'best_model'))

        # Move the model off of the GPU to save memory
        if 'cuda' in device:
            model.to('cpu')

        # Load in the training results
        train_results = pd.read_csv(os.path.join(td, 'log.csv'))

        return TorchMessage(model), train_results
