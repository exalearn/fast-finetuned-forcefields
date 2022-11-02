"""Functions that use the model through interfaces designed for workflow engines"""
import time
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple, Union, Optional

import ase
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch_geometric.data import DataLoader, Data

from fff.learning.gc.data import AtomsDataset
from fff.learning.gc.models import SchNet


# TODO (wardlt): Make a common base class for these functions
def evaluate_schnet(model: SchNet,
                    atoms: list[ase.Atoms],
                    batch_size: int = 64,
                    device: str = 'cpu') -> Tuple[list[float], list[np.ndarray]]:
    """Run inference for a series of structures

    Args:
        model: Model to evaluate. Either a SchNet model or the bytes corresponding to a serialized model
        atoms: List of structures to evaluate
        batch_size: Number of molecules to evaluate per batch
        device: Device on which to run the computation
    Returns:
        - Energies for each inference
        - Forces for each inference
    """

    # Place the model on the GPU in eval model
    model.eval()
    model.to(device)

    with TemporaryDirectory() as tmp:
        # Make the data loader
        dataset = AtomsDataset.from_atoms(atoms, root=tmp)
        loader = DataLoader(dataset, batch_size=batch_size)

        # Run all entries
        energies = []
        forces = []
        for batch in loader:
            # Move the data to the array
            batch.to(device)

            # Get the energies then compute forces with autograd
            energ_batch, force_batch = _eval_batch(model, batch)

            # Split the forces
            n_atoms = batch.n_atoms.detach().numpy()
            forces_np = force_batch.detach().cpu().numpy()
            forces_per = np.split(forces_np, np.cumsum(n_atoms)[:-1])

            # Add them to the output lists
            energies.extend(energ_batch.detach().cpu().numpy().tolist())
            forces.extend(forces_per)

    model.to('cpu')  # Move it off the GPU memory

    return energies, forces


def _eval_batch(model: SchNet, batch: Data) -> (torch.Tensor, torch.Tensor):
    """Get the energies and forces for a certain batch of molecules

    Args:
        model: Model to evaluate
        batch: Batch of data to evaluate
    Returns:
        Energy and forces for the batch
    """
    batch.pos.requires_grad = True
    energ_batch = model(batch)
    force_batch = torch.autograd.grad(energ_batch, batch.pos, grad_outputs=torch.ones_like(energ_batch), retain_graph=True)[0]
    return energ_batch, force_batch


# TODO (wardlt): Pass the optimizer as an argument?
def train_schnet(model: SchNet,
                 training_data: list[ase.Atoms],
                 validation_data: list[ase.Atoms],
                 num_epochs: int,
                 device: str = 'cpu',
                 batch_size: int = 32,
                 learning_rate: float = 1e-3,
                 reset_weights: bool = False,
                 patience: int = None) -> Tuple[SchNet, pd.DataFrame]:
    """Train a SchNet model

    Args:
        model: Model to be retrained
        training_data: Structures used for training
        validation_data: Structures used for validation
        num_epochs: Number of training epochs
        device: Device (e.g., 'cuda', 'cpu') used for training
        batch_size: Batch size during training
        learning_rate: Initial learning rate for optimizer
        reset_weights: Whether to reset the weights before training
        patience: Patience until learning rate is lowered. Default: epochs / 8
    Returns:
        - model: Retrained model
        - history: Training history
    """

    # If desired, re-initialize weights
    if reset_weights:
        for module in model.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    # Start the training process
    with TemporaryDirectory(prefix='spk') as td:
        td = Path(td)
        # Save the batch to an ASE Atoms database
        train_dataset = AtomsDataset.from_atoms(training_data, td / 'train')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        valid_dataset = AtomsDataset.from_atoms(validation_data, td / 'valid')
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        # Make the trainer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if patience is None:
            patience = num_epochs // 8
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.8, min_lr=1e-6)

        # Loop over epochs
        log = []
        model.train()
        model.to(device)
        start_time = time.perf_counter()
        for epoch in range(num_epochs):
            # Get the

            # Iterate over all batches in the training set
            optimizer.zero_grad()
            train_losses = defaultdict(list)
            for batch in train_loader:
                batch.to(device)
                # TODO (wardlt): The original implementation made a second data loader. I don't know what that's about
                energy, force = _eval_batch(model, batch)

                # Get the forces in energy and forces
                energy_loss = F.mse_loss(energy, batch.y)
                force_loss = F.mse_loss(force, batch.f)
                total_loss = energy_loss + force_loss

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
                energy, force = _eval_batch(model, batch)
                energy_loss = F.mse_loss(energy, batch.y)
                force_loss = F.mse_loss(force, batch.f)
                total_loss = energy_loss + force_loss

                with torch.no_grad():
                    valid_losses['valid_loss_force'].append(force_loss.item())
                    valid_losses['valid_loss_energy'].append(energy_loss.item())
                    valid_losses['valid_loss_total'].append(total_loss.item())

            valid_losses = dict((k, np.mean(v)) for k, v in valid_losses.items())

            # Reduce the learning rate
            scheduler.step(valid_losses['valid_loss_total'])

            # Store the log line
            log.append({'epoch': epoch, 'time': time.perf_counter() - start_time, **train_losses, **valid_losses})

        return model, pd.DataFrame(log)
