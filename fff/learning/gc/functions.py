"""Functions that use the model through interfaces designed for workflow engines"""
from tempfile import TemporaryDirectory
from typing import Tuple

import ase
import numpy as np
import torch
from torch_geometric.data import DataLoader

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
            batch.pos.requires_grad = True

            # Get the energies then compute forces with autograd
            energ_batch = model(batch)
            force_batch = torch.autograd.grad(energ_batch, batch.pos, grad_outputs=torch.ones_like(energ_batch), retain_graph=False)[0]

            # Split the forces
            n_atoms = batch.n_atoms.detach().numpy()
            forces_np = force_batch.detach().cpu().numpy()
            forces_per = np.split(forces_np, np.cumsum(n_atoms)[:-1])

            # Add them to the output lists
            energies.extend(energ_batch.detach().cpu().numpy().tolist())
            forces.extend(forces_per)

    model.to('cpu')  # Move it off the GPU memory

    return energies, forces
