import ase
import numpy as np
import pickle as pkl

import torch
from ase import build
from pytest import fixture

from fff.learning.gc.ase import SchnetCalculator
from fff.learning.gc.data import AtomsDataset
from fff.learning.gc.functions import GCSchNetForcefield
from fff.learning.gc.models import SchNet, load_pretrained_model


@fixture
def model() -> SchNet:
    return SchNet(neighbor_method='radius')


@fixture()
def ff():
    return GCSchNetForcefield()


def test_load_schnet(test_file_path):
    """Test loading a model from disk"""
    model = load_pretrained_model(test_file_path / 'example-schnet.pt', 1, 0.1, 10, device='cpu')
    assert model.mean == 1.
    assert model.std == 0.1
    assert model.atom_ref.weight.detach().numpy().max() == 0


def test_data_loader(example_waters, tmp_path):
    dataset = AtomsDataset.from_atoms(example_waters, root=tmp_path)

    # Make sure it gives the positions correctly
    mol = example_waters[0]
    pos = mol.get_positions() - mol.get_center_of_mass()
    assert np.isclose(dataset[0].pos, pos).all()


def test_run(model, ff, tmp_path):
    water = build.molecule('H2O')
    energies, forces = ff.evaluate(model, [water] * 4)
    assert len(energies) == 4
    assert forces[0].shape == (3, 3)
    assert len(forces)


def test_changing_energy_scales(test_file_path, ff):
    """Ensure that changing atomrefs and energy scales have the desired effect on model performance"""
    water = build.molecule('H2O')

    # Get the forces and energy without alternation
    model = load_pretrained_model(test_file_path / 'example-schnet.pt')
    orig_energy, orig_forces = ff.evaluate(model, [water])

    # Turn off the atomic reference energies
    model.atom_ref = None
    new_energy, new_forces = ff.evaluate(model, [water])
    assert np.isclose(new_energy, orig_energy).all()
    assert np.isclose(new_forces, orig_forces).all()

    # Try changing the mean energy
    model = load_pretrained_model(test_file_path / 'example-schnet.pt', mean=1, std=1)
    new_energy, new_forces = ff.evaluate(model, [water])
    assert np.isclose(new_energy, np.add(orig_energy, 3), atol=1e-3).all()
    assert np.isclose(new_forces, orig_forces).all()

    # Try changing the standard deviation
    model = load_pretrained_model(test_file_path / 'example-schnet.pt', mean=0, std=2)
    new_energy, new_forces = ff.evaluate(model, [water])
    assert np.isclose(new_energy, np.multiply(orig_energy, 2), atol=1e-2).all()
    assert np.isclose(new_forces, np.multiply(orig_forces, 2)).all()

    # Change the atomref for H
    model = load_pretrained_model(test_file_path / 'example-schnet.pt')
    with torch.no_grad():
        model.atom_ref.weight[1] = -1
    new_energy, new_forces = ff.evaluate(model, [water])
    assert np.isclose(np.subtract(new_energy, orig_energy), -2).all()
    assert np.isclose(new_forces, orig_forces).all()


def test_train(model, example_waters, ff):
    model, log = ff.train(model, example_waters[:8], example_waters[8:], 8)
    assert 'lr' in log.columns
    assert len(log) == 8


def test_ase(model, example_waters):
    atoms: ase.Atoms = example_waters[0]

    calc = SchnetCalculator(model)
    assert calc.device == 'cpu'
    atoms.set_calculator(calc)
    forces = calc.get_forces(atoms)
    numerical_forces = calc.calculate_numerical_forces(atoms, d=1e-4)
    assert np.isclose(forces, numerical_forces, atol=1e-2).all()

    calc2 = pkl.loads(pkl.dumps(calc))
    forces2 = calc2.get_forces(atoms)
    assert np.isclose(forces2, forces).all()
