import numpy as np

from ase import build
from pytest import fixture

from fff.learning.gc.data import AtomsDataset
from fff.learning.gc.functions import evaluate_schnet, train_schnet
from fff.learning.gc.models import SchNet


@fixture
def model() -> SchNet:
    return SchNet(neighbor_method='radius')


def test_data_loader(example_waters, tmp_path):
    dataset = AtomsDataset.from_atoms(example_waters, root=tmp_path)

    # Make sure it gives the positions correctly
    mol = example_waters[0]
    pos = mol.get_positions() - mol.get_center_of_mass()
    assert np.isclose(dataset[0].pos, pos).all()


def test_run(model, tmp_path):
    water = build.molecule('H2O')
    energies, forces = evaluate_schnet(model, [water] * 4)
    assert len(energies) == 4
    assert forces[0].shape == (3, 3)
    assert len(forces)


def test_train(model, example_waters):
    model, log = train_schnet(model, example_waters[:8], example_waters[8:], 8)
    assert len(log) == 8
