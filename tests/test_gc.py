import numpy as np

from fff.learning.gc.data import AtomsDataset


def test_data_loader(example_waters, tmp_path):
    loader = AtomsDataset.from_atoms(example_waters, root=tmp_path)

    # Make sure it gives the positions correctly
    mol = example_waters[0]
    pos = mol.get_positions() - mol.get_center_of_mass()
    assert np.isclose(loader[0].pos, pos).all()
