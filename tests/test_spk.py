"""Testing SchNetPack"""
import pickle as pkl

import schnetpack as spk
from ase import build
import numpy as np
from pytest import fixture

from fff.learning.spk import ase_to_spkdata, SchnetPackForcefield, SpkCalculator
from fff.sampling.md import MolecularDynamics


@fixture
def schnet():
    n_features = 6

    schnet = spk.representation.SchNet(
        n_atom_basis=n_features,
        n_filters=n_features,
        n_gaussians=25,
        n_interactions=3,
        cutoff=5.,
        cutoff_network=spk.nn.cutoff.CosineCutoff
    )
    energy_model = spk.atomistic.Atomwise(
        n_in=n_features,
        property='energy',
        derivative='forces',
        negative_dr=True
    )
    return spk.AtomisticModel(representation=schnet, output_modules=energy_model)


def test_db(example_waters, tmp_path):
    assert len(ase_to_spkdata(example_waters, tmp_path / 'test.db')) == 10

    h2o = build.molecule('H2O')
    assert len(ase_to_spkdata([h2o], tmp_path / 'test2.db')) == 1


def test_train(example_waters, schnet):
    spk = SchnetPackForcefield()
    msg, log = spk.train(schnet, example_waters[:8], example_waters[8:], 2)
    assert len(log) == 2

    log = spk.train(schnet, example_waters[:8], example_waters[8:], 2, huber_deltas=(0.3, 3))
    assert len(log) == 2


def test_run(example_waters, schnet):
    spk = SchnetPackForcefield()
    energies, forces = spk.evaluate(schnet, example_waters, batch_size=2)
    assert np.shape(energies) == (10,)
    assert np.shape(forces) == (10, 3, 3)


def test_ase(schnet):
    h2o = build.molecule('H2O')
    calc = SpkCalculator(schnet, energy='energy', forces='forces')
    h2o.set_calculator(calc)
    h2o.get_forces()
    assert h2o.get_forces().shape == (3, 3)

    # Test running the dynamics
    traj = MolecularDynamics().run_sampling(h2o, 100, calc, timestep=0.1)
    assert len(traj) > 0

    # Test sending it as a message
    calc_2 = pkl.loads(pkl.dumps(calc))
    traj = MolecularDynamics().run_sampling(h2o, 100, calc_2, timestep=0.1)
    assert len(traj) > 0
