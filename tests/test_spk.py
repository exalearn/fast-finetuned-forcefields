"""Testing SchNetPack"""
import pickle as pkl

import schnetpack as spk
from ase import build
import numpy as np
from pytest import fixture

from fff.learning.spk import ase_to_spkdata, train_schnet, evaluate_schnet, SPKCalculatorMessage
from fff.simulation.md import run_dynamics


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
    log = train_schnet(schnet, example_waters, 2)
    assert len(log) == 2


def test_run(example_waters, schnet):
    energies, forces = evaluate_schnet(schnet, example_waters, batch_size=2)
    assert np.shape(energies) == (10,)
    assert np.shape(forces) == (10, 3, 3)


def test_ase(schnet):
    h2o = build.molecule('H2O')
    calc = spk.interfaces.SpkCalculator(schnet, energy='energy', forces='forces')
    h2o.set_calculator(calc)
    h2o.get_forces()
    assert h2o.get_forces().shape == (3, 3)

    # Test running the dynamics
    traj = run_dynamics(h2o, calc, 0.1, 100)
    assert len(traj) > 0

    # Test sending it as a message
    msg = SPKCalculatorMessage(schnet)
    msg = pkl.loads(pkl.dumps(msg))
    traj = run_dynamics(h2o, msg, 0.1, 100)
    assert len(traj) > 0


