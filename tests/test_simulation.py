"""Test features related to simulation"""

from ase.calculators.lj import LennardJones
from ase.calculators.singlepoint import SinglePointCalculator
from ase.build import bulk

from pytest import fixture


from fff.simulation.md import run_dynamics


@fixture()
def calc():
    return LennardJones()


@fixture()
def atoms():
    return bulk('Cu') * [4, 4, 4]


def test_md(calc, atoms):
    assert len(atoms) == 64
    traj = run_dynamics(atoms, calc, 1, 1000, log_interval=100)
    assert len(traj) == 11

    # Make sure it has both the energy and the forces
    assert isinstance(traj[0].calc, SinglePointCalculator)  # SPC is used to store results
    assert traj[0].get_forces().shape == (64, 3)
    assert traj[0].get_total_energy()
