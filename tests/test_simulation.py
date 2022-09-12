"""Test features related to simulation"""
import pickle as pkl

from ase.calculators.lj import LennardJones
from ase.calculators.psi4 import Psi4
from ase.calculators.singlepoint import SinglePointCalculator
from ase.build import molecule
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pytest import fixture, mark

from fff.simulation import run_calculator, write_to_string
from fff.simulation.md import run_dynamics
from fff.simulation.utils import read_from_string


@fixture()
def atoms():
    return molecule('H2O')


@mark.parametrize(
    'calc', [LennardJones(), {'calc': 'psi4', 'method': 'hf', 'basis': 'sto-3g'}]
)
def test_single(calc, atoms):
    xyz = write_to_string(atoms, 'xyz')
    atoms_msg = run_calculator(xyz, calc)
    atoms = read_from_string(atoms_msg, 'json')
    assert len(atoms) == 3
    assert 'forces' in atoms.calc.results  # Ensure that forces have been computed
    assert 'energy' in atoms.calc.results


def test_md(atoms):
    calc = LennardJones()
    MaxwellBoltzmannDistribution(atoms, temperature_K=60)
    traj = run_dynamics(atoms, calc, 1, 1000, log_interval=100)
    assert len(traj) == 11

    # Make sure it has both the energy and the forces
    assert isinstance(traj[0].calc, SinglePointCalculator)  # SPC is used to store results
    assert traj[0].get_forces().shape == (3, 3)
    assert traj[0].get_total_energy()
    assert traj[0].get_total_energy() != traj[-1].get_total_energy()

