"""Test features related to simulation"""
import pickle as pkl

import numpy as np
from ase.calculators.lj import LennardJones
from ase.calculators.psi4 import Psi4
from ase.calculators.singlepoint import SinglePointCalculator
from ase.build import molecule
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import parsl
from parsl import Config, ThreadPoolExecutor
from pytest import fixture, mark
from ttm.ase import TTMCalculator

from fff.simulation import run_calculator, write_to_string
from fff.simulation.ase import CachedCalculator, AsyncCalculator
from fff.simulation.mctbp import optimize_structure, run_mctbp
from fff.simulation.md import run_dynamics
from fff.simulation.utils import read_from_string


@fixture()
def atoms():
    return molecule('H2O')


@fixture()
def cluster():
    xyz = """30

O       7.581982610000000     -0.663324770000000      5.483883860000000
H       8.362350460000000     -0.079370470000000      5.498567580000000
H       7.846055030000000     -1.464757200000000      5.041030880000000
O       9.456702229999999      1.642301080000000      8.570644379999999
H      10.114471399999999      1.655581000000000      9.261547090000001
H       9.181962009999999      2.562770840000000      8.428308489999999
O       9.664885520000000      1.027763610000000      5.758778100000000
H       9.485557560000000      1.914335850000000      5.411871910000000
H       9.760457990000001      1.144007330000000      6.710969450000000
O       6.000383380000000      4.009448050000000      7.349214080000000
H       5.983903880000000      4.025474550000000      6.383275510000000
H       5.536083220000000      3.203337670000000      7.608772750000000
O       4.833731170000000      1.482195020000000      7.883007530000000
H       5.628127100000000      0.955450120000000      8.084721569999999
H       4.134047510000000      1.149705890000000      8.438218120000000
O       7.110025880000000      0.051394890000000      8.205573080000001
H       7.262372970000000     -0.325556960000000      7.328944680000000
H       7.906465050000000      0.552607360000000      8.419908520000000
O       6.173881530000000      3.688445090000000      4.528872010000000
H       5.701079370000000      4.022632120000000      3.771772860000000
H       5.903837200000000      2.759263990000000      4.641063690000000
O       5.429551600000000      1.145089270000000      5.097751140000000
H       6.135000710000000      0.486118580000000      5.133624550000000
H       5.085167410000000      1.211727260000000      5.997292520000000
O       8.597597120000000      4.222480770000000      8.031750680000000
H       7.641802790000000      4.166800020000000      7.848542690000000
H       8.760176660000001      5.097825050000000      8.372748370000000
O       8.954336169999999      3.647526740000000      5.177083970000000
H       8.927373890000000      3.954237700000000      6.090421680000000
H       8.043264389999999      3.698852060000000      4.860042570000000
"""
    return read_from_string(xyz, 'xyz')


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


def test_optimize(atoms):
    calc = LennardJones()
    min_atoms, traj_atoms = optimize_structure(atoms, calc)

    assert min_atoms.get_forces().max() < 1e-3
    assert len(traj_atoms) > 0


def test_mctbp(cluster):
    calc = TTMCalculator()
    min_atoms, traj_atoms = run_mctbp(cluster, calc, 2)

    assert min_atoms.get_forces().max() < 5e-3  # Should be minimized
    assert len(traj_atoms) > 4


def test_cached(atoms, tmpdir):
    """Test a cached ASE database"""

    # Make a calculator with a cache
    calc = LennardJones()
    db_path = tmpdir / 'test.db'
    cached_calc = CachedCalculator(calc, db_path)

    # Run a calculation, verify that it should not
    result = cached_calc.get_forces(atoms)
    assert not cached_calc.last_lookup

    # Run again, make sure we get the same result, and it was from a cache
    cached_result = cached_calc.get_forces(atoms)
    assert cached_calc.last_lookup
    assert np.isclose(result, cached_result).all()

    # Verify that perturbing the structure gives a different result
    atoms.rattle()
    new_result = cached_calc.get_forces(atoms)
    assert not cached_calc.last_lookup
    assert not np.isclose(cached_result, new_result).all()
    assert cached_calc.get_db_size() == 2


def test_async(atoms, tmpdir):
    # Make a Parsl configuration
    config = Config(executors=[ThreadPoolExecutor()], run_dir=str(tmpdir / 'runinfo'))
    parsl.load(config)

    # Make a calculator
    calc = LennardJones()
    async_calc = AsyncCalculator(calc)

    # Run it both with the async and otherwise
    result = calc.get_forces(atoms)
    async_result = async_calc.get_forces(atoms)
    assert np.isclose(result, async_result).all()
