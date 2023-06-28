"""Test features related to simulation"""
from concurrent.futures import TimeoutError

from parsl import Config, ThreadPoolExecutor
from ase.calculators.nwchem import NWChem
from ase.calculators.lj import LennardJones
from pytest import mark, raises
import numpy as np
import parsl

from fff.simulation import run_calculator, write_to_string
from fff.simulation.ase import CachedCalculator, AsyncCalculator
from fff.simulation.utils import read_from_string


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


def test_same_process(atoms):
    calc = LennardJones()
    xyz = write_to_string(atoms, 'xyz')
    atoms_msg = run_calculator(xyz, calc, subprocess=False)
    atoms = read_from_string(atoms_msg, 'json')
    assert len(atoms) == 3
    assert 'forces' in atoms.calc.results  # Ensure that forces have been computed
    assert 'energy' in atoms.calc.results


def test_timeout(atoms):
    calc = LennardJones()
    xyz = write_to_string(atoms, 'xyz')
    with raises(TimeoutError):
        run_calculator(xyz, calc, timeout=1e-6)


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


def test_nwchem(atoms):
    """Test NWChem with rank replacement"""

    nwchem = NWChem(theory='scf', basis='3-21g', command='mpirun -np FFF_TOTAL_RANKS nwchem PREFIX.nwi > PREFIX.nwo')

    xyz = write_to_string(atoms, 'xyz')
    run_calculator(xyz, nwchem, ranks_per_node=2, subprocess=False)
