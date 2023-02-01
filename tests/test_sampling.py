from pathlib import Path

import ase
from ase.calculators.lj import LennardJones
from ase.calculators.singlepoint import SinglePointCalculator
from ttm.ase import TTMCalculator

from fff.sampling.md import MolecularDynamics
from fff.sampling.mctbp import optimize_structure, MCTBP
from fff.sampling.mhm import MHMSampler


def test_md(atoms):
    calc = LennardJones()
    md = MolecularDynamics()
    atoms, traj = md.run_sampling(atoms, 1000, calc, timestep=1, log_interval=100, temperature=100)
    assert len(traj) == 9
    assert atoms.get_temperature() > 25

    # Make sure it has both the energy and the forces
    assert isinstance(traj[0].calc, SinglePointCalculator)  # SPC is used to store results
    assert traj[0].get_forces().shape == (3, 3)
    assert traj[0].get_total_energy()
    assert traj[0].get_total_energy() == traj[-1].get_total_energy()


def test_optimize(atoms):
    calc = LennardJones()
    min_atoms, traj_atoms = optimize_structure(atoms, calc)

    assert min_atoms.get_forces().max() < 1e-3
    assert len(traj_atoms) > 0


def test_mctbp(cluster):
    calc = TTMCalculator()
    mctbp = MCTBP()
    min_atoms, traj_atoms = mctbp.run_sampling(cluster, 2, calc)

    assert isinstance(min_atoms, ase.Atoms)
    assert len(traj_atoms) > 4


def test_mhm(cluster, tmpdir):
    calc = TTMCalculator()
    mhm = MHMSampler(scratch_dir=tmpdir)
    min_atoms, traj_atoms = mhm.run_sampling(cluster, 2, calc)

    assert isinstance(min_atoms, ase.Atoms)
    assert len(traj_atoms) > 4
    assert any(a.info['mhm-source'].startswith('md') for a in traj_atoms)
    assert any(a.info['mhm-source'].startswith('qn') for a in traj_atoms)

    assert len(list(Path(tmpdir).glob("*"))) == 0
