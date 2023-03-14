from pathlib import Path

import ase
import numpy as np
from ase.calculators.lj import LennardJones
from ase.calculators.singlepoint import SinglePointCalculator
from ttm.ase import TTMCalculator

from fff.sampling.md import MolecularDynamics
from fff.sampling.mctbp import optimize_structure, MCTBP
from fff.sampling.mhm import MHMSampler


def test_md(atoms):
    calc = LennardJones()
    md = MolecularDynamics()
    md_atoms, traj = md.run_sampling(atoms, 1000, calc, timestep=1, log_interval=100, temperature=100)
    assert len(traj) == 9
    assert md_atoms.get_temperature() > 25

    # Ensure the structures are different
    assert not np.isclose(atoms.positions, md_atoms.positions).all()
    assert not np.isclose(traj[0].positions, atoms.positions).all()
    assert not np.isclose(traj[-1].positions, md_atoms.positions).all()
    assert not np.isclose(traj[0].positions, traj[-1].positions).all()

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

    # Test only returning the minima
    mctbp.return_minima_only = True
    _, traj_atoms = mctbp.run_sampling(cluster, 8, calc)
    assert len(traj_atoms) == 9  # An initial relax plus 8 more
    assert all(np.max(a.get_forces()) < 0.02 for a in traj_atoms)


def test_mhm(cluster, tmpdir):
    calc = TTMCalculator()
    mhm = MHMSampler(scratch_dir=tmpdir)
    min_atoms, traj_atoms = mhm.run_sampling(cluster, 4, calc)

    assert isinstance(min_atoms, ase.Atoms)
    assert len(traj_atoms) > 4
    assert any(a.info['mhm-source'].startswith('md') for a in traj_atoms)
    assert any(a.info['mhm-source'].startswith('qn') for a in traj_atoms)

    assert len(list(Path(tmpdir).glob("*"))) == 0, 'Did not clean up after self'

    # Try with returning minima only
    mhm.return_minima_only = True
    _, traj_atoms, = mhm.run_sampling(cluster, 4, calc)

    assert len(traj_atoms) <= 4  # Cannot find more than 4 unique minima
    assert all(np.max(a.get_forces()) < 0.02 for a in traj_atoms)
