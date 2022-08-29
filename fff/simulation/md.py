"""Sampling structures using molecular dynamics"""
import os
from tempfile import TemporaryDirectory
from typing import Optional, List
from pathlib import Path
import logging

from ase.calculators.calculator import Calculator
from ase.md import VelocityVerlet
from ase.io import Trajectory
from ase import Atoms, units


logger = logging.getLogger(__name__)


def run_dynamics(atoms: Atoms, calc: Calculator, timestep: float, steps: int,
                 log_interval: int = 100, temp_dir: Optional[Path] = None) -> List[Atoms]:
    """Run molecular dynamics to produce a list of sampled structures

    Args:
        atoms: Initial structure
        calc: Calculator used to compute energy and forces
        timestep: Timestep size (units: fs)
        steps: Number of steps to run
        log_interval: Number of steps between storing structures
        temp_dir:
    Returns:
        - Final structure
    """

    # Set the calculator
    atoms.set_calculator(calc)

    # Define the dynamics
    dyn = VelocityVerlet(atoms, timestep=timestep * units.fs)

    # Store the trajectory data to a temporary directory
    with TemporaryDirectory(dir=temp_dir, prefix='fff') as tmp:
        # Move to the temporary directory so that no files written to disk overlap
        os.chdir(tmp)

        # Define the output path
        traj_path = Path(tmp) / "md.traj"
        logger.info(f'Writing trajectory to {traj_path}')
        props_to_write = ['energy', 'forces', 'stress']
        with Trajectory(str(traj_path), mode='w', atoms=atoms, properties=props_to_write) as traj:
            dyn.attach(traj, interval=log_interval)

            # Run the dynamics
            dyn.run(steps)

        # Read the trajectory back in and return the atoms
        return [x for x in Trajectory(str(traj_path), mode='r')]
