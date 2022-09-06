"""Collections of Python functions for generating new training data"""
from tempfile import TemporaryDirectory
from typing import Optional
import os

from ase.calculators.calculator import Calculator
from ase import Atoms


def run_calculator(atoms: Atoms, calc: Calculator, temp_path: Optional[str] = None) -> Atoms:
    """Run an NWChem computation on the requested cluster

    Args:
        atoms: Cluster to evaluate
        calc: ASE calculator to use
        temp_path: Base path for the scratch files
    Returns:
        Atoms after the calculation
    """

    atoms = atoms.copy()  # Make a duplicate
    with TemporaryDirectory(dir=temp_path, prefix='fff') as temp_dir:
        # Execute from the temp so that the calculators do not interfere
        os.chdir(temp_dir)

        # Run the calculation
        atoms.calc = calc
        atoms.get_forces()
        atoms.get_potential_energy()

        return atoms
