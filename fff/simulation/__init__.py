"""Collections of Python functions for generating new training data"""
from tempfile import TemporaryDirectory
from typing import Optional
import os

from ase.calculators.calculator import Calculator
from ase import Atoms
from ase.calculators.psi4 import Psi4

from fff.simulation.utils import write_to_string


def run_calculator(atoms: Atoms, calc: Calculator | dict, temp_path: Optional[str] = None) -> str:
    """Run an NWChem computation on the requested cluster

    Args:
        atoms: Cluster to evaluate
        calc: ASE calculator to use. Either the calculator object or a dictionary describing the settings
            (only Psi4 supported at the moment for dict)
        temp_path: Base path for the scratch files
    Returns:
        Atoms after the calculation in a JSON format
    """

    atoms = atoms.copy()  # Make a duplicate

    with TemporaryDirectory(dir=temp_path, prefix='fff') as temp_dir:
        # Execute from the temp so that the calculators do not interfere
        os.chdir(temp_dir)

        # Special case for Psi4 which sets the run directory on creating the object
        if isinstance(calc, dict):
            calc = calc.copy()
            assert calc.pop('calc') == 'psi4', 'only psi4 is supported for now'
            calc = Psi4(**calc, directory=temp_dir, PSI_SCRATCH=temp_path)

        # Run the calculation
        atoms.calc = calc
        atoms.get_forces()
        atoms.get_potential_energy()

        return write_to_string(atoms, 'json')
