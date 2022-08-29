"""Collections of Python functions for generating new training data"""
from typing import Optional, Tuple

from ase.calculators.calculator import Calculator
from ase import Atoms


def run_calculator(atoms: Atoms, calc: Calculator, temp_path: Optional[str] = None) -> Tuple[Atoms, float]:
    """Run an NWChem computation on the requested cluster

    Args:
        atoms: Cluster to evaluate
        calc: ASE calculator to use
        temp_path: Base path for the scratch files
    Returns:
        Atoms after the calculation
    """
    from tempfile import TemporaryDirectory
    import time
    import os

    with TemporaryDirectory(dir=temp_path, prefix='fff') as temp_dir:
        # Execute from the temp so that the calculators do not interfere
        os.chdir(temp_dir)

        # Run the calculation
        start_time = time.perf_counter()
        atoms.set_calculator(calc)
        atoms.get_forces()
        run_time = time.perf_counter() - start_time

        return atoms, run_time
