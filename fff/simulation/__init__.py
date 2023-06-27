"""Collections of Python functions for generating new training data"""
from concurrent.futures import ProcessPoolExecutor
from tempfile import TemporaryDirectory
from typing import Optional

from ase.calculators.calculator import Calculator, FileIOCalculator
from ase.calculators.psi4 import Psi4

from fff.simulation.utils import write_to_string, read_from_string


def run_calculator(xyz: str,
                   calc: Calculator | dict,
                   temp_path: Optional[str] = None,
                   subprocess: bool = True,
                   timeout: Optional[float] = None,
                   nodes: int = 1,
                   ranks_per_node: int = 1,
                   threads_per_rank: int = 1) -> str:
    """Run a computation



    Args:
        xyz: Cluster to evaluate
        calc: ASE calculator to use. Either the calculator object or a dictionary describing the settings (only Psi4 supported at the moment for dict)
        temp_path: Base path for the scratch files
        subprocess: Whether to run computation in a subprocess. Useful for calculators which do not exit cleanly
        timeout: How long to wait for result, used only if subprocess is True
        nodes: Number of nodes to use for this computation
        ranks_per_node: Number of ranks per node
        threads_per_rank: Number of threads to use per rank
    Returns:
        Atoms after the calculation in a JSON format
    """

    # Update the run command if needed
    def _update_command(command: str) -> str:
        """Replace placeholders with actual node values"""
        total_ranks = nodes * ranks_per_node
        return command.replace('FFF_NUM_NODES', str(nodes)).replace('FFF_RANKS_PER_NODE', str(ranks_per_node)) \
            .replace('FFF_TOTAL_RANKS', str(total_ranks)).replace('FFF_THREADS_PER_RANK', str(threads_per_rank))

    if isinstance(calc, FileIOCalculator):
        calc.command = _update_command(calc.command)
    elif isinstance(calc, dict) and 'command' in calc:
        calc['command'] = _update_command(calc['command'])

    # Some calculators do not clean up their resources well
    if subprocess:
        with ProcessPoolExecutor(max_workers=1) as exe:
            fut = exe.submit(_run_calculator, str(xyz), calc, temp_path)  # str ensures proxies are resolved
            return fut.result(timeout=timeout)
    else:
        return _run_calculator(str(xyz), calc, temp_path)


def _run_calculator(xyz: str, calc: Calculator | dict, temp_path: Optional[str] = None) -> str:
    """Runs the above function, designed to be run inside a new Process"""

    # Parse the atoms object
    atoms = read_from_string(xyz, 'xyz')
    with TemporaryDirectory(dir=temp_path, prefix='fff') as temp_dir:
        # Special case for Psi4 which sets the run directory on creating the object
        if isinstance(calc, dict):
            calc = calc.copy()
            assert calc.pop('calc') == 'psi4', 'only psi4 is supported for now'
            calc = Psi4(**calc, PSI_SCRATCH=temp_path, directory=temp_dir)

        # Update the run directory
        calc.directory = temp_dir

        # Run the calculation
        atoms.calc = calc
        try:
            atoms.get_forces()
            atoms.get_potential_energy()
        except BaseException as exc:
            raise ValueError(f'Calculation failed: {exc}')

        # Convert it to JSON
        return write_to_string(atoms, 'json')
