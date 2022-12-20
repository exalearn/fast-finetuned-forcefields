"""Minima hopping method, as implemented in ASE"""
from itertools import chain
from tempfile import TemporaryDirectory
from pathlib import Path
import os

from ase.io import Trajectory
from ase.optimize.minimahopping import MinimaHopping
from ase.calculators.calculator import Calculator
import ase

from fff.sampling.base import CalculatorBasedSampler

_disallowed_kwargs = ['minima_traj']


class MHMSampler(CalculatorBasedSampler):
    """Run minima hopping and return all minima"""

    def _run_sampling(self, atoms: ase.Atoms, steps: int, calc: Calculator, **kwargs) -> (ase.Atoms, list[ase.Atoms]):
        # Store where we started
        old_path = Path.cwd()

        # Make sure the user isn't overloading any files I care about
        assert not any(x in kwargs for x in _disallowed_kwargs), f'kwargs may not contain: {", ".join(_disallowed_kwargs)}'

        # Attach the calculator
        atoms.set_calculator(calc)

        # Run all this in a subdirectory
        try:
            with TemporaryDirectory(dir=self.scratch_dir) as tmp:
                tmp = Path(tmp)
                os.chdir(tmp)

                # Make the minima hopping class with any user-provided options
                mhm = MinimaHopping(atoms=atoms, **kwargs)

                # Run for a certain number of iterations
                mhm(totalsteps=steps)

                # Get the structures that are output by the run
                with Trajectory(tmp / 'minima.traj', mode='r') as traj:
                    minima = [a for a in traj]

                # Get the structures that are produced along the way
                all_steps = []
                for f in chain(tmp.glob('qn[0-9]*.traj')):
                    with Trajectory(tmp / f, mode='r') as traj:
                        new_steps = [a for a in traj]
                        all_steps.extend(new_steps[1:-1])  # Not the first or last

                return minima[-1], minima[1:-1] + all_steps
        finally:
            os.chdir(old_path)  # Go back to where we started
