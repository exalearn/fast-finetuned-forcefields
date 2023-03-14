"""Minima hopping method, as implemented in ASE"""
from functools import partialmethod
from itertools import chain
from tempfile import TemporaryDirectory
from pathlib import Path
import os

from ase.io import Trajectory
from ase.optimize import QuasiNewton
from ase.optimize.minimahopping import MinimaHopping
from ase.calculators.calculator import Calculator
import ase

from fff.sampling.base import CalculatorBasedSampler

_disallowed_kwargs = ['minima_traj']


class MHMSampler(CalculatorBasedSampler):
    """Run minima hopping and return all minima"""

    def __init__(self, return_minima_only: bool = False,
                 scratch_dir: Path | None = None):
        """

        Args:
            return_minima_only: Whether to only return minima produced at each step
            scratch_dir: Location in which to store temporary files
        """
        super().__init__(scratch_dir=scratch_dir)
        self.return_minima_only = return_minima_only

    def _run_sampling(self, atoms: ase.Atoms, steps: int, calc: Calculator, **kwargs) -> (ase.Atoms, list[ase.Atoms]):
        # Store where we started
        old_path = Path.cwd()

        # Make sure the user isn't overloading any files I care about
        assert not any(x in kwargs for x in _disallowed_kwargs), f'kwargs may not contain: {", ".join(_disallowed_kwargs)}'

        # Attach the calculator
        atoms.set_calculator(calc)

        # Run all this in a subdirectory
        with TemporaryDirectory(dir=self.scratch_dir, prefix='mhm-') as tmp:
            try:
                tmp = Path(tmp).absolute()
                os.chdir(tmp)

                # Make the minima hopping class with any user-provided options
                opt = QuasiNewton
                opt.run = partialmethod(opt.run, steps=100)
                mhm = MinimaHopping(atoms=atoms, optimizer=opt, **kwargs)

                # Run for a certain number of iterations
                mhm(totalsteps=steps)

                # Get the structures that are output by the run
                minima = []
                with Trajectory('minima.traj', mode='r') as traj:
                    for i, a in enumerate(traj):
                        a.info['mhm-source'] = f'min-{i}'
                        minima.append(a)

                # Get the structures that are produced along the way
                all_steps = []
                for f in chain(tmp.glob('md[0-9]*.traj'), tmp.glob('qn[0-9]*.traj')):
                    with Trajectory(f, mode='r') as traj:
                        new_steps = []
                        for i, a in enumerate(traj):
                            a.info['mhm-source'] = f'{f.name[:-5]}-{i}'
                            new_steps.append(a)
                        all_steps.extend(new_steps[1:-1])  # Not the first or last

                return minima[-1], minima[1:-1] + ([] if self.return_minima_only else all_steps)
            finally:
                os.chdir(old_path)  # Go back to where we started
