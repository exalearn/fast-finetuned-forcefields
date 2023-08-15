"""Evaluate the performance of CP2K on Polaris.

The script will let us change the number of nodes,
which MOF is used as inputs,
what type of calculation is performed,
and the number of steps"""
from argparse import ArgumentParser
from platform import node
from zipfile import ZipFile
from pathlib import Path
import json

from ase import Atoms, units
from ase.db import connect
from parsl import Config, HighThroughputExecutor, python_app
from parsl.launchers import SimpleLauncher
from parsl.providers import PBSProProvider
import pandas as pd
import parsl

from fff.simulation import read_from_string


@python_app
def run_cp2k(name: str, cp2k_opts: dict, atoms: Atoms, run_type: str, max_steps: int, scratch_dir: Path) -> tuple[list[Atoms], float, int]:
    """Run CP2K for a certain number of optimization or MD steps

    Args:
        name: Name of the structure
        calc: Options for the CP2k calculator
        atoms: Starting structure
        run_type: Which type of dynamics to run
        max_steps: Maximum number of steps to run
        scratch_dir: Directory in which to save files
    """
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.constraints import ExpCellFilter
    from ase.optimize import QuasiNewton
    from ase.md import VelocityVerlet
    from ase.io import Trajectory
    from ase import units
    from ase.calculators.cp2k import CP2K
    from time import perf_counter
    from pathlib import Path
    import re
    import os

    # Make the output directory
    start_dir = Path().cwd()
    run_dir = Path(scratch_dir) / 'run' / f'{run_type}-{name}'
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(run_dir)

        # Make the calculator
        start_time = perf_counter()
        calc = CP2K(**cp2k_opts)

        # Define the dynamics
        atoms.calc = calc
        if run_type == 'md':
            MaxwellBoltzmannDistribution(atoms, temperature_K=300)
            dyn = VelocityVerlet(atoms, timestep=1 * units.fs)
        elif run_type == 'qn':
            ecf = ExpCellFilter(atoms)
            dyn = QuasiNewton(ecf, logfile='opt.log')
        else:
            raise ValueError(f'Calculation type not supported: {run_type}')

        # Define the output path
        traj_path = "md.traj"
        props_to_write = ['energy', 'forces', 'momenta', 'stress']
        with Trajectory(str(traj_path), mode='w', atoms=atoms, properties=props_to_write) as traj:
            dyn.attach(traj, interval=1)

            # Run the dynamics
            dyn.run(steps=max_steps)

        # Read the trajectory back in and return the atoms
        atoms.calc = None
        output = []
        with Trajectory(str(traj_path), mode='r') as traj:
            output.extend([x for x in traj])

        # Kill the calculator by deleting the object to stop the underlying
        #  shell and then set the `_shell` parameter of the object so that the
        #  calculator object's destructor will skip the shell shutdown process
        #  when the object is finally garbage collected
        calc.__del__()
        calc._shell = None

        # Count the number of electrons
        text = Path('cp2k.out').read_text()
        n_electrons = sum(map(int, re.findall(r' Number of electrons:\s+(\d+)', text)[:2]))
    finally:
        os.chdir(start_dir)


    return output, perf_counter() - start_time, n_electrons


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--num-to-run', default=1, type=int, help='Number of MOFs to run')
    parser.add_argument('--run-every', default=10, type=int, help='How many steps to take in size between MOFs to run')
    parser.add_argument('--num-nodes', default=1, type=int, help='Number of nodes per job')
    parser.add_argument('--calculation-type', default='md', help='Which calculation type to run')
    parser.add_argument('--max-steps', default=10, type=int, help='Maximum number of steps to run')
    args = parser.parse_args()

    # Determine my hostname
    hostname = node().strip('-0123456789')
    print(f'Running on {hostname}')

    # Load the MOF list
    mofs = pd.read_csv('../data/qmof_database/qmof.csv')
    mofs.drop_duplicates('info.natoms', keep='first', inplace=True)
    mofs.sort_values('info.natoms', inplace=True)
    print(f'Loaded {len(mofs)} MOFs')

    mofs = mofs.iloc[::args.run_every].head(args.num_to_run)
    print(f'Loaded {len(mofs)} MOFs to run')

    # Set up Parsl
    config = Config(
        retries=0,
        executors=[HighThroughputExecutor(
            label='simulation',
            prefetch_capacity=0,
            start_method="fork",  # Needed to avoid interactions between MPI and os.fork
            max_workers=1,
            provider=PBSProProvider(
                account="CSC249ADCD08",
                worker_init=f"""
module reset
module swap PrgEnv-nvhpc PrgEnv-gnu
module load conda
module load cudatoolkit-standalone/11.4.4
module load cray-libsci cray-fftw
module list
cd $PBS_O_WORKDIR
hostname
pwd

# Load anaconda
conda activate /lus/grand/projects/CSC249ADCD08/fast-finetuned-forcefields/env-polaris
which python
""",
                walltime="12:00:00",
                queue="preemptable",
                scheduler_options="#PBS -l filesystems=home:eagle:grand",
                launcher=SimpleLauncher(),  # Launches only a single copy of the workflows
                select_options="ngpus=4",
                nodes_per_block=args.num_nodes,
                init_blocks=0,
                min_blocks=0,
                max_blocks=1,
                cpus_per_node=64,
            ),
        )],
    )
    parsl.load(config)

    # Make the calculator
    cp2k_opts = dict(
        inp="""&FORCE_EVAL
&DFT
  &XC
     &XC_FUNCTIONAL PBE
     &END XC_FUNCTIONAL
     &vdW_POTENTIAL
        DISPERSION_FUNCTIONAL PAIR_POTENTIAL
        &PAIR_POTENTIAL
            TYPE DFTD3(BJ)
            PARAMETER_FILE_NAME dftd3.dat
            REFERENCE_FUNCTIONAL PBE
        &END PAIR_POTENTIAL
    &END vdW_POTENTIAL
  &END XC
  &SCF
    MAX_DIIS  8
    EPS_SCF  1.0E-06
    &OT
      MINIMIZER  CG
      PRECONDITIONER  FULL_SINGLE_INVERSE
    &END OT
    &OUTER_SCF  T
      MAX_SCF  25
      EPS_SCF  1.0E-06
    &END OUTER_SCF
  &END SCF
  &MGRID
    ! CUTOFF 600
    REL_CUTOFF [Ry] 60.0
    COMMENSURATE TRUE
    NGRIDS 5
  &END MGRID
&END DFT
&END FORCE_EVAL""",
        basis_set_file='BASIS_MOLOPT',
        basis_set='DZVP-MOLOPT-SR-GTH',
        pseudo_potential='GTH-PBE',
        cutoff=1000 * units.Ry,
        xc=None,
        uks=True,
        command=f'mpiexec -n {args.num_nodes * 4} --ppn 4 --cpu-bind depth --depth 8 -env OMP_NUM_THREADS=8 '
                '/lus/grand/projects/CSC249ADCD08/cp2k/set_affinity_gpu_polaris.sh '
                '/lus/grand/projects/CSC249ADCD08/cp2k/cp2k-git/exe/local_cuda/cp2k_shell.psmp',
    )  # Use BLYP as we have PPs for it

    # Load the previous runs
    old_runs = pd.read_json('runtimes.json', lines=True)

    # Run the MOFs
    output_dir = Path('trajectories')
    output_dir.mkdir(exist_ok=True)
    for name in mofs['qmof_id']:
        # Skip if we've already ran it
        if len(old_runs) > 0 and len(old_runs.query(f'num_nodes=={args.num_nodes} and name=="{name}" and run_type=="{args.calculation_type}" and max_steps=={args.max_steps}')) > 0:
        #len(old_runs.query(f'num_nodes=={args.num_nodes} and name=="{name}" and run_type=="{args.calculation_type}" and max_steps=={args.max_steps} and hostname=="{hostname}"')) > 0:
            continue

        # Pull the relaxed structure
        with ZipFile('../data/qmof_database/relaxed_structures.zip') as zf:
            with zf.open(f'relaxed_structures/{name}.cif') as fp:
                atoms = read_from_string(fp.read().decode(), 'cif')

        # Run it
        try:
            traj, run_time, n_electrons = run_cp2k(name, cp2k_opts, atoms, args.calculation_type, args.max_steps, scratch_dir=Path('./tmp').resolve()).result()
        except ValueError:
            print(f'Failure for {name}')
            continue

        # Cound the number of electrons

        # Save the output to the runtime directory and append the run summary to a JSON file
        with connect(output_dir / f'{name}-{args.calculation_type}-{args.max_steps}.db', append=False) as db:
            for frame in traj:
                db.write(frame)
        with open('runtimes.json', 'a') as fp:
            print(json.dumps({
                'name': name,
                'composition': atoms.get_chemical_formula(),
                'n_atoms': len(atoms),
                'n_electrons': n_electrons,
                'num_nodes': args.num_nodes,
                'run_type': args.calculation_type,
                'max_steps': args.max_steps,
                'steps': len(traj),
                'hostname': hostname,
                'run_time': run_time
            }), file=fp)
