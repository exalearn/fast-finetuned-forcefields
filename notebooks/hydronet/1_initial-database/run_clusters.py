"""Run clusters with different basis sets and store results to an ASE database

It is presently hard-coded to run on Cori KNL nodes. You will need to change the Parsl configuration,
 the NWChem executable path, and the NWChem memory/core settings to adapt to a different system
"""

import json
import zipfile
from argparse import ArgumentParser
from concurrent.futures import as_completed
from io import TextIOWrapper, StringIO
from pathlib import Path
from typing import Iterable, Tuple, Optional

import ase
import pandas as pd
import parsl
from ase.calculators.nwchem import NWChem
from ase.db import connect
from ase.io import read
from parsl.dataflow.futures import AppFuture
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SimpleLauncher
from parsl.providers import SlurmProvider
from tqdm import tqdm

# Compute node information (settings for Perlmutter CPU)
cores_per_node = 128
memory_per_node = 500  # In GB
scratch_path = '/pscratch/sd/w/wardlt/nwchem-db/'  # Fix
disk_space = 1700  # In GB


@parsl.python_app
def run_nwchem(atoms: ase.Atoms, calc: NWChem, temp_path: Optional[str] = None) -> Tuple[ase.Atoms, float]:
    """Run an NWChem computation on the requested cluster

    Args:
        atoms: Cluster to evaluate
        calc: NWChem calculator to use
        temp_path: Base path for the scratch files
    Returns:
        Atoms after the calculation
    """
    from tempfile import gettempdir
    from hashlib import sha256
    from shutil import rmtree
    from pathlib import Path
    import time
    import os

    # Make a run directory based on the input XYZ
    run_hash = sha256(atoms.positions.tobytes()).hexdigest()[-8:]
    temp_dir = Path(temp_path or gettempdir()) / f'fff-{run_hash}'
    if (temp_dir / 'nwchem/nwchem.db').exists():
        calc.parameters['restart_kw'] = 'restart'

    # Update the scratch directory
    calc.directory = str(temp_dir)
    calc.scratch = str(temp_dir)
    calc.perm = str(temp_dir)

    # Run the calculation
    start_time = time.perf_counter()
    atoms.set_calculator(calc)
    atoms.get_forces()
    run_time = time.perf_counter() - start_time

    # Remove only if exiting successfully
    rmtree(temp_dir)

    return atoms, run_time


def generate_structures_from_zip(path: Path) -> Iterable[Tuple[str, ase.Atoms]]:
    """Iterate over all structures in Henry's ZIP file

    Yields:
        Tuple of (filename, ase.Atoms) object
    """

    with zipfile.ZipFile(path) as zp:
        for info in zp.infolist():
            if info.filename.endswith(".xyz"):
                with zp.open(info, mode='r') as fp:
                    atoms = read(TextIOWrapper(fp), format='xyz')
                    atoms.set_center_of_mass([0., 0., 0.])
                    yield info.filename, atoms


if __name__ == "__main__":
    # Make the argument parser
    parser = ArgumentParser()
    parser.add_argument('--num-nodes', default=4, help='Number of nodes for each NWChem computation', type=int)
    parser.add_argument('--num-parallel', default=1, help='Number of NWChem computations to run in parallel', type=int)
    parser.add_argument('--runs-per-job', default=1, help='Number of NWChem computations per job', type=int)
    parser.add_argument('--basis', default='aug-cc-pvdz', help='Basis set to use for all atoms')
    parser.add_argument('--max-size', default=None, type=int, help='Maximum size of cluster to run')
    parser.add_argument('--temp-dir', default=None, help='Where to store the temporary files')
    parser.add_argument('--structures', default='data/initial_MP2.zip', help='Path of initial structures to use')
    args = parser.parse_args()

    # Make a generator over structures to read
    if args.structures.endswith('zip'):
        strc_iter = generate_structures_from_zip(args.structures)
    elif args.structures.endswith('csv'):
        strcs = pd.read_csv(args.structures).sample(frac=1)  # Shuffle for better utilization
        filenames = strcs['coord_hash'].apply(lambda x: f'hydrodb_{x[:6]}.xyz')
        atoms = strcs['xyz'].apply(lambda x: read(StringIO(x), format='xyz'))
        for a in atoms:
            a.set_center_of_mass([0., 0., 0.])
        strc_iter = zip(filenames, atoms)
    else:
        raise ValueError(f'File type for {args.structures} is not supported')

    # Make the NWChem calculator
    ranks_per_node = cores_per_node
    calc = NWChem(
        memory=f'{memory_per_node / ranks_per_node:.1f} gb',
        basis={'*': args.basis},
        basispar='spherical',
        set={
            'lindep:n_dep': 0,
            'cphf:maxsub': 95,
            'mp2:aotol2e fock': '1d-14',
            'mp2:aotol2e': '1d-14',
            'mp2:backtol': '1d-14',
            'cphf:maxiter': 999,
            'cphf:thresh': '6.49d-5',
            'int:acc_std': '1d-16'
        },
        scf={
            'maxiter': 99,
            'tol2e': '1d-15',
        },
        mp2={'freeze': 'atomic'},
        theory='mp2',
        pretasks=[{
            'theory': 'dft',
            'dft': {
                'xc': 'hfexch',
                'maxiter': 50,
            },
            'set': {
                'quickguess': 't',
                'fock:densityscreen': 'f',
                'lindep:n_dep': 0,
            }
        }],
        # Note: Parsl sets --ntasks-per-node=1 in #SBATCH. For some reason, --ntasks in srun overrides it
        command=(f'srun -N {args.num_nodes} '
                 f'--ntasks={ranks_per_node * args.num_nodes} '
                 f'--export=ALL,OMP_NUM_THREADS={1} '
                 f'--ntasks-per-node={ranks_per_node} '
                 # Note: Parsl sets --ntasks-per-node=1 in #SBATCH. For some reason, --ntasks in srun overrides it
                 '--cpu-bind=cores shifter nwchem PREFIX.nwi > PREFIX.nwo'),
    )

    # Make the Parsl configuration
    config = parsl.Config(
        app_cache=False,  # No caching needed
        retries=16,  # Will restart a job if it fails for any reason
        strategy='htex_auto_scale',  # Will kill unused workers after 2 minutes
        executors=[HighThroughputExecutor(
            label='launch_from_mpi_nodes',
            max_workers=args.runs_per_job,
            cores_per_worker=1e-6,
            start_method='thread',
            provider=SlurmProvider(
                partition=None,  # 'debug'
                account='m3196',
                launcher=SimpleLauncher(),
                walltime='24:00:00',
                nodes_per_block=args.num_nodes * args.runs_per_job,
                init_blocks=args.num_parallel // args.runs_per_job,
                min_blocks=0,
                max_blocks=args.num_parallel // args.runs_per_job,  # Maximum number of jobs
                scheduler_options='''#SBATCH --image=ghcr.io/nwchemgit/nwchem-720.nersc.mpich4.mpi-pr:latest
#SBATCH -C cpu
#SBATCH --qos=preempt''',
                worker_init='''
module load python
conda activate /global/cfs/cdirs/m1513/lward/fast-finedtuned-forcefields/env-cpu/

export COMEX_MAX_NB_OUTSTANDING=6
export FI_CXI_RX_MATCH_MODE=hybrid
export COMEX_EAGER_THRESHOLD=16384
export FI_CXI_RDZV_THRESHOLD=16384
export FI_CXI_OFLOW_BUF_COUNT=6
export MPICH_SMP_SINGLE_COPY_MODE=CMA

which python
hostname
pwd''',
                cmd_timeout=1200,
            ),
        )]
    )
    parsl.load(config)

    print(f'Submitting structures...')
    n_skipped = 0
    with connect('initial.db', type='db') as db:
        # Submit structures to Parsl
        futures = []
        for filename, atoms in strc_iter:
            # Skip if structure is too larger
            if len(atoms) // 3 > args.max_size:
                continue

            # Store some tracking information
            atoms.info['filename'] = filename
            atoms.info['basis'] = args.basis
            atoms.info['num_nodes'] = args.num_nodes
            atoms.info['num_parallel'] = args.num_parallel

            # Skip if this structure is already in the database
            if db.count(filename=filename, basis=args.basis) > 0:
                n_skipped += 1
                continue

            # Submit the calculation to run
            future = run_nwchem(atoms, calc, temp_path=scratch_path)
            futures.append(future)
    print(f'Submitted {len(futures)}. {n_skipped} were already complete')

    # Loop over the futures and store them if the complete
    n_failures = 0
    for future in tqdm(as_completed(futures), total=len(futures), desc='completed'):
        # Get the result
        future: AppFuture = future
        exc = future.exception()
        if exc is not None:
            filename = future.task_def["args"][0].info["filename"]
            print(f'Failure for {future.task_def["args"][0].info["filename"]}. {str(exc)}')
            with open('failures.json', 'a') as fp:
                print(json.dumps({'name': filename, 'error': str(exc)}), file=fp)
            n_failures += 1
            continue
        atoms, runtime = future.result()

        # Store it (open a new connect each time to ensure results are written)
        with connect('initial.db', type='db') as db:
            db.write(atoms, **atoms.info, runtime=runtime)
    if n_failures > 0:
        print(f'Total failure count {n_failures}')
