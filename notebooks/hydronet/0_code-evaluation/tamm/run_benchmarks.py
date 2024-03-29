"""Run NWChem calculations with different configurations"""
from argparse import ArgumentParser
from pathlib import Path
from csv import DictWriter
from io import StringIO
import platform
import json

from parsl.executors import HighThroughputExecutor
from parsl.launchers import SimpleLauncher
from parsl.providers import LocalProvider, PBSProProvider
from ase.io import read
from tqdm import tqdm
import pandas as pd
import parsl


@parsl.python_app
def run_tamm(atoms, tamm_command: str, tamm_template: dict, basis, scratch_dir: str | None = None, walltime=None):
    """Run an NWChem computation on the requested cluster

    Args:
        atoms: Cluster to evaluate
        tamm_command: Command to use to invoke TAMM
        tamm_template: Template input file
        basis: Basis set
        scratch_dir: Directory for scratch data
    Returns:
        - Runtime (s)
        - Energy (eV)
    """
    from fff.simulation.tamm import TAMMCalculator
    from tempfile import mkdtemp
    from shutil import rmtree
    from ase import units
    import time

    tmpdir = mkdtemp(dir=scratch_dir)
    cleanup = False
    try:
        calc = TAMMCalculator(
            directory=tmpdir,
            command=tamm_command,
            template=tamm_template,
            basisset=basis,
        )
        start_time = time.monotonic()
        forces = calc.get_potential_energy(atoms)

        run_time = time.monotonic() - start_time
        energy = calc.get_potential_energy(atoms) / units.Ha
        cleanup = True  # Cleanup if we succeed
    finally:
        if cleanup:
            rmtree(tmpdir)

    return run_time, energy, forces


if __name__ == "__main__":
    # Make the argument parser
    parser = ArgumentParser()
    parser.add_argument('--max-repeats', default=1, help='Number of times to repeat the computations', type=int)
    parser.add_argument('--num-nodes', default=1, help='Number of nodes on which to run', type=int)
    parser.add_argument('--ranks-per-node', default=1, help='Number of ranks per node', type=int)
    parser.add_argument('--basis', default='aug-cc-pvdz', help='Basis set to use for all atoms')
    parser.add_argument('--timeout', default=None, type=float, help='Timeout for DFT calculations')
    parser.add_argument('--method', default='HartreeFock', help='Method to run (based on the name of the executable)')
    parser.add_argument('--min-size', type=int, help='Minimum size to evaluate')
    args = parser.parse_args()

    # Recognize which node we're running on,
    hostname = platform.node().rstrip('-0123456789')
    if hostname.startswith('bettik-linux'):
        tamm_command = f'mpirun -n 12 /home/lward/Software/nwchemex/tamm_install/bin/{args.method} tamm.json > tamm.out'
        parsl_exec = HighThroughputExecutor(
            max_workers=1,
            provider=LocalProvider()
        )
    elif hostname.startswith('uan'):
        tamm_command = (f'mpiexec -n {args.num_nodes * args.ranks_per_node} --ppn {args.ranks_per_node} --depth=1 --cpu-bind depth --env OMP_NUM_THREADS=1 --env FI_CXI_RX_MATCH_MODE=software'
                        f' /lus/gila/projects/CSC249ADCD08_CNDA/tamm/install/tamm_cc/bin/{args.method} tamm.json > tamm.out')
        parsl_exec = HighThroughputExecutor(
            label='launch_from_mpi_nodes',
            max_workers=1,
            provider=PBSProProvider(
                queue='workq',
                account='CSC249ADCD08_CNDA',
                nodes_per_block=args.num_nodes,
                select_options='system=sunspot,place=scatter',
                launcher=SimpleLauncher(),
                cpus_per_node=108,
                walltime="2:00:00",
                worker_init='''
# Activate environment and drop into directory
source /soft/datascience/conda-2023-01-31/miniconda3/bin/activate /lus/gila/projects/CSC249ADCD08_CNDA/fast-finetuned-forcefields/env-cpu
cd ${PBS_O_WORKDIR}
which python
hostname
pwd

# TAMM-specific env variables
export CRAYPE_LINK_TYPE=dynamic
export FI_CXI_RX_MATCH_MODE=software
module load mpich
ONEAPI_MPICH_GPU=NO_GPU module load oneapi/eng-compiler/2022.12.30.003
'''
            )
        )

    else:
        raise ValueError(f'No executable specified for {hostname}')

    print(f'Running on {hostname}. Nodes={args.num_nodes}')

    # Read in the TAMM template
    with open('tamm-h2o.json') as fp:
        template = json.load(fp)

    # Load in the example structures
    examples = pd.read_csv('../example_structures.csv')
    print(f'Loaded in {len(examples)} example structures')

    # Load in what has been run already
    out_file = Path('runtimes.csv')
    fields = ['id', 'hostname', 'method', 'basis', 'n_waters', 'ttm_energy', 'qc_energy', 'runtime', 'num_nodes', 'ranks_per_node']
    if not out_file.exists():
        with out_file.open('w', newline='') as fp:
            writer = DictWriter(fp, fieldnames=fields)
            writer.writeheader()

    already_ran = pd.read_csv(out_file)[['id', 'hostname', 'method', 'num_nodes', 'ranks_per_node', 'basis']].apply(tuple, axis=1).tolist()
    print(f'Found {len(already_ran)} structures already in dataset')

    # Load in the Parsl configuration
    config = parsl.Config(
        app_cache=False,  # No caching needed
        retries=1,  # Will restart a job if it fails for any reason
        executors=[parsl_exec]
    )
    parsl.load(config)

    # Loop over them
    run_dir = Path('tamm_runs')
    run_dir.mkdir(exist_ok=True)
    for rid, row in tqdm(examples.iterrows()):
        # Make sure it has been run fewer than the desired number of times
        if already_ran.count((row['id'], hostname, args.method, args.num_nodes, args.ranks_per_node, args.basis)) >= args.max_repeats:
            continue

        # Skip if too small
        if args.min_size is not None and row['n_waters'] < args.min_size:
            continue

        # Parse it as an ASE atoms object
        atoms = read(StringIO(row['xyz']), format='xyz')
        atoms.center()

        # Run it
        future = run_tamm(atoms, tamm_command, template, args.basis, scratch_dir=str(run_dir.absolute()))
        run_time, energy, forces = future.result()

        # Save results to disk
        with out_file.open('a', newline='') as fp:
            writer = DictWriter(fp, fieldnames=fields)
            writer.writerow({
                'id': row['id'],
                'hostname': hostname,
                'method': args.method,
                'num_nodes': args.num_nodes,
                'ranks_per_node': args.ranks_per_node,
                'n_waters': row['n_waters'],
                'basis': args.basis,
                'ttm_energy': row['ttm_energy'],
                'qc_energy': energy,
                'runtime': run_time
            })

        # If runtime is too long, break
        if args.timeout is not None and run_time > args.timeout:
            print('We have exceeded the walltime limit.')
            break
