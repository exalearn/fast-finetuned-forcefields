"""Run NWChem calculations with different configurations"""
from argparse import ArgumentParser
from pathlib import Path
from csv import DictWriter
from io import StringIO
import platform
import json

from parsl.executors import HighThroughputExecutor
from parsl.launchers import SimpleLauncher
from parsl.providers import SlurmProvider
from ase.io import read
from tqdm import tqdm
import pandas as pd
import parsl


@parsl.python_app
def run_nwchem(atoms, basis, nodes, ranks_per_node, twostep: bool, walltime=None):
    """Run an NWChem computation on the requested cluster

    Args:
        atoms: Cluster to evaluate
    Returns:
        - Runtime (s)
        - Energy (eV)
        - forces (eV/ang)
        - twostep: Whether to perform a DFT calculation first
    """
    from ase.calculators.nwchem import NWChem
    from tempfile import TemporaryDirectory
    import time

    # Compute the number of ranks and such
    total_ranks = nodes * ranks_per_node
    cores_per_rank = 68 // ranks_per_node  # Set for Cori KNL
    
    # Create the options for the DFT first step
    new_opts = {}
    if twostep:
        new_opts = {
            'initwfc': {
                'dft': {
                    'xc': 'hfexch',
                    # 'convergence': {
                    #     'energy': '1d-12',
                    #     'gradient': '5d-19'
                    # },
                    # 'tolerances': {'acccoul': 15},
                    'maxiter': 50,
                },
                'set': {
                    'quickguess': 't',
                    'fock:densityscreen': 'f',
                    'lindep:n_dep': 0,
                }
            }
        }

    with TemporaryDirectory(dir='/global/cscratch1/sd/wardlt/nwchem-bench/') as tmpdir:
        nwchem = NWChem(
            scratch=str(tmpdir),
            perm=str(tmpdir),
            memory=f'{90 / ranks_per_node:.1f} gb',
            basis={'*': basis},
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
            scf={'maxiter': 99, 'tol2e': '1d-15'},
            mp2={'freeze': 'atomic'},
            # Note: Parsl sets --ntasks-per-node=1 in #SBATCH. For some reason, --ntasks in srun overrides it
            command=(f'srun -N {nodes} --ntasks={ranks_per_node*nodes} --export=ALL,OMP_NUM_THREADS={cores_per_rank} '
                      '--cpu-bind=cores shifter nwchem PREFIX.nwi > PREFIX.nwo'),
            **new_opts
        )
        start_time = time.monotonic()
        forces = nwchem.get_forces(atoms)

        run_time = time.monotonic() - start_time
        energy = nwchem.get_potential_energy(atoms) * 0.036749322176  # Convert to Ha

    return run_time, energy, forces


if __name__ == "__main__":
    # Make the argument parser
    parser = ArgumentParser()
    parser.add_argument('--ranks-per-node', default=64, help='Number of ranks per node', type=int)
    parser.add_argument('--max-repeats', default=1, help='Number of times to repeat the computations', type=int)
    parser.add_argument('--num-nodes', default=1, help='Number of nodes on which to run', type=int)
    parser.add_argument('--basis', default='aug-cc-pvdz', help='Basis set to use for all atoms')
    parser.add_argument('--dftguess', action='store_true', help='Run a DFT calculation to get an initial WFC guess')
    parser.add_argument('--timeout', default=None, type=float, help='Timeout for DFT calculations')
    args = parser.parse_args()

    # Recognize which node we're running on
    hostname = platform.node().rstrip('0123456789')
    print(f'Running on {hostname}. Nodes={args.num_nodes}. Ranks per Node: {args.ranks_per_node}')

    # Load in the example structures
    examples = pd.read_csv('example_structures.csv')
    print(f'Loaded in {len(examples)} example structures')

    # Load in what has been run already
    out_file = Path('runtimes.csv')
    fields = ['id', 'hostname', 'ranks_per_node', 'basis', 'n_waters', 'ttm_energy', 'mp2_energy', 'runtime', 'num_nodes', 'gradients', 'dftguess']
    if not out_file.exists():
        with out_file.open('w', newline='') as fp:
            writer = DictWriter(fp, fieldnames=fields)
            writer.writeheader()

    already_ran = pd.read_csv(out_file)[['id', 'hostname', 'num_nodes', 'ranks_per_node', 'basis', 'dftguess']].apply(tuple, axis=1).tolist()
    print(f'Found {len(already_ran)} structures already in dataset')

    # Load in the Parsl configuration
    config = parsl.Config(
        app_cache=False,  # No caching needed
        retries=1,  # Will restart a job if it fails for any reason
        executors=[HighThroughputExecutor(
            label='launch_from_mpi_nodes',
            max_workers=1,
            provider=SlurmProvider(
                partition='regular',
                account='m1513',
                launcher=SimpleLauncher(),
                walltime='36:00:00',
                nodes_per_block=args.num_nodes,  # Number of nodes per job
                init_blocks=0,
                min_blocks=1,
                max_blocks=1,  # Maximum number of jobs
                scheduler_options='#SBATCH --image=ghcr.io/nwchemgit/nwchem-702.mpipr.nersc:latest\n#SBATCH -C knl',
                worker_init=f'''
module load python
conda activate /global/project/projectdirs/m1513/lward/hydronet/env

module swap craype-{{${{CRAY_CPU_TARGET}},mic-knl}}
export OMP_NUM_THREADS={68 // args.ranks_per_node}
export MPICH_GNI_MAX_EAGER_MSG_SIZE=131026
export MPICH_GNI_NUM_BUFS=80
export MPICH_GNI_NDREG_MAXSIZE=16777216
export MPICH_GNI_MBOX_PLACEMENT=nic
export MPICH_GNI_RDMA_THRESHOLD=65536
export COMEX_MAX_NB_OUTSTANDING=6

which python
hostname
pwd
                ''',
                cmd_timeout=120,
            ),
        )]
    )
    parsl.load(config)

    # Loop over them
    for rid, row in tqdm(examples.iterrows()):
        # Make sure it has been run fewer than the desired number of times
        if already_ran.count((row['id'], hostname, args.num_nodes, args.ranks_per_node, args.basis, args.dftguess)) >= args.max_repeats:
            continue

        # Parse it as an ASE atoms object
        atoms = read(StringIO(row['xyz']), format='xyz')

        # Run it
        future = run_nwchem(atoms, args.basis, args.num_nodes, args.ranks_per_node, args.dftguess)
        run_time, energy, forces = future.result()

        # Save results to disk
        with out_file.open('a', newline='') as fp:
            writer = DictWriter(fp, fieldnames=fields)
            writer.writerow({
                'id': row['id'],
                'hostname': hostname,
                'num_nodes': args.num_nodes,
                'ranks_per_node': args.ranks_per_node,
                'n_waters': row['n_waters'],
                'basis': args.basis,
                'ttm_energy': row['ttm_energy'],
                'mp2_energy': energy,
                'runtime': run_time,
                'gradients': json.dumps(forces.tolist()),
                'dftguess': args.dftguess
            })

        # If runtime is too long, break
        if run_time > args.timeout:
            print('We have exceeded the walltime limit.')
            break
