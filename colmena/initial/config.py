from parsl.executors import HighThroughputExecutor
from parsl.providers import CobaltProvider, AdHocProvider, SlurmProvider
from parsl.addresses import address_by_hostname
from parsl.launchers import AprunLauncher, SrunLauncher, SimpleLauncher
from parsl.channels import SSHChannel
from parsl import Config


def theta_debug_and_lambda(log_dir: str) -> Config:
    """Configuration where simulation tasks run on Theta and ML tasks run on Lambda.

    Args:
        log_dir: Path to store monitoring DB and parsl logs
    Returns:
        (Config) Parsl configuration
    """
    # Set a Theta config for using the KNL nodes with a single worker per node
    config = Config(
        run_dir=log_dir,
        retries=1,
        executors=[
            HighThroughputExecutor(
                label='knl',
                max_workers=1,
                address=address_by_hostname(),
                provider=CobaltProvider(
                    queue='debug-cache-quad',
                    account='CSC249ADCD08',
                    launcher=AprunLauncher(overrides="-d 64 --cc depth -j 1"),
                    worker_init='''
module load miniconda-3
source activate /lus/theta-fs0/projects/CSC249ADCD08/fast-finedtuned-forcefields/env-cpu
which python
''',
                    nodes_per_block=8,
                    init_blocks=0,
                    min_blocks=0,
                    max_blocks=1,
                    cmd_timeout=300,
                    walltime='00:60:00',
                    scheduler_options='#COBALT --attrs enable_ssh=1:filesystems=home,theta-fs0'
                )),
            HighThroughputExecutor(
                address='localhost',
                label="v100",
                available_accelerators=8,
                worker_ports=(54928, 54875),  # Hard coded to match up with SSH tunnels
                worker_logdir_root='/lambda_stor/homes/lward/fast-finedtuned-forcefields/parsl-run/logs',
                provider=AdHocProvider(
                    channels=[SSHChannel('lambda1.cels.anl.gov',
                                         script_dir='/lambda_stor/homes/lward/fast-finedtuned-forcefields/parsl-run')],
                    worker_init='''
# Activate conda environment
source /homes/lward/miniconda3/bin/activate /lambda_stor/homes/lward/fast-finedtuned-forcefields/env
which python
''',
                ),
            )]
    )

    return config


def local_parsl(log_dir: str) -> Config:
    """Configuration which runs all tasks on a single worker"""

    return Config(
        run_dir=log_dir,
        executors=[HighThroughputExecutor(
            available_accelerators=1,
        )]
    )


def theta_debug_and_venti(log_dir: str) -> Config:
    """Configuration where simulation tasks run on Theta and ML tasks run on Lambda.

    Args:
        log_dir: Path to store monitoring DB and parsl logs
    Returns:
        (Config) Parsl configuration
    """
    # Set a Theta config for using the KNL nodes with a single worker per node
    config = Config(
        run_dir=log_dir,
        retries=1,
        executors=[
            HighThroughputExecutor(
                label='cpu',
                max_workers=1,
                address=address_by_hostname(),
                provider=CobaltProvider(
                    queue='debug-cache-quad',  # Flat has lower utilization, even though xTB is (slightly) faster on cache
                    account='CSC249ADCD08',
                    launcher=AprunLauncher(overrides="-d 256 --cc depth -j 4"),
                    worker_init='''
module load miniconda-3
module swap PrgEnv-intel PrgEnv-gnu
source activate /lus/theta-fs0/projects/CSC249ADCD08/fast-finedtuned-forcefields/env-cpu
which python
''',  # Active the environment
                    nodes_per_block=8,
                    init_blocks=0,
                    min_blocks=0,
                    max_blocks=1,
                    cmd_timeout=300,
                    walltime='1:00:00',
                    scheduler_options='#COBALT --attrs enable_ssh=1:filesystems=theta-fs0,home',
                )),
            HighThroughputExecutor(
                address='localhost',
                label="gpu",
                available_accelerators=20,
                worker_ports=(54928, 54875),  # Hard coded to match up with SSH tunnels
                worker_logdir_root='/home/lward/multi-site-campaigns/parsl-logs',
                provider=AdHocProvider(
                    channels=[SSHChannel('lambda5.cels.anl.gov', script_dir='/home/lward/multi-site-campaigns/parsl-logs')],
                    worker_init='''
# Activate conda environment
source /homes/lward/miniconda3/bin/activate /home/lward/fast-finetuned-forcefields/env
which python
''',
                ),
            )]
    )

    return config


def theta_and_venti(log_dir: str) -> Config:
    """Configuration where simulation tasks run on Theta and ML tasks run on Lambda.

    Args:
        log_dir: Path to store monitoring DB and parsl logs
    Returns:
        (Config) Parsl configuration
    """
    # Set a Theta config for using the KNL nodes with a single worker per node
    config = Config(
        run_dir=log_dir,
        retries=1,
        executors=[
            HighThroughputExecutor(
                label='cpu',
                max_workers=1,
                address=address_by_hostname(),
                provider=CobaltProvider(
                    queue='default',  # Flat has lower utilization, even though xTB is (slightly) faster on cache
                    account='CSC249ADCD08',
                    launcher=AprunLauncher(overrides="-d 256 --cc depth -j 4"),
                    worker_init='''
module load miniconda-3
module swap PrgEnv-intel PrgEnv-gnu
source activate /lus/theta-fs0/projects/CSC249ADCD08/fast-finedtuned-forcefields/env-cpu
which python
''',  # Active the environment
                    nodes_per_block=128,
                    init_blocks=0,
                    min_blocks=0,
                    max_blocks=1,
                    cmd_timeout=300,
                    walltime='00:60:00',
                    scheduler_options='#COBALT --attrs enable_ssh=1:filesystems=theta-fs0,home',
                )),
            HighThroughputExecutor(
                address='localhost',
                label="gpu",
                available_accelerators=20,
                worker_ports=(54928, 54875),  # Hard coded to match up with SSH tunnels
                worker_logdir_root='/home/lward/fast-finetuned-forcefields/parsl-logs',
                provider=AdHocProvider(
                    channels=[SSHChannel('lambda5.cels.anl.gov', script_dir='/home/lward/fast-finetuned-forcefields/parsl-logs')],
                    worker_init='''
# Activate conda environment
source /homes/lward/miniconda3/bin/activate /home/lward/fast-finetuned-forcefields/env
which python
''',
                ),
            )]
    )

    return config


def perlmutter_nwchem(log_dir: str, qc_nodes: int = 128) -> Config:
    """Configuration which uses Perlmutter GPU for ML tasks and Perlmutter CPU to run MPI tasks

    Args:
        log_dir: Path in which to write logs
        qc_nodes: Number of nodes to use for NWChem
    """

    return Config(
        run_dir=log_dir,
        retries=1,
        executors=[
            HighThroughputExecutor(
                label='cpu',
                max_workers=qc_nodes,  # Maximum possible number
                cores_per_worker=1e-6,
                start_method='thread',
                provider=SlurmProvider(
                    partition=None,  # 'debug'
                    account='m1513',
                    launcher=SimpleLauncher(),
                    walltime='24:00:00',
                    nodes_per_block=qc_nodes,
                    init_blocks=0,
                    min_blocks=0,
                    max_blocks=1,
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
            ),
            HighThroughputExecutor(
                label="gpu",
                available_accelerators=4,  # Four GPUs per note
                cpu_affinity='block',
                provider=SlurmProvider(
                    partition=None,  # 'debug'
                    account='m1513',
                    launcher=SrunLauncher(overrides="--gpus-per-node 4 -c 64"),
                    walltime='1:00:00',
                    nodes_per_block=2,  # So that we have a total of 8 GPUs
                    init_blocks=0,
                    min_blocks=0,
                    max_blocks=1,  # Maximum number of jobs
                    cmd_timeout=1200,
                    scheduler_options='''#SBATCH -C gpu
#SBATCH --qos=regular''',
                    worker_init='''
module load python
module list
source activate /global/cfs/cdirs/m1513/lward/fast-finedtuned-forcefields/env-gpu/

nvidia-smi
which python
hostname
pwd''',
                ))]
    )
