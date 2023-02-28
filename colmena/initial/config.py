from parsl.executors import HighThroughputExecutor
from parsl.providers import CobaltProvider, AdHocProvider
from parsl.addresses import address_by_hostname
from parsl.launchers import AprunLauncher
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

