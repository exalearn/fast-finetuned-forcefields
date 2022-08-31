from functools import partial, update_wrapper
from threading import Event
from datetime import datetime
from random import choice
from pathlib import Path
from typing import Dict
import hashlib
import logging
import argparse
import json
import sys

import torch
from colmena.models import Result
from colmena.redis.queue import ClientQueues, make_queue_pairs
from colmena.thinker import BaseThinker, event_responder, result_processor, ResourceCounter, task_submitter
from colmena.task_server.funcx import FuncXTaskServer
from funcx import FuncXClient
from torch import nn
import proxystore as ps
from ase.db import connect
import numpy as np

from fff.learning.spk import TorchMessage, train_schnet, SPKCalculatorMessage
from fff.simulation.md import run_dynamics

logger = logging.getLogger('main')


class Thinker(BaseThinker):
    """Class that schedules work on the HPC"""

    def __init__(
            self,
            queues: ClientQueues,
            out_dir: Path,
            db_path: Path,
            search_path: Path,
            model: nn.Module,
            n_models: int,
            n_samplers: int,
            ps_names: Dict[str, str],
    ):
        """
        Args:
            queues: Queues to send and receive work
            out_dir: Directory in which to write output files
            db_path: Path to the training data which has been collected so far
            search_path: Path to a databases of geometries to use for searching
            model: Initial model being trained. All further models will be trained using these starting weights
            n_models: Number of models to train in the ensemble
            ps_names: Mapping of task type to ProxyStore object associated with it
        """
        # Make the resource tracker
        #  For now, we only control resources over how many samplers are run at a time
        rec = ResourceCounter(n_samplers, ['sample'])
        super().__init__(queues, resource_counter=rec)

        # Save key configuration
        self.db_path = db_path
        self.starting_model = model
        self.ps_names = ps_names
        self.n_models = n_models
        self.out_dir = out_dir
        self.n_samplers = n_samplers

        # Load in the search space
        with connect(search_path) as db:
            self.search_space = [x.toatoms() for x in db.select('')]
        self.logger.info(f'Loaded a search space of {len(self.search_space)} geometries at {search_path}')

        # State that evolves as we run
        self.training_round = 0

        # Create a proxy for the starting model. It never changes
        #  We store it as a TorchMessage object which can be deserialized more easily
        train_store = ps.store.get_store(self.ps_names['train'])
        self.starting_model_proxy = train_store.proxy(TorchMessage(self.starting_model), key='starting-model-0')

        # Create a proxy for the "active" model that we'll use to generate trajectories
        sample_store = ps.store.get_store(self.ps_names['sample'])
        self.active_mode_proxy = sample_store.proxy(SPKCalculatorMessage(self.starting_model), key='active-model-0')

        # Coordination between threads
        self.start_training = Event()

        # Initialize the system settings
        self.start_training.set()  # Start by training the model
        self.rec.reallocate(None, 'sample', 'all')  # Immediately begin sampling new structures too

    @event_responder(event_name='start_training')
    def train_models(self):
        """Submit the models to be retrained"""
        self.training_round += 1

        # Load in the training dataset
        with connect(self.db_path) as db:
            logger.info(f'Connected to a database with {len(db)} entries at {self.db_path}')
            all_examples = [x.toatoms() for x in db.select("")]
        self.logger.info(f'Loaded {len(all_examples)} training examples')

        # Send off the models to be trained
        for i in range(self.n_models):
            # Sample the dataset with replacement
            subset = np.random.choice(all_examples, size=len(all_examples), replace=True)

            # Send a training job
            self.queues.send_inputs(
                self.starting_model_proxy, subset,
                method='train_schnet',
                topic='train',
                task_info={'model_id': i, 'training_round': self.training_round},
            )

    @result_processor(topic='train')
    def store_models(self, result: Result):
        """Store a model once it finishes updating"""
        logger.info(f'Received result from model {result.task_info["model_id"]}. Success: {result.success}')

        # Save the result to disk
        with open(self.out_dir / 'training-results.json', 'a') as fp:
            print(result.json(exclude={'inputs', 'value'}), file=fp)

        # Save the model to disk
        if result.success:
            model_dir = self.out_dir / 'models'
            model_dir.mkdir(exist_ok=True)
            with open(model_dir / f'model-{result.task_info["model_id"]}-round-{self.training_round}', 'wb') as fp:
                torch.save(result.value, fp)

        self.done.set()

    @task_submitter(task_type='sample')
    def submit_sampler(self):
        """Perform molecular dynamics to generate new structures"""

        # Get a random structure from the database
        starting_point = choice(self.search_space)

        # Submit it with the latest model
        self.queues.send_inputs(
            starting_point, self.active_mode_proxy,
            method='run_dynamics',
            topic='sample'
        )

    @result_processor(topic='sample')
    def store_sampling_results(self, result: Result):
        self.logger.info(f'Received sampling result. Success: {result.success}')

        # Save the result to disk
        with open(self.out_dir / 'sampling-results.json', 'a') as fp:
            print(result.json(exclude={'inputs', 'value'}), file=fp)

        # Count the new structures
        if result.success:
            traj = result.value
            self.logger.info(f'Produced {len(traj)} new structures')


if __name__ == '__main__':
    # User inputs
    parser = argparse.ArgumentParser()

    # Network configuration details
    group = parser.add_argument_group(title='Network Configuration',
                                      description='How to connect to the Redis task queues and task servers, etc')
    group.add_argument("--redishost", default="127.0.0.1", help="Address at which the redis server can be reached")
    group.add_argument("--redisport", default="6379", help="Port on which redis is available")

    # Computational infrastructure information
    group = parser.add_argument_group(title='Compute Infrastructure',
                                      description='Information about how to run the tasks')
    group.add_argument("--ml-endpoint", help='FuncX endpoint ID for model training and interface')
    group.add_argument("--qc-endpoint", help='FuncX endpoint ID for quantum chemistry')
    group.add_argument("--num-qc-workers", required=True, type=int, help="Total number of quantum chemistry workers.")

    # Problem configuration
    group = parser.add_argument_group(title='Problem Definition',
                                      description='Defining the search space, models and optimizers-related settings')
    group.add_argument('--starting-model', help='Path to the MPNN h5 files', required=True)
    group.add_argument('--training-set', help='Path to ASE DB used to train the initial models', required=True)
    group.add_argument('--search-space', help='Path to ASE DB of starting structures for molecular dynamics sampling',
                       required=True)

    # Parameters related to training the models
    group = parser.add_argument_group(title="Training Settings")
    group.add_argument("--num-epochs", type=int, default=8, help="Maximum number of training epochs")
    group.add_argument("--ensemble-size", type=int, default=2, help="Number of models to train to create ensemble")

    # Parameters related to sampling for new structures
    group = parser.add_argument_group(title="Sampling Settings")
    group.add_argument("--num-samplers", type=int, default=1, help="Number of agents to use to sample structures")

    # Parameters related to ProxyStore
    known_ps = [None, 'redis', 'file', 'globus']
    group = parser.add_argument_group(title='ProxyStore', description='Settings related to ProxyStore')
    group.add_argument('--simulate-ps-backend', default="file", choices=known_ps,
                       help='ProxyStore backend to use with "simulate" topic')
    group.add_argument('--sample-ps-backend', default="file", choices=known_ps,
                       help='ProxyStore backend to use with "sample" topic')
    group.add_argument('--train-ps-backend', default="file", choices=known_ps,
                       help='ProxyStore backend to use with "train" topic')
    group.add_argument('--ps-threshold', default=1000, type=int,
                       help='Min size in bytes for transferring objects via ProxyStore')
    group.add_argument('--ps-globus-config', default=None,
                       help='Globus Endpoint config file to use with the ProxyStore Globus backend')

    # Parse the arguments
    args = parser.parse_args()
    run_params = args.__dict__

    # Load in the model
    starting_model = torch.load(args.starting_model, map_location='cpu')

    # Check that the dataset exists
    with connect(args.training_set) as db:
        assert len(db) > 0
        pass

    # Prepare the output directory and logger
    start_time = datetime.utcnow()
    params_hash = hashlib.sha256(json.dumps(run_params).encode()).hexdigest()[:6]
    out_dir = Path('runs') / f'{start_time.strftime("%y%b%d-%H%M%S")}-{params_hash}'
    out_dir.mkdir(parents=True)

    # Set up the logging
    handlers = [logging.FileHandler(out_dir / 'runtime.log'), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO, handlers=handlers)
    logging.info(f'Run directory: {out_dir}')

    # Make the PS scratch directory
    ps_file_dir = out_dir / 'proxy-store'
    ps_file_dir.mkdir()

    # Init only the required ProxyStore backends
    ps_backends = {args.simulate_ps_backend, args.sample_ps_backend, args.train_ps_backend}
    logger.info(f'Initializing ProxyStore backends: {ps_backends}')
    if 'redis' in ps_backends:
        ps.store.init_store(ps.store.STORES.REDIS, name='redis', hostname=args.redishost, port=args.redisport,
                            stats=True)
    if 'file' in ps_backends:
        ps.store.init_store(ps.store.STORES.FILE, name='file', store_dir=str(ps_file_dir.absolute()), stats=True)
    if 'globus' in ps_backends:
        if args.ps_globus_config is None:
            raise ValueError('Must specify --ps-globus-config to use the Globus ProxyStore backend')
        endpoints = ps.store.globus.GlobusEndpoints.from_json(args.ps_globus_config)
        ps.store.init_store(ps.store.STORES.GLOBUS, name='globus', endpoints=endpoints, stats=True, timeout=600)
    ps_names = {'simulate': args.simulate_ps_backend, 'sample': args.sample_ps_backend,
                'train': args.train_ps_backend}

    # Connect to the redis server
    client_queues, server_queues = make_queue_pairs(args.redishost,
                                                    name=start_time.strftime("%d%b%y-%H%M%S"),  # Avoid clashes
                                                    port=args.redisport,
                                                    topics=['simulate', 'sample', 'train'],
                                                    serialization_method='pickle',
                                                    keep_inputs=False,
                                                    proxystore_name=ps_names,
                                                    proxystore_threshold=args.ps_threshold)

    # Apply wrappers to functions that will be used

    def _wrap(func, **kwargs):
        out = partial(func, **kwargs)
        update_wrapper(out, func)
        return out

    my_train_schnet = _wrap(train_schnet, num_epochs=args.num_epochs, device='cuda')
    my_run_dynamics = _wrap(run_dynamics, timestep=0.1, steps=10000, log_interval=100)

    # Create the task server
    fx_client = FuncXClient()  # Authenticate with FuncX
    task_map = dict((f, args.ml_endpoint) for f in [my_train_schnet])
    task_map.update(dict((f, args.qc_endpoint) for f in [my_run_dynamics]))
    doer = FuncXTaskServer(task_map, fx_client, server_queues)

    # Create the thinker
    thinker = Thinker(
        client_queues,
        out_dir=out_dir,
        db_path=Path(args.training_set),
        search_path=Path(args.search_space),
        model=starting_model,
        n_models=args.ensemble_size,
        n_samplers=args.num_samplers,
        ps_names=ps_names
    )
    logging.info('Created the method server and task generator')

    try:
        # Launch the servers
        doer.start()
        thinker.start()
        logging.info('Launched the servers')

        # Wait for the task generator to complete
        thinker.join()
        logging.info('Task generator has completed')
    finally:
        client_queues.send_kill_signal()

    # Wait for the method server to complete
    doer.join()

    # Cleanup ProxyStore backends (i.e., delete objects on the filesystem
    # for file/globus backends)
    for ps_backend in ps_backends:
        if ps_backend is not None:
            ps.store.get_store(ps_backend).cleanup()
