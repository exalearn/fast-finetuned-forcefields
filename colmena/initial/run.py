from functools import partial, update_wrapper
from threading import Event, Lock
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from random import choice
from pathlib import Path
from typing import Dict
import hashlib
import logging
import argparse
import shutil
import json
import sys

import ase
from ase.db import connect
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from colmena.models import Result
from colmena.redis.queue import ClientQueues, make_queue_pairs
from colmena.thinker import BaseThinker, event_responder, result_processor, ResourceCounter, task_submitter
from colmena.task_server.funcx import FuncXTaskServer
from funcx import FuncXClient
from torch import nn
import proxystore as ps
import numpy as np
import torch
from ttm.ase import TTMCalculator

from fff.learning.spk import TorchMessage, train_schnet, SPKCalculatorMessage
from fff.simulation import run_calculator
from fff.simulation.md import run_dynamics

logger = logging.getLogger('main')


@dataclass
class Trajectory:
    """Tracks the state of searching along individual trajectories

    We mark the starting point, the last point produced from sampling,
    and the last point we produced that has been validated
    """
    starting: ase.Atoms  # Starting point of the trajectory
    last_validated: ase.Atoms = None  # Last validated point on the trajectory
    current: ase.Atoms = None  # Last point produced along the trajectory
    validated: bool = True  # Whether `current` has been validated
    running: bool = False  # Whether the trajectory is running

    def __post_init__(self):
        self.last_validated = self.current = self.starting

    @property
    def eligible(self):
        """Whether this trajectory is eligible for selecting as a simpling calculation"""
        return self.validated and not self.running

    def set_validation(self, success: bool):
        """Set whether the trajectory was successfully validated

        Args:
            success: Whether the validation was successful
        """
        if success:
            self.last_validated = self.current  # Move the last validated forward
        self.validated = True


@dataclass
class SimulationTask:
    atoms: ase.Atoms  # Structure to be run
    traj_id: int  # Which trajectory this came from
    ml_eng: float  # Energy predicted from machine learning model
    ml_std: float | None = None  # Uncertainty of the model


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
            n_simulators: int,
            energy_tolerance: float,
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
            n_simulators: Number of workers to set aside for simulation
            n_samplers: Number of workers to set aside for sampling structures
            energy_tolerance: How large of an energy difference to accept when auditing
            ps_names: Mapping of task type to ProxyStore object associated with it
        """
        # Make the resource tracker
        #  For now, we only control resources over how many samplers are run at a time
        rec = ResourceCounter(n_samplers + n_simulators, ['sample', 'simulate'])
        super().__init__(queues, resource_counter=rec)

        # Save key configuration
        self.db_path = db_path
        self.starting_model = model
        self.ps_names = ps_names
        self.n_models = n_models
        self.out_dir = out_dir
        self.energy_tolerance = energy_tolerance
        self.to_run = 10

        # Load in the search space
        with connect(search_path) as db:
            self.search_space = [Trajectory(x.toatoms()) for x in db.select('')]
        self.logger.info(f'Loaded a search space of {len(self.search_space)} geometries at {search_path}')

        # State that evolves as we run
        self.training_round = 0
        self.num_complete = 0

        # Create a proxy for the starting model. It never changes
        #  We store it as a TorchMessage object which can be deserialized more easily
        train_store = ps.store.get_store(self.ps_names['train'])
        self.starting_model_proxy = train_store.proxy(TorchMessage(self.starting_model))

        # Create a proxy for the "active" model that we'll use to generate trajectories
        sample_store = ps.store.get_store(self.ps_names['sample'])
        self.active_model_proxy = sample_store.proxy(SPKCalculatorMessage(self.starting_model))

        # Coordination between threads
        self.start_training = Event()  # Starts training on the latest data
        self.active_updated = False  # Whether the active model has already been updated for this batch
        self.models_ready = Event()  # Start sampling only after the first models are read
        self.has_tasks = Event()  # Whether there are tasks ready to submit
        self.task_queue_audit: list[SimulationTask] = []  # Tasks for checking trajectories
        self.task_queue_active: deque[SimulationTask] = deque(maxlen=self.to_run)  # Tasks produce optimal training data
        self.task_queue_lock = Lock()  # Locks both of the task queues

        # Initialize the system settings
        self.start_training.set()  # Start by training the model
        self.rec.reallocate(None, 'sample', n_samplers)  # Immediately begin sampling new structure
        self.rec.reallocate(None, 'simulate', n_simulators)  # Will block until first sample is back

    @event_responder(event_name='start_training')
    def train_models(self):
        """Submit the models to be retrained"""
        self.training_round += 1
        self.active_updated = False

        # Load in the training dataset
        with connect(self.db_path) as db:
            logger.info(f'Connected to a database with {len(db)} entries at {self.db_path}')
            all_examples = np.array([x.toatoms() for x in db.select("")], dtype=object)
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
        self.logger.info(f'Received result from model {result.task_info["model_id"]}. Success: {result.success}')

        # Save the result to disk
        with open(self.out_dir / 'training-results.json', 'a') as fp:
            print(result.json(exclude={'inputs', 'value'}), file=fp)

        # Save the model to disk
        if result.success:
            # Unpack the result and training history
            model_msg, train_log = result.value

            # Store the result to disk
            model_dir = self.out_dir / 'models'
            model_dir.mkdir(exist_ok=True)
            model_path = model_dir / f'model-{result.task_info["model_id"]}-round-{self.training_round}'
            with open(model_path, 'wb') as fp:
                torch.save(model_msg, fp)
            self.logger.info(f'Saved model to: {model_path}')

            # Save the training data
            with open(self.out_dir / 'training-history.json', 'a') as fp:
                print(json.dumps(train_log.todict(orient='list')), file=fp)

            # Update the "active" model
            if not self.active_updated:
                sample_store = ps.store.get_store(self.ps_names['sample'])
                sample_store.evict(ps.proxy.get_key(self.active_model_proxy))
                self.active_model_proxy = sample_store.proxy(SPKCalculatorMessage(model_msg.get_model()))
                self.active_updated = True
                self.logger.info('Updated the active model')

        self.models_ready.set()  # Signals that inference can start

    @task_submitter(task_type='sample')
    def submit_sampler(self):
        """Perform molecular dynamics to generate new structures"""
        self.models_ready.wait()

        # Pick randomly from the eligible trajectories
        traj_id = choice([i for i, x in enumerate(self.search_space) if x.eligible])

        # Start from the last validated structure in the series
        starting_point = self.search_space[traj_id].last_validated
        self.search_space[traj_id].running = True  # Mark that we're already running this structure

        # Give it some velocity if there is not
        if starting_point.get_velocities().max() < 1e-6:
            self.logger.info(f'Starting from the first step. Initializing temperature')
            MaxwellBoltzmannDistribution(starting_point, temperature_K=100, rng=np.random.RandomState(traj_id))

        # Submit it with the latest model
        self.queues.send_inputs(
            starting_point, self.active_model_proxy,
            method='run_dynamics',
            topic='sample',
            task_info={'traj_id': traj_id}
        )

    @result_processor(topic='sample')
    def store_sampling_results(self, result: Result):
        """Store the results of a sampling run"""
        self.rec.release('sample', 1)

        traj_id = result.task_info['traj_id']
        self.logger.info(f'Received sampling result for trajectory {traj_id}. Success: {result.success}')

        # Mark that we are done running this trajectory, but it is not validated
        self.search_space[traj_id].validated = False
        self.search_space[traj_id].running = False

        # Save the result to disk
        with open(self.out_dir / 'sampling-results.json', 'a') as fp:
            print(result.json(exclude={'inputs', 'value'}), file=fp)

        # If successful, submit the structures for auditing
        if result.success:
            traj = result.value
            self.logger.info(f'Produced {len(traj)} new structures')

            # Add the latest one to the audit list
            with self.task_queue_lock:
                self.task_queue_audit.append(SimulationTask(
                    atoms=traj[-1], traj_id=traj_id, ml_eng=traj[-1].get_potential_energy()
                ))
                self.has_tasks.set()

    @task_submitter(task_type="simulate")
    def submit_simulation(self):
        """Submit a new simulation to check results from sampling/gather new training data"""

        # Get a simulation to run
        to_run = None
        task_type = None
        while to_run is None:
            self.has_tasks.wait()
            with self.task_queue_lock:  # Wait for another thread to add structures
                if len(self.task_queue_audit) > 0:
                    to_run = self.task_queue_audit.pop(0)  # Get the one off the front of the list
                    task_type = 'audit'
                elif len(self.task_queue_active) > 0:
                    to_run = self.task_queue_active.popleft()  # Get the most recent one
                    task_type = 'active'
                else:
                    self.logger.info('No tasks are available to run. Waiting...')
                    self.has_tasks.clear()  # We don't have any tasks to run

        # Submit it
        self.logger.info(f'Selected a {task_type} to run next')
        self.queues.send_inputs(to_run.atoms, method='run_calculator', topic='simulate',
                                task_info={'traj_id': to_run.traj_id, 'task_type': task_type,
                                           'ml_energy': to_run.ml_eng})

    @result_processor(topic='simulate')
    def collect_calculation(self, result: Result):
        """Store the results from a simulation"""
        self.rec.release('simulate', 1)

        # Get the associated trajectory
        traj_id = result.task_info['traj_id']
        traj = self.search_space[traj_id]
        self.logger.info(f'Received a simulation from trajectory {traj_id}. Success: {result.success}')

        # Write output to disk regardless of whether we were successful
        with open(self.out_dir / 'simulation-result.json', 'a') as fp:
            print(result.json(exclude={'value', 'inputs'}), file=fp)

        # Store the result in the database if successful
        task_type = result.task_info['task_type']
        if result.success:
            # Store in the training set
            atoms: ase.Atoms = result.value
            with connect(self.db_path) as db:
                db.write(atoms, runtime=result.time_running)
            self.num_complete += 1
            self.logger.info(f'Finished {self.num_complete}/10 structures')
            if self.num_complete >= 10:
                self.done.set()

            # If an audit calculation, check whether the energy is close to the ML prediction
            if task_type == 'audit':
                ml_eng = result.task_info['ml_energy']
                difference = abs(ml_eng - atoms.get_potential_energy()) / len(atoms)
                was_successful = difference < self.energy_tolerance
                traj.set_validation(was_successful)
                self.logger.info(f'Audit for {traj_id} complete. Result: {was_successful}.'
                                 f' Difference: {difference:.3f} eV/atom')
        elif task_type == 'audit':
            traj.set_validation(False)


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

    # Problem configuration
    group = parser.add_argument_group(title='Problem Definition',
                                      description='Defining the search space, models and optimizers-related settings')
    group.add_argument('--starting-model', help='Path to the MPNN h5 files', required=True)
    group.add_argument('--training-set', help='Path to ASE DB used to train the initial models', required=True)
    group.add_argument('--search-space', help='Path to ASE DB of starting structures for molecular dynamics sampling',
                       required=True)

    # Parameters related to training the models
    group = parser.add_argument_group(title="Training Settings")
    group.add_argument("--num-epochs", type=int, default=32, help="Maximum number of training epochs")
    group.add_argument("--ensemble-size", type=int, default=2, help="Number of models to train to create ensemble")

    # Parameters related to sampling for new structures
    group = parser.add_argument_group(title="Sampling Settings")
    group.add_argument("--num-samplers", type=int, default=1, help="Number of agents to use to sample structures")
    group.add_argument("--run-length", type=int, default=1000, help="How many timesteps to run sampling calculations."
                                                                     " Longest time between auditing states.")
    group.add_argument("--energy-tolerance", type=float, default=0.01,
                       help="Maximum allowable energy different to accept results of sampling run")

    # Parameters related to gathering more training data
    group = parser.add_argument_group(title="Simulation Settings")
    group.add_argument("--num-simulators", default=1, type=int, help="Number of simulation workers")

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

    # Make a copy of the training data
    train_path = out_dir / 'train.db'
    shutil.copyfile(args.training_set, train_path)

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
    my_run_dynamics = _wrap(run_dynamics, timestep=0.1, steps=args.run_length, log_interval=100)
    my_run_simulation = _wrap(run_calculator, calc=TTMCalculator())

    # Create the task server
    fx_client = FuncXClient()  # Authenticate with FuncX
    task_map = dict((f, args.ml_endpoint) for f in [my_train_schnet])
    task_map.update(dict((f, args.qc_endpoint) for f in [my_run_dynamics, my_run_simulation]))
    doer = FuncXTaskServer(task_map, fx_client, server_queues)

    # Create the thinker
    thinker = Thinker(
        client_queues,
        out_dir=out_dir,
        db_path=train_path,
        search_path=Path(args.search_space),
        model=starting_model,
        n_models=args.ensemble_size,
        n_samplers=args.num_samplers,
        n_simulators=args.num_simulators,
        energy_tolerance=args.energy_tolerance,
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
    logging.info('Task server has completed')

    # Cleanup ProxyStore backends (i.e., delete objects on the filesystem
    # for file/globus backends)
    for ps_backend in ps_backends:
        if ps_backend is not None:
            ps.store.get_store(ps_backend).cleanup()
    logging.info('ProxyStores cleaned. Exiting now')
