"""Utilities for comparing model performance"""
from pathlib import Path
from scipy.stats import kendalltau
from fff.simulation.utils import read_from_string
import pandas as pd
import numpy as np
import json


def assess_against_holdout(runs: Path, tags: list[str]) -> pd.DataFrame:
    """Gather the model performance on a hold-out set of molecular energies and forces
    
    Args:
        runs: Path to the run files for each run to compare
        tags: List of data to extract from run parameters
    Returns:
        DataFrame containing performance summary
    """
    data = []
    for run in runs:
        # Get the energy tolerance of my run
        params = json.loads((run / 'runparams.json').read_text())
        
        # Load in the duplicates
        duplicates = json.loads((run / 'duplicates.json').read_text())

        # Process all
        eval_data = []
        for duplicate in duplicates:
            try:
                # Get the mean error in energy and force for all trajectories
                dup_data = pd.read_csv(run / 'final-model' / 'benchmark.csv')
                dup_data['run'] = duplicate
                dup_data['energy_error_per_atom'] = dup_data['energy_error_per_atom'].abs() * 1000
                dup_data['energy_error_per_atom-init'] = dup_data['energy_error_per_atom-init'].abs() * 1000
                
                eval_data.append(dup_data)
            except FileNotFoundError:
                print(f'Could not assess {duplicate} for {run}')
                continue
        n_duplicates = len(eval_data)
        eval_data = pd.concat(eval_data)

        # Store the mean and SEM for each
        all_errors = eval_data.groupby(['traj', 'run'])[['energy_error_per_atom', 'force_rmsd', 'energy_error_per_atom-init', 'force_rmsd-init']].mean()
        record = {'name': run.name}
        record.update(dict((c, params[c]) for c in tags))
        record['n_duplicates'] = n_duplicates
        for c in all_errors.columns:
            record[f'{c}-mean'] = all_errors[c].mean()
            record[f'{c}-sem'] = all_errors[c].sem()
        data.append(record)

    return pd.DataFrame(data)


def assess_from_mctbp_runs(runs: Path, tags: list[str]) -> pd.DataFrame:
    """Gather model performance from when it was used to perform MCTBP sampling
    
    Args:
        runs: Path to the run files for each run to compare
        tags: List of data to extract from run parameters
    Returns:
        DataFrame containing performance summary
    """
    data = []
    for run in runs:
        # Get the energy tolerance of my run
        params = json.loads((run / 'runparams.json').read_text())
        energy_tol = params['energy_tolerance']
        
        # Load in the duplicates
        duplicates = json.loads((run / 'duplicates.json').read_text())

        # Process all of them
        run_data = []
        for duplicate in duplicates:
            try:
                dup_data = summarize_mctbp_results(Path(duplicate))
                dup_data['run'] = duplicate
                run_data.append(dup_data)
            except FileNotFoundError:
                print(f'Could not assess {duplicate} for {run}')
                continue
        n_duplicates = len(run_data)
        run_data = pd.concat(run_data)

        # Store the mean and SEM for each
        record = {'name': run.name}
        record.update(dict((c, params[c]) for c in tags))
        record['n_duplicates'] = n_duplicates
        record['number_sampled'] = len(run_data)

        for c in ['energy_error', 'force_error', 'max_force']:
            record[f'{c}-mean'] = run_data[c].mean()
            record[f'{c}-sem'] = run_data[c].sem()

        # Get the ranking performance for different structures
        tau = run_data.groupby(['structure', 'run'], group_keys=False).apply(lambda x: kendalltau(x['ml_energy'], x['target_energy'])[0])
        record['tau-mean'] = tau.mean()
        record['tau-sem'] = tau.sem()

        data.append(record)

        
    return pd.DataFrame(data)


def summarize_mctbp_results(run: Path) -> pd.DataFrame:
    """Get the energy and force errors from the MCTBP run
    
    Args:
        path: Run directory
    Returns:
        Dictionary containing the energy MAE (per atom) and Force RMSD
    """

    # Load the MCTBP results
    mctbp_dir = run / 'final-model' / 'mctbp-pure'
    data = pd.read_csv(mctbp_dir / 'sampled.csv')

    # Get the energy and force errors
    data['n_atoms'] = data['xyz'].apply(lambda x: int(x.split("\n")[0].strip()))
    data['energy_error'] = ((data['ml_energy'] - data['target_energy']) / data['n_atoms']).abs() * 1000
    for c in ['target_forces', 'ml_forces']:
        data[c] = data[c].apply(json.loads).apply(np.array)
    data['force_error'] = (data['target_forces'] - data['ml_forces']).apply(lambda x: np.linalg.norm(x, axis=1)).apply(lambda x: np.sqrt((x ** 2).sum()))
    
    # Store the max force for each frame
    data['max_force'] = data['target_forces'].apply(lambda x: np.linalg.norm(x, axis=1).max())
    
    return data


def summarize_md_results(run: Path, timestep: float = 0.2) -> pd.DataFrame:
    """Load the results of the MD runs performed using the surrogate model as the output
    
    Args:
        run: Path to the Colmena run output files
    Returns:
        Dataframe containing the errors associated which each MD test for this run
    """
    
    md_runs = run.glob('md-*')
    output = []
    for md_run in md_runs:
        data = pd.read_csv(md_run / 'sampled.csv')
        data['run'] = md_run.parent
        
        # Get the real time for each
        config = json.loads((md_run / 'params.json').read_text())
        data['time'] = config['md_log_frequency'] * data['minimum_num'] * timestep
        
        # Count the atoms
        data['n_atoms'] = data['xyz'].apply(lambda x: read_from_string(x, 'xyz')).apply(len)
        
        # Compute the errors
        data['energy_error_per_atom'] = (data['target_energy'] - data['ml_energy']).abs() / data['n_atoms']
        for c in ['target_energy', 'ml_energy']:
            data[f'{c}_per_atom'] = data[c] / data['n_atoms']
        for c in ['target_forces', 'ml_forces']:
            data[c] = data[c].apply(json.loads).apply(np.array)
        data['force_error'] = [x - y for x, y in zip(data['target_forces'], data['ml_forces'])]

        output.append(data)
    return pd.concat(output, ignore_index=True)
