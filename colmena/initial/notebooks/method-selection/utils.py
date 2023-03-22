"""Utilities for comparing model performance"""
from pathlib import Path
from scipy.stats import kendalltau
import numpy as np
import json

import pandas as pd

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
        record = dict((c, params[c]) for c in tags)
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

        # Drop outliers
        iqr = np.percentile(run_data['energy_error'], 75) - np.percentile(run_data['energy_error'], 25)
        outlier_thresh = run_data['energy_error'].median() + 4 * iqr
        run_data.query(f'energy_error < {outlier_thresh}', inplace=True)

        # Store the mean and SEM for each
        record = dict((c, params[c]) for c in tags)
        record['n_duplicates'] = n_duplicates
        record['number_sampled'] = len(run_data)

        for c in ['energy_error', 'force_error', 'max_force']:
            record[f'{c}-mean'] = run_data[c].mean()
            record[f'{c}-sem'] = run_data[c].sem()

        # Get the ranking performance for different structures
        tau = run_data.groupby(['structure', 'run'], group_keys=False).apply(lambda x: kendalltau(x['ml_energy'], x['ttm_energy'])[0])
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
    mctbp_dir = run / 'final-model' / 'mctbp'
    data = pd.read_csv(mctbp_dir / 'sampled.csv')

    # Get the energy and force errors
    data['energy_error'] = ((data['ml_energy'] - data['ttm_energy']) / data['n_waters'] / 3).abs() * 1000
    for c in ['ttm_forces', 'ml_forces']:
        data[c] = data[c].apply(json.loads).apply(np.array)
    data['force_error'] = (data['ttm_forces'] - data['ml_forces']).apply(lambda x: np.linalg.norm(x, axis=1)).apply(lambda x: np.sqrt((x ** 2).sum()))
    
    # Store the max force for each frame
    data['max_force'] = data['ttm_forces'].apply(lambda x: np.linalg.norm(x, axis=1).max())
    
    return data
