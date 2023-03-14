"""Python implementation of the Monte Carlo Thermal Basin Paving algorithm

Based off of code by Kristina Herman: https://github.com/HenrySprueill/fast-finedtuned-forcefields/raw/main/utils/mctbp.py"""
from tempfile import TemporaryDirectory
from pathlib import Path
import logging

from ase.calculators.calculator import Calculator
from ase.optimize.sciopt import SciPyFminBFGS
from ase.io.trajectory import Trajectory
import ase
import numpy as np
import math

import fff.sampling.mctbp.water_cluster_moves as mcm
from fff.sampling.base import CalculatorBasedSampler

logger = logging.getLogger(__name__)


class MCTBP(CalculatorBasedSampler):
    """The Monte Carlo Thermal Basin Paving technique

    Returns both the last structure sampled and any produced optimization
    """

    def __init__(self,
                 fmax: float = 0.005,
                 min_tol: float = 1.2,
                 max_disp: float = 1.,
                 temp: float = 0.8,
                 use_eff_temp: bool = True,
                 use_dsi: bool = True,
                 return_incomplete: bool = True,
                 return_minima_only: bool = False,
                 scratch_dir: Path | None = None):
        """

        Args:
            fmax: Convergence criterion for optimization steps
            min_tol: minimum distance between atoms after MC move to make it an "acceptable" structure
            max_disp: Maximum displacement during monte carlo move
            temp: Monte carlo temperature
            use_eff_temp: Whether to adjust the temperature during
            use_dsi: Use dissimilarity index to judge structures
            return_incomplete: Whether to return structures even if part of the sampling procedure fails
            return_minima_only: Whether to only return minima produced at each step
            scratch_dir: Location in which to store temporary files
        """
        super().__init__(scratch_dir=scratch_dir)
        self.fmax = fmax
        self.min_tol = min_tol
        self.max_disp = max_disp
        self.temp = temp
        self.use_eff_temp = use_eff_temp
        self.return_incomplete = return_incomplete
        self.return_minima_only = return_minima_only
        self.use_dsi = use_dsi

    def _run_sampling(self, atoms: ase.Atoms, steps: int, calc: Calculator, **kwargs) -> (ase.Atoms, list[ase.Atoms]):
        """Run an MC-TBP simulation

        Args:
            atoms: Starting water cluster geometry
            calc: Calculator used for computing energy and forces
            steps: number of MC moves to make
        Returns:
            - Last structure during sampling run
            - Every structure sampled along the way
        """

        success, error_msg, last_strc, traj = self.perform_mctbp(atoms, steps, calc, **kwargs)
        if not (success or self.return_incomplete):
            raise ValueError(error_msg)
        return last_strc, traj

    def perform_mctbp(self, atoms: ase.Atoms, steps: int, calc: Calculator, **kwargs) -> (bool, str, ase.Atoms, list[ase.Atoms]):
        """Run an MC-TBP simulation

        Args:
            atoms: Starting water cluster geometry
            calc: Calculator used for computing energy and forces
            steps: number of MC moves to make
        Returns:
            - Whether the run completed successfully
            - Reason the run did not complete, if
            - Last structure during sampling run
            - Every structure sampled along the way
        """

        eff_temp = None

        # Initialize tracking arrays
        all_sampled = []  # all structures that were sampled
        energy_list = []  # energies that were sampled

        # Start by optimizing the structure
        opt_cluster, sampled_structures = optimize_structure(atoms, calc, self.scratch_dir, fmax=self.fmax)
        if self.return_minima_only:
            all_sampled.append(opt_cluster)
        else:
            all_sampled.extend(sampled_structures + [opt_cluster])
        energy_list.append(opt_cluster.get_potential_energy())  # Pulls a cached result
        curr_e = opt_cluster.get_potential_energy()

        # Loop over a set number of optimization steps
        new_cluster = opt_cluster.copy()  # Start with a copy of this initial structure
        for i in range(steps):
            # Start with a fresh copy of the initial cluster
            new_cluster = new_cluster.copy()

            # Perturb structure
            accept = False
            for _ in range(100000):
                mc_cluster, move_type = get_move(new_cluster.get_positions(), self.max_disp, len(atoms) // 3)
                accept = check_atom_overlap(mc_cluster, min_tol=self.min_tol)
                if accept:
                    break
            new_cluster.set_positions(mc_cluster)
            if not accept:  # TODO (wardlt): Implement a scheme that uses exceptions
                return False, 'Did not find an acceptable move', new_cluster, all_sampled

            # Find the nearest local minimum to this new cluster
            opt_new_cluster, sampled_structures = optimize_structure(new_cluster, calc, self.scratch_dir, fmax=self.fmax)
            if self.return_minima_only:
                all_sampled.append(opt_new_cluster)
            else:
                all_sampled.extend(sampled_structures + [opt_new_cluster])
            new_e = opt_new_cluster.get_potential_energy()
            energy_list.append(new_e)

            # Determine whether to accept the move
            if self.use_eff_temp:
                accept_move = move_acceptance(curr_e, new_e,
                                              new_cluster.get_positions(), opt_new_cluster.get_positions(),
                                              self.temp if eff_temp is None else eff_temp,
                                              use_dsi=self.use_dsi)
                bin_count = get_histogram_count(energy_list)
                eff_temp = calc_eff_temp(self.temp, bin_count)
            else:
                accept_move = move_acceptance(curr_e, new_e, self.temp)

            if accept_move:
                accept_structure = check_optimized_structure(opt_new_cluster.get_positions())
                if accept_structure:
                    new_cluster = opt_new_cluster
                    curr_e = new_e

        return True, None, new_cluster, all_sampled


def optimize_structure(atoms: ase.Atoms, calc: Calculator, scratch_dir: Path | None = None,
                       fmax: float = 1e-3, max_steps: int = 100, mult: float = 5) -> (ase.Atoms, list[ase.Atoms]):
    """Optimize new structure

    Args:
        atoms: Initial Structure
        fmax: Convergence threshold
        calc: Calculator used to compute energy and forces
        max_steps: Number of steps for each
        mult: Factor by which to increase convergence threshold id optimization fails
    Returns:
        - Minimum structure
        - Any sampled along the way
    """

    # Run everything inside a temporary directory
    with TemporaryDirectory(dir=scratch_dir, prefix='mctbp') as tmp:
        tmp = Path(tmp)

        # Store the trajectory
        atoms.calc = calc
        traj_path = tmp / 'optimization.traj'
        with Trajectory(traj_path, mode='w', atoms=atoms, properties=['energy', 'forces']) as traj:
            dyn = SciPyFminBFGS(atoms, logfile=tmp / 'optimization.log')
            dyn.attach(traj)

            # Attempt to optimize in 5 tries
            for attempt in range(5):
                try:
                    dyn.run(fmax=fmax * (mult ** attempt), steps=max_steps)
                except Exception:
                    continue  # Proceeds to a looser convergence criterion
                break

        # Read the trajectory of the states which were sampled
        with Trajectory(str(traj_path), mode='r') as traj:
            traj_atoms = [x for x in traj]
        return atoms, traj_atoms


def get_move(new_cluster, max_disp, num_waters):
    """
    For each molecule, either:
    1. Translate in x-, y-, z-coords
    2. Rotate about H-O-H bisector
    3. Rotate about O-H bond
    4. Rotate about O-O axis (with any other molecule)
    """
    move_type = np.random.randint(0, 4)
    if move_type == 0:  # Translate single molecule by (x,y,z)
        num_trans = np.random.randint(1, int(num_waters / 3))
        ind_trans = np.random.randint(0, num_waters, size=num_trans)
        for i in range(len(ind_trans)):
            rand_molecule = ind_trans[i]
            x_val = np.random.uniform(0, max_disp)
            y_val = np.random.uniform(0, max_disp)
            z_val = np.random.uniform(0, max_disp)
            new_cluster[rand_molecule * 3:rand_molecule * 3 + 3, :] += np.array([x_val, y_val, z_val])
    elif move_type == 1:  # rotate about H-O-H bisector
        num_rots = np.random.randint(1, int(num_waters) / 3)
        ind_rots = np.random.randint(0, num_waters, size=num_rots)
        for i in range(len(ind_rots)):
            mol_rot = ind_rots[i]
            theta_val = np.random.uniform(0, 2 * np.pi)
            new_cluster[mol_rot * 3:(mol_rot * 3 + 3), :] = mcm.rotate_around_HOH_bisector_axis(new_cluster[mol_rot * 3:(mol_rot * 3 + 3), :], theta_val)
        pass
    elif move_type == 2:  # rotate about O-H bond
        ind_rot = np.random.randint(0, num_waters)  # choose molecule to rotate
        theta_val = np.random.uniform(0, 2 * np.pi)
        monomer_geom = new_cluster[ind_rot * 3:ind_rot * 3 + 3, :]
        which_h = np.random.randint(0, 2)  # choose which hydrogen
        new_cluster[ind_rot * 3:ind_rot * 3 + 3, :] = mcm.rotate_around_local_axis(monomer_geom, 0, which_h + 1, theta_val)
    else:  # rotate about O-O axis
        ind_rots = np.random.randint(0, num_waters, size=2)  # choose two oxygens to rotate about
        theta_val = np.random.uniform(0, 2 * np.pi)
        dimer_geom = np.concatenate((new_cluster[ind_rots[0] * 3:(ind_rots[0] * 3 + 3), :], new_cluster[ind_rots[1] * 3:(ind_rots[1] * 3 + 3), :]), axis=0)
        rotate_dimer = mcm.rotate_around_local_axis(dimer_geom, 0, 3, theta_val)
        new_cluster[ind_rots[0] * 3:(ind_rots[0] * 3 + 3), :] = rotate_dimer[0:3, :]
        new_cluster[ind_rots[1] * 3:(ind_rots[1] * 3 + 3), :] = rotate_dimer[3:, :]
        pass

    return new_cluster, move_type


def check_optimized_structure(opt_cluster):
    """
    Determine whether any atoms in the new configuration are too close to one another
    args: xyz-coordinates of system after making a move
    return: whether the new configuration is allowed (or if atoms are overlapping)
    """
    # Check intramolecular geometry (angle: 80-135, bond: 0.82-1.27)
    accept = True
    cluster = opt_cluster.copy()
    hoh_ang = np.zeros(int(np.size(cluster, axis=0) / 3))
    oh_dist = np.zeros(int(np.size(cluster, axis=0) / 3) * 2)
    for i in range(int(np.size(cluster, axis=0) / 3)):
        oh_vec1 = cluster[3 * i, :] - cluster[3 * i + 1, :]
        oh_vec2 = cluster[3 * i, :] - cluster[3 * i + 2, :]
        oh_dist[2 * i] = np.linalg.norm(oh_vec1)
        oh_dist[2 * i + 1] = np.linalg.norm(oh_vec2)
        hoh_ang[i] = np.arccos(np.dot(oh_vec1, oh_vec2) / (oh_dist[2 * i] * oh_dist[2 * i + 1]))
    hoh_ang *= (180 / np.pi)
    if (0.82 > oh_dist).any() or (oh_dist > 1.27).any() or (80 > hoh_ang).any() or (hoh_ang > 135).any():
        accept = False

    return accept


def check_atom_overlap(new_config, min_tol=1.0):
    """
    Determine whether any atoms in the new configuration are too close to one another
    args: xyz-coordinates of system after making a move
    return: whether the new configuration is allowed (or if atoms are overlapping)
    """
    cluster = new_config.copy()
    dist_matrix = np.zeros((np.size(cluster, axis=0), np.size(cluster, axis=0)))
    for i in range(np.size(cluster, axis=0)):
        atom1 = cluster[i, :]
        diff = cluster - atom1
        dist_matrix[:, i] = np.sqrt(np.sum(np.square(diff), axis=1))
        dist_matrix[i, i] = 10
        if (i + 1) % 3 == 0:
            dist_matrix[i - 1, i] = 10
            dist_matrix[i - 2, i] = 10
            dist_matrix[i - 2, i - 1] = 10
            dist_matrix[i, i - 1] = 10
            dist_matrix[i, i - 2] = 10
            dist_matrix[i - 1, i - 2] = 10
    if np.min(dist_matrix) >= min_tol:
        accept = True
    else:
        accept = False

    return accept


def move_acceptance(prev_e, curr_e, prev_struct=None, curr_struct=None, temp=0.8, use_dsi=True, dsi_threshold=0.5, norm_const=0.5):
    """
    Calculates whether a move was accepted based on the change in energy and the temp
    args: energy before move (float), energy after move (float), structures (np.array), temperature (float)
    return: whether the move is accepted (bool)
    """
    if curr_e <= prev_e:
        accept = True
    else:
        if use_dsi:
            dsi = get_dsi(prev_struct, curr_struct)
            if dsi <= dsi_threshold:
                accept = False
            else:
                coeff = math.exp(min(1, -(curr_e - prev_e - norm_const * dsi) / temp))  # missing k_b
                rand_num = np.random.uniform(0, 1)
                if rand_num <= coeff:
                    accept = True
                else:
                    accept = False
        else:
            coeff = math.exp(min(1, -(curr_e - prev_e) / temp))  # missing k_b
            rand_num = np.random.uniform(0, 1)
            if rand_num <= coeff:
                accept = True
            else:
                accept = False
    return accept


def coords_to_xyz(coords_array):
    """
    Prints array of coordinates in standard xyz format (for visualization)
    args: numpy array of coordinates
    return: None
    """
    for i in range(int(np.size(coords_array, axis=0) / 3)):
        print(f'O  {coords_array[i * 3, 0]}  {coords_array[i * 3, 1]}  {coords_array[i * 3, 2]}')
        print(f'H  {coords_array[i * 3 + 1, 0]}  {coords_array[i * 3 + 1, 1]}  {coords_array[i * 3 + 1, 2]}')
        print(f'H  {coords_array[i * 3 + 2, 0]}  {coords_array[i * 3 + 2, 1]}  {coords_array[i * 3 + 2, 2]}')

    return None


def calc_eff_temp(t_init, bin_count, norm_coeff=0.015):
    """
    Calculate the "effective" temperature using initial temp and histogram of energies sampled
    args: initial temp (float), bin_count (int)
    return: effective temperature (float)
    """
    temp_eff = t_init + np.exp(norm_coeff * bin_count)
    return temp_eff


def get_histogram_count(energy_list, bin_width=0.5):
    """
    args: list of energies, width of bin
    return: bin count for that energy
    """
    bins = np.arange(min(energy_list), max(energy_list) + 2 * bin_width, bin_width)
    counts, bin_edges = np.histogram(energy_list, bins=bins)

    for i in range(len(bins) - 1):
        if bins[i] <= energy_list[-1] < bins[i + 1]:
            bin_idx = i

    return counts[bin_idx]


def get_dsi(prev_structure, curr_structure):
    """
    Get Dis-Similarity Index (DSI) for previous structure and current structure
    args: previously sampled structure and current structure (both numpy arrays)
    return: DSI (float)
    """
    prev_o = prev_structure[0:np.size(prev_structure, axis=0):3, :]
    curr_o = curr_structure[0:np.size(curr_structure, axis=0):3, :]

    nW = np.size(prev_o, axis=0)
    num_dists = int(nW * (nW - 1) / 2)

    pairwise_dist_prev = np.zeros(num_dists)
    pairwise_dist_curr = np.zeros(num_dists)

    counter = 0
    for i in range(0, nW - 1):
        for j in range(i + 1, nW):
            pairwise_dist_prev[counter] = np.linalg.norm(prev_o[i, :] - prev_o[j, :])
            pairwise_dist_curr[counter] = np.linalg.norm(curr_o[i, :] - curr_o[j, :])
            counter += 1

    dsi = np.linalg.norm(np.sort(pairwise_dist_prev) - np.sort(pairwise_dist_curr))

    return dsi


def xyz_to_numpy(filename, cluster_size):
    coords = np.zeros((cluster_size * 3, 3))
    with open(filename, 'r') as f:
        count = 0
        for line in f:
            if line.startswith('O '):
                atom = line.strip().split()
                coords[count, 0] = float(atom[1])
                coords[count, 1] = float(atom[2])
                coords[count, 2] = float(atom[3])
                count += 1
            elif line.startswith('H '):
                atom = line.strip().split()
                coords[count, 0] = float(atom[1])
                coords[count, 1] = float(atom[2])
                coords[count, 2] = float(atom[3])
                count += 1
            else:
                pass
    return coords
