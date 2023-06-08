"""Convert data from JSON to the format used by SchNetPack"""

from multiprocessing import Pool
from argparse import ArgumentParser
from ase.calculators.singlepoint import SinglePointCalculator
from ase import units
from schnetpack.data import AtomsData
from tqdm import tqdm
from glob import glob
from shutil import rmtree
import numpy as np
import gzip
import json
import ase
import os


def atoms_from_dict(record: dict) -> ase.Atoms:
    """Generate an ASE Atoms object from the graph dictionary
    Args:
        record: Record from which to generate an Atoms object
    Returns:
        atoms: The Atoms object
    """
    # Make the atoms object
    atoms = ase.Atoms(positions=record['coords'], numbers=record['z'])

    # Add energy, if available
    if 'energy' in record:
        forces = np.zeros_like(record['coords'])  # Assume we're at a local equilibrium
        calc = SinglePointCalculator(atoms, energy=record['energy'], forces=forces)
        atoms.set_calculator(calc)

    return atoms

# Parse inputs
parser = ArgumentParser()
parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing files")
parser.add_argument("--batch-size", type=int, default=10000, help="Number of objects to write per commit")
args = parser.parse_args()

# Get the input files to read
files = glob("../../../hydronet/data/output/geom_*.json.gz")

# Make the output directory
if os.path.isdir("data"):
    if args.overwrite:
        rmtree("data")
    else:
        raise ValueError("Output directory exists. Overwrite by adding --overwrite flag")
os.makedirs("data", exist_ok=False)

# Make the SPK databases
database = dict()
for name in ['train', 'test', 'valid']:
    data = AtomsData(
        f"./data/{name}.db",
        available_properties=['energy', 'forces']
    )
    database[name] = data

# Loop over each file
for file in files:
    # Get a name for this run
    name = os.path.basename(file)[5:-8]
    db = database[name]

    # Convert every cluster in the dataset to an ASE database
    with Pool() as pool:
        with gzip.open(file) as fp:
            # Wrap with a progress bar
            counter = tqdm(desc=name, leave=True)

            # Iterate over chunks
            while True:
                # Check a chunk of atoms objects
                records = [json.loads(i) for i, _ in zip(fp, range(args.batch_size))]
                atoms = pool.map(atoms_from_dict, records)

                # Collect the force and energy in eV (converting from the kcal/mol in hydronet)
                properties = [{'energy': np.array([a.get_potential_energy() / units.Hartree]),
                               'forces': np.zeros_like(a.positions)}
                              for a in atoms]

                # Add to the output
                db.add_systems(atoms, properties)
                counter.update(len(atoms))

                # If the batch is not batch_size, then it is because we
                #  ran out of input data
                if len(atoms) < args.batch_size:
                    break
