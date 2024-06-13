import shutil
from pathlib import Path
from functools import partial
from tempfile import TemporaryDirectory
from typing import Optional, Callable, Iterator

import torch
from ase.db import connect
from ase import Atoms
from ase.db.row import AtomsRow
from torch_geometric.data import InMemoryDataset, Data


def convert_atoms_to_pyg(mol: Atoms) -> Data:
    """Convert an Atoms object to a PyG Data object

    Args:
        mol: Atoms object to be converted
    Returns:
        Converted object ready to be used in a PyG SchNet model
    """

    # Normalize the coordinates
    if any(mol.pbc):
        # Ensure they are within the box
        mol = mol.copy()
        mol.wrap()
        pos = mol.get_positions()
    else:
        # Center the cluster around 0
        pos = mol.get_positions() - mol.get_center_of_mass()
    pos = torch.tensor(pos, dtype=torch.float)  # coordinates

    # Only operate if there is a calculator
    y = torch.zeros((), dtype=torch.float)  # potential energy
    f = torch.zeros_like(pos)
    if mol.calc is not None:
        results = mol.calc.results
        if 'energy' in results:
            y = torch.tensor(mol.get_potential_energy(), dtype=torch.float)  # potential energy
        if 'forces' in results:
            f = torch.tensor(mol.get_forces(), dtype=torch.float)  # gradients

    # Convert it to a PyG data record
    size = int(pos.size(dim=0) / 3)
    atomic_number = mol.get_atomic_numbers()
    z = torch.tensor(atomic_number, dtype=torch.long)
    return Data(x=z, z=z, pos=pos, y=y, f=f, n_atoms=len(mol), size=size)


def _preprocess(row: AtomsRow, pre_filter, pre_transform) -> Data | None:
    """Preprocess function used by the dataset.

    As a static function, so we can use it in a process pool"""
    mol = row.toatoms()
    init_data = convert_atoms_to_pyg(mol)
    data = init_data

    if pre_filter is not None and pre_filter(data):
        return
    if pre_transform is not None:
        data = pre_transform(data)
    return data


# TODO (wardlt): Write diskless version
class AtomsDataset(InMemoryDataset):
    """Dataset created from a list of ase.Atoms objects"""

    def __init__(self,
                 db_path: str | Path,
                 root: Optional[str] = None,
                 max_size: int | None = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None
                 ):
        """
        Args:
            root: Directory where processed output will be saved
            max_size: Maximum number of entries in dataset. Will pick the first entries.
            transform: Transform to apply to data
            pre_transform: Pre-transform to apply to data
            pre_filter: Pre-filter to apply to data
        """
        self.db_path = Path(db_path).absolute()
        self.max_size = max_size
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # NB: database file
        return [self.db_path.name]

    @property
    def processed_file_names(self):
        return ['dataset.pt']

    def download(self):
        # Keep a record of the raw data along with the processed results
        shutil.copyfile(self.db_path, self.raw_paths[0])

    def process(self):
        # NB: coding for atom types
        data_list = []

        # Loop over all datasets (there should be only one)
        proc_fun = partial(_preprocess, pre_filter=self.pre_filter, pre_transform=self.pre_transform)
        for db_path in self.raw_paths:
            # Loop over all records
            def _generate_atoms(batch_size=100000) -> Iterator[Atoms]:
                row_count = 0
                with connect(db_path) as conn:
                    done = False
                    while not done:
                        done = True

                        # Determine how many entries to draw
                        if self.max_size is None:
                            to_draw = batch_size  # The batch size
                        else:
                            to_draw = min(self.max_size - row_count, batch_size)  # Either the batch size or target number
                            if to_draw <= 0:
                                break

                        # Pull that chunk
                        for row in conn.select('', offset=row_count, limit=to_draw):
                            done = False  # We're not done if we hit any data
                            row_count += 1
                            yield row

            for x in map(proc_fun, _generate_atoms()):
                if x is not None:
                    data_list.append(x)

        # Collate once
        output = self.collate(data_list)
        del data_list
        torch.save(output, self.processed_paths[0])
        return output

    @classmethod
    def from_atoms(cls, atoms_lst: list[Atoms], root: str | Path, **kwargs) -> 'AtomsDataset':
        """Make a data loader from a list of ASE atoms objects

        Args:
            atoms_lst: List of atoms objects to put in dataset
            root: Directory to store the cached data
        """

        with TemporaryDirectory() as tmp:
            # Write a database that will get deleted after you make the loader
            db_path = Path(tmp) / 'temp.db'
            with connect(db_path, type='db') as conn:
                for atoms in atoms_lst:
                    conn.write(atoms)

            # Make the loader
            return cls(db_path, root=root, **kwargs)
