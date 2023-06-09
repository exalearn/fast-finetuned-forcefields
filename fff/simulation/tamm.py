"""Interface to the Tensor Algebra for Many-body Methods (TAMM) library

We are using TAMM to run CC and MP2 computations until the full interface to NWChemEx is available.
"""
import json
import logging
from pathlib import Path
from os import path as osp

from ase.calculators.calculator import FileIOCalculator
from ase import units, Atoms

logger = logging.getLogger(__name__)


class TAMMCalculator(FileIOCalculator):
    """Tensor Algebra for Many-body Methods (TAMM) library interface provided by
    its `CoupledCluster <https://github.com/NWChemEx-Project/CoupledCluster/tree/CC>`_ interface.

    The method used by TAMM is defined by which executable you choose (e.g., /path/to/CCSD_T to run CCSD(T)),
    the basis is defined by the `basisset` argument,
    and all other options are defined as part of a template dictionary passed to `template`."""

    default_parameters = {
        'basisset': 'cc-pvdz',
        'template': {}
    }
    implemented_properties = ['energy']

    def write_input(self, atoms: Atoms, properties=None, system_changes=None):
        # Write the atoms to geometry output
        geometry = {
            'coordinates': [],
            'units': 'bohr'
        }
        positions_bohr = atoms.positions.copy() / units.Bohr
        for sym, pos in zip(atoms.symbols, positions_bohr):
            coords = "\t".join(f"{x:.8f}" for x in pos)
            geometry['coordinates'].append(
                f'{sym}\t{coords}'
            )

        # Join geometry and basis with the rest of the settings
        output = self.parameters['template'].copy()
        output['geometry'] = geometry
        output['basis']['basisset'] = self.parameters['basisset']

        # Write to disk in the output directory
        with open(osp.join(self.directory, 'tamm.json'), 'w') as fp:
            json.dump(output, fp, indent=2)

    def read_results(self):
        # Find the output directory
        output_paths = list(Path(self.directory).glob('tamm.*_files'))
        if len(output_paths) > 1:  # pragma: no cover
            raise ValueError(f'Found {len(output_paths)} output directories when expecting one')
        output_path = output_paths[0]

        # Find the highest-level computation that has been performed
        output_json_dir = output_path / 'restricted' / 'json'
        if not output_json_dir.exists():  # pragma: no cover
            raise ValueError('No output JSONs found')

        output_jsons = list(output_json_dir.glob('*.json'))
        output_json = None
        for level in ['ccsd_t', 'ccsd', 'scf']:
            for path in output_jsons:
                if path.name.endswith(f'{level}.json'):
                    output_json = path
                    break
            if output_json is not None:
                break

        if output_json is None:  # pragma: no cover
            raise ValueError('No output file found')

        logging.info(f'Reading from {output_json}')

        # Read in the energy
        with output_json.open() as fp:
            output = json.load(fp)["output"]
        if level == 'ccsd_t':
            energy = output["CCSD(T)"]["(T)Energies"]["total"]
        elif level == 'ccsd':
            energy = output["CCSD"]["final_energy"]["total"]
        elif level == "scf":
            energy = output["SCF"]["final_energy"]
        else:
            raise NotImplementedError(f'No support for {level} yet')
        self.results['energy'] = energy * units.Ha
