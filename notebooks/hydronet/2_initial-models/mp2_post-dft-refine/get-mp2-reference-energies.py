"""Compute reference energies for different molecules"""
from pathlib import Path
import json

from ase.calculators.socketio import SocketIOCalculator
from ase.calculators.nwchem import NWChem
from ase.optimize import QuasiNewton
from ase import build

from fff.simulation.utils import write_to_string

unixsocket = 'nwchem'

if __name__ == "__main__":
    # Get the output file
    out_file = Path('reference-energies.json')
    if out_file.is_file():
        reference_energies = json.loads(out_file.read_text())
    else:
        reference_energies = {}

    # Make the run directory
    run_dir = Path('nwchem-opt')
    run_dir.mkdir(exist_ok=True)

    for mol_name in ['H2O', 'H2']:
        atoms = build.molecule(mol_name)
        for basis in ['aug-cc-pvdz', 'aug-cc-pvtz', 'aug-cc-pvqz']:
            # Skip if we've already done this molecule
            if basis in reference_energies.get(mol_name, {}):
                continue

            # Set up NWChem calculator
            nwchem = NWChem(
                directory=str(run_dir),
                theory='mp2',
                basis={'*': basis},
                pretasks=[{
                    'theory': 'dft',
                    'dft': {
                        'xc': 'hfexch',
                        'maxiter': 50,
                    },
                    'set': {
                        'quickguess': 't',
                        'fock:densityscreen': 'f',
                        'lindep:n_dep': 0,
                    }
                }],
                set={
                    'lindep:n_dep': 0,
                    'cphf:maxsub': 95,
                    'mp2:aotol2e fock': '1d-14',
                    'mp2:aotol2e': '1d-14',
                    'mp2:backtol': '1d-14',
                    'cphf:maxiter': 999,
                    'cphf:thresh': '6.49d-5',
                    'int:acc_std': '1d-16'
                },
                scf={'maxiter': 99, 'tol2e': '1d-15'},
                mp2={'freeze': 'atomic'},
                task='optimize',
                driver={'socket': {'unix': unixsocket}}
            )

            # Optimize the structure
            with SocketIOCalculator(nwchem, unixsocket=unixsocket) as calc:
                atoms.calc = calc
                QuasiNewton(atoms, logfile=run_dir / 'opt.log').run(0.01)
                energy = atoms.get_potential_energy()

            # Save the energy
            if mol_name not in reference_energies:
                reference_energies[mol_name] = {}
            reference_energies[mol_name][basis] = {'energy': energy, 'xyz': write_to_string(atoms, 'xyz')}
            out_file.write_text(json.dumps(reference_energies, indent=2))
