"""Test the TAMM interface"""
from pathlib import Path
import json

from ase.build import molecule
from pytest import fixture

from fff.simulation.tamm import TAMMCalculator

_test_files = Path(__file__).parent / 'files'
tamm_dir = Path('/home/lward/Software/nwchemex/tamm_install/bin/')


@fixture(params=[
    'ccsd_t', 'cd_ccsd', 'scf', 'cd_mp2'
])
def tamm_command(request):
    level = request.param
    if tamm_dir.exists():
        exc = 'HartreeFock' if level == 'scf' else level.upper()
        return f"mpirun -n 2 {tamm_dir}/{exc} tamm.json > tamm.out"
    else:
        return f'bash -c "mkdir -p tamm.cc-pvdz_files/restricted/json;' \
               f' cp {_test_files / "tamm-h2o.output.json"} tamm.cc-pvdz_files/restricted/json/tamm.{level}.json";' \
               f' cp {_test_files / "tamm-h2o.out"} tamm.out'


@fixture()
def atoms():
    return molecule('H2O')


@fixture()
def tamm_template():
    with (_test_files / 'tamm-h2o.json').open() as fp:
        return json.load(fp)


def test_tamm(tamm_command, tamm_template, atoms, tmpdir):
    calc = TAMMCalculator(command=tamm_command, basisset='cc-pvdz', template=tamm_template, directory=tmpdir)
    eng = calc.get_potential_energy(atoms)
    assert eng < 0
