from pathlib import Path

from ase.db import connect
from pytest import fixture

_my_path = Path(__file__).parent


@fixture()
def example_waters():
    with connect(_my_path / 'files' / 'test.db') as db:
        return [x.toatoms() for x in db.select('')]
