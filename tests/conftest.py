import numpy as np
import pytest


class DatumStub:
    def __init__(self):
        self._rows = [[0]]

    def rows(self, *r):
        self._rows = list(r)

    def __iter__(self):
        return iter(self._rows)

    @property
    def shape(self):
        return np.array(self._rows).shape


class DatumsStub:
    def __init__(self):
        self._datums = []

    def add(self):
        self._datums.append(DatumStub())
        return self._datums[-1]

    def get_list(self):
        return self._datums or [DatumStub()]


@pytest.fixture(scope='session')
def gym_interface():
    return [('reset', ()), ('step', (0,)), ('render', ()), ('close', ())]


@pytest.fixture(scope='session')
def gym_properties():
    return ['action_space', 'observation_space']


@pytest.fixture()
def datums():
    return DatumsStub()
