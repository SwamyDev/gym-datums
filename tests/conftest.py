from functools import wraps

import numpy as np
import pytest

from gym_datums.envs import PortfolioEnv


class DatumStub:
    def __init__(self):
        self._rows = [[1e-9], [1e-9]]

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


@pytest.fixture()
def baseline_datums():
    return DatumStub()


@pytest.fixture
def make_env(datums):
    def factory(window_size=1, calc_returns=False, cash=1, baseline=None):
        return PortfolioEnv(datums.get_list(), window_size, cash, calc_returns, baseline)

    return factory


@pytest.fixture
def make_ready_env(make_env):
    @wraps(make_env)
    def reset_wrapper(*args, **kwargs):
        env = make_env(*args, **kwargs)
        env.reset()
        return env

    return reset_wrapper
