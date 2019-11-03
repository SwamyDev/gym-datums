import numpy as np
import pytest
from gym import spaces

from gym_datums.envs import PortfolioEnv
from tests.aux import assert_that, follows_contract, assert_obs_eq


@pytest.fixture
def make_env(datums):
    def factory(window_size=1):
        return PortfolioEnv(datums.get_list(), window_size=window_size)

    return factory


def test_adherence_to_gym_contract(make_env, gym_interface, gym_properties):
    assert_that(make_env(), follows_contract(gym_interface, gym_properties))


@pytest.mark.parametrize('series, minmax', [
    ([[0]], (0, 0)),
    ([[-1], [0], [1]], (-1, 1)),
])
def test_single_value_single_datum_observation_space(make_env, datums, series, minmax):
    datums.add().rows(*series)
    assert make_env().observation_space == spaces.Box(minmax[0], minmax[1], shape=(1,), dtype=np.float32)


def test_multiple_values_observation_space(make_env, datums):
    datums.add().rows([2, 3, 1], [3, 4, 2])
    assert make_env().observation_space == spaces.Box(1, 4, shape=(3,), dtype=np.float32)


def test_multiple_datums_observation_space(make_env, datums):
    datums.add().rows([1], [2], [3])
    datums.add().rows([-1], [0], [1])
    assert make_env().observation_space == spaces.Box(-1, 3, shape=(1, 2), dtype=np.float32)


def test_windowed_observation_space(make_env, datums):
    datums.add().rows([1], [2], [3])
    assert make_env(window_size=2).observation_space == spaces.Box(1, 3, shape=(1, 2), dtype=np.float32)


@pytest.mark.parametrize("values, assets, window_size, shape", [
    (3, 2, 1, (3, 2)),
    (3, 1, 3, (3, 3)),
    (2, 5, 3, (2, 5, 3)),
])
def test_combined_setup_observation_space(make_env, datums, values, assets, window_size, shape):
    for _ in range(assets):
        series = [[1] * values] * 10
        datums.add().rows(*series)
    assert make_env(window_size).observation_space == spaces.Box(1, 1, shape, dtype=np.float32)


def test_reset_returns_first_value_from_datums(make_env, datums):
    datums.add().rows([1], )
    assert_obs_eq(make_env().reset(), [1])
