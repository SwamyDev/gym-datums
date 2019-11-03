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
    assert make_env().observation_space == make_box(minmax[0], minmax[1], shape=(1,))


def make_box(high, low, shape):
    return spaces.Box(high, low, shape, dtype=np.float32)


def test_multiple_values_observation_space(make_env, datums):
    datums.add().rows([2, 3, 1], [3, 4, 2])
    assert make_env().observation_space == make_box(1, 4, shape=(3,))


def test_multiple_datums_observation_space(make_env, datums):
    datums.add().rows([1], [2], [3])
    datums.add().rows([-1], [0], [1])
    assert make_env().observation_space == make_box(-1, 3, shape=(1, 2))


def test_windowed_observation_space(make_env, datums):
    datums.add().rows([1], [2], [3])
    assert make_env(window_size=2).observation_space == make_box(1, 3, shape=(1, 2))


@pytest.mark.parametrize("num_values, assets, window_size, shape", [
    (3, 2, 1, (3, 2)),
    (3, 1, 3, (3, 3)),
    (2, 5, 3, (2, 5, 3)),
])
def test_combined_setup_observation_space(make_env, datums, num_values, assets, window_size, shape):
    fill_datums(datums, assets, num_values)
    assert make_env(window_size).observation_space == make_box(1, 1, shape)


def fill_datums(datums, assets, num_values, value=1):
    for _ in range(assets):
        series = [[value] * num_values] * 10
        datums.add().rows(*series)


def test_reset_returns_first_value_from_datums(make_env, datums):
    datums.add().rows([1], )
    assert_obs_eq(make_env().reset(), [1])


@pytest.mark.parametrize('num_values, assets, window_size', [
    (3, 1, 1),
    (3, 2, 1),
    (1, 3, 1),
    (3, 1, 2),
    (1, 1, 5),
])
def test_observations_follow_observation_space_shape(make_env, datums, num_values, assets, window_size):
    fill_datums(datums, assets, num_values)
    env = make_env(window_size)
    assert env.reset().shape == env.observation_space.shape


def test_reset_returns_properly_windowed_observation(make_env, datums):
    datums.add().rows([1], [2], [3])
    assert_obs_eq(make_env(window_size=2).reset(), [[1, 2]])


def test_reset_combined_setup_observation(make_env, datums):
    datums.add().rows([2, 1, 3],
                      [3, 2, 4])
    datums.add().rows([0, -1, 1],
                      [1, 0, 2])
    assert_obs_eq(make_env(window_size=2).reset(), [[[2, 3], [0, 1]],
                                                    [[1, 2], [-1, 0]],
                                                    [[3, 4], [1, 2]]])
