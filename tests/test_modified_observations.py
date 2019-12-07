import pytest
from numpy import log

from gym_datums.wrappers.observation_wrappers import LogReturnsWrapper
from tests.aux import assert_obs_eq, unpack_obs, idle_step, make_box


@pytest.fixture
def make_returns_env(make_env):
    def factory(*args, **kwargs):
        return LogReturnsWrapper(make_env(*args, **kwargs))

    return factory


@pytest.fixture
def make_ready_returns_env(make_returns_env):
    def factory(*args, **kwargs):
        env = make_returns_env(*args, **kwargs)
        env.reset()
        return env

    return factory


def test_resetting_with_returns(make_returns_env, datums):
    datums.add().rows(
        [2, 1, 4],
        [4, 2, 8],
        [1, 4, 2],
        [1, 1, 1],
    )
    assert_obs_eq(make_returns_env(window_size=2).reset(), [[log(2), log(0.25)],
                                                            [log(2), log(2)],
                                                            [log(2), log(0.25)]])


def test_stepping_with_returns(make_ready_returns_env, datums):
    datums.add().rows(
        [2, 1, 4],
        [4, 2, 8],
        [1, 4, 2],
        [1, 1, 1],
    )
    env = make_ready_returns_env(window_size=2)
    assert_obs_eq(unpack_obs(idle_step(env)), [[log(0.25), log(1)],
                                               [log(2), log(0.25)],
                                               [log(0.25), log(0.5)]])


def test_multiple_assets_with_returns(make_returns_env, datums):
    datums.add().rows(
        [2, 1, 4],
        [4, 2, 8],
        [1, 4, 2],
        [1, 1, 1],
    )

    datums.add().rows(
        [2, 2, 4],
        [8, 1, 2],
        [1, 8, 6],
        [1, 1, 1],
    )
    assert_obs_eq(make_returns_env(window_size=2).reset(), [[[log(2), log(0.25)], [log(4), log(0.125)]],
                                                            [[log(2), log(2)], [log(0.5), log(8)]],
                                                            [[log(2), log(0.25)], [log(0.5), log(3)]]])


def test_returns_with_single_values(make_returns_env, datums):
    datums.add().rows([2], [2], [3])
    env = make_returns_env(window_size=1)
    assert_obs_eq(env.reset(), [log(1)])
    assert_obs_eq(unpack_obs(idle_step(env)), [log(1.5)])


def test_returns_wrapper_adjusts_the_min_max_range_properly(make_returns_env, datums):
    datums.add().rows(
        [2, 2, 4],
        [8, 1, 2],
        [1, 8, 6],
        [1, 1, 1],
    )
    assert make_returns_env().observation_space == make_box(high=log(8), low=log(0.125), shape=(3,))
