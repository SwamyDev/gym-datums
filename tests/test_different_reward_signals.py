import numpy as np
from pytest import approx

from gym_datums.wrappers.reward_wrappers import RelativeReward, OnlyFinalReward, SharpRatioReward
from tests.aux import unpack_reward


def test_immediate_relative_reward(make_ready_env, datums):
    datums.add().rows([1], [1], [0.5], [0.5], [1])
    env = RelativeReward(make_ready_env(cash=10))
    assert unpack_reward(env.step([0, 1])) == 1
    assert unpack_reward(env.step([0, 1])) == 0.5
    assert unpack_reward(env.step([0, 1])) == 1
    assert unpack_reward(env.step([0, 1])) == 2


def test_relative_reward_resets_properly(make_ready_env, datums):
    datums.add().rows([1], [1], [0.5])
    env = RelativeReward(make_ready_env(cash=10))
    assert unpack_reward(env.step([0, 1])) == 1
    assert unpack_reward(env.step([0, 1])) == 0.5
    env.reset()
    assert unpack_reward(env.step([0, 1])) == 1
    assert unpack_reward(env.step([0, 1])) == 0.5


def test_final_value_of_portfolio_as_only_reward(make_ready_env, datums):
    datums.add().rows([1], [1], [2], [1], [2])
    env = OnlyFinalReward(make_ready_env(cash=10))
    assert unpack_reward(env.step([0, 1])) == 0.0
    assert unpack_reward(env.step([1, 0])) == 0.0
    assert unpack_reward(env.step([0, 1])) == 0.0
    assert unpack_reward(env.step([1, 0])) == 4.0


def test_calculate_growing_sharp_ratio(make_ready_env, datums, baseline_datums):
    datums.add().rows([1], [1], [2], [1], [2])
    baseline_datums.rows(1.0, 1.2, 1.4, 1.6, 2)
    env = SharpRatioReward(make_ready_env(cash=10, baseline=baseline_datums))
    assert unpack_reward(env.step([0, 1])) == approx(-0.2)
    assert unpack_reward(env.step([1, 0])) == approx(np.mean([-0.2, 0.6]) / np.std([-0.2, 0.6], ddof=1))
    assert unpack_reward(env.step([0, 1])) == approx(np.mean([-0.2, 0.6, 0.4]) / np.std([-0.2, 0.6, 0.4], ddof=1))
    assert unpack_reward(env.step([1, 0])) == approx(np.mean([-0.2, 0.6, 0.4, 2]) / np.std([-0.2, 0.6, 0.4, 2], ddof=1))


def test_combine_reward_wrappers(make_ready_env, datums, baseline_datums):
    datums.add().rows([1], [1], [2], [1], [2])
    baseline_datums.rows(1.0, 1.2, 1.4, 1.6, 2)
    env = OnlyFinalReward(SharpRatioReward(make_ready_env(cash=10, baseline=baseline_datums)))
    assert unpack_reward(env.step([0, 1])) == 0
    assert unpack_reward(env.step([1, 0])) == 0
    assert unpack_reward(env.step([0, 1])) == 0
    assert unpack_reward(env.step([1, 0])) == approx(np.mean([-0.2, 0.6, 0.4, 2]) / np.std([-0.2, 0.6, 0.4, 2], ddof=1))
