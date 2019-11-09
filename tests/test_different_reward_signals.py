from gym_datums.wrappers.reward_wrappers import RelativeReward, OnlyFinalReward
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
