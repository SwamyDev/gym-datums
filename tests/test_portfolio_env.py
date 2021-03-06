import numpy as np
import pytest
from pytest import approx

from gym_datums.envs.portfolio_env import DatumsError, PortfolioResetNeeded
from tests.aux import assert_that, follows_contract, assert_obs_eq, unpack_obs, unpack_done, unpack_reward, until_done, \
    unpack_info, idle_step, make_box


def test_adherence_to_gym_contract(make_env, gym_interface, gym_properties):
    assert_that(make_env(), follows_contract(gym_interface, gym_properties))


@pytest.mark.parametrize('series, minmax', [
    ([[1]], (1, 1)),
    ([[2], [1], [3]], (1, 3)),
])
def test_single_value_single_datum_observation_space(make_env, datums, series, minmax):
    datums.add().rows(*series)
    assert make_env().observation_space == make_box(minmax[0], minmax[1], shape=(1,))


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
    datums.add().rows([1], [2])
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
    datums.add().rows(
        [2, 1, 3],
        [3, 2, 4],
        [1, 1, 1],
    )
    datums.add().rows(
        [1e-6, 1e-9, 1],
        [1, 1e-6, 2],
        [1, 1, 1],
    )
    assert_obs_eq(make_env(window_size=2).reset(), [[[2, 3], [1e-6, 1]],
                                                    [[1, 2], [1e-9, 1e-6]],
                                                    [[3, 4], [1, 2]]])


def test_stepping_the_environment_returns_next_observation(make_ready_env, datums):
    datums.add().rows([1], [2], [3])
    env = make_ready_env()
    assert_obs_eq(unpack_obs(idle_step(env)), [2])
    assert_obs_eq(unpack_obs(idle_step(env)), [3])


def test_resetting_the_environment_resets_observations(make_ready_env, datums):
    datums.add().rows([1], [2], [3])
    env = make_ready_env()
    assert_obs_eq(unpack_obs(idle_step(env)), [2])
    env.reset()
    assert_obs_eq(unpack_obs(idle_step(env)), [2])


def test_environment_is_done_at_the_last_observation(make_ready_env, datums):
    datums.add().rows([1], [2], [3])
    env = make_ready_env()
    assert not unpack_done(idle_step(env))
    assert unpack_done(idle_step(env))


def test_raise_an_error_when_not_enough_data_for_a_single_step(make_env, datums):
    datums.add().rows([1], [2])
    env = make_env(window_size=2)
    with pytest.raises(DatumsError):
        env.reset()


def test_raise_an_error_when_stepping_past_done(make_ready_env, datums):
    datums.add().rows([1], [2])
    env = make_ready_env()
    list(until_done(env, [1, 0]))
    with pytest.raises(PortfolioResetNeeded):
        idle_step(env)


@pytest.mark.parametrize('invalid', [0, np.nan, np.inf, -1e-5])
def test_raise_error_when_upcoming_invalid_datum_is_encountered(make_ready_env, datums, invalid):
    datums.add().rows([2], [1], [invalid])
    env = make_ready_env()
    with pytest.raises(DatumsError):
        idle_step(env)


@pytest.mark.parametrize('num_assets', [1, 2])
def test_action_shape_is_dependent_on_number_of_assets(make_ready_env, datums, num_assets):
    fill_datums(datums, num_assets, 1)
    env = make_ready_env()
    assert env.action_space == make_box(low=-1, high=1, shape=(num_assets + 1,))


def test_idle_reward(make_ready_env, datums):
    datums.add().rows([1], [1], [2])
    env = make_ready_env(cash=10)
    assert unpack_reward(idle_step(env)) == 1.0


def test_positive_reward(make_ready_env, datums):
    datums.add().rows([1], [1], [2], [1], [2])
    env = make_ready_env(cash=10)
    assert unpack_reward(env.step([0, 1])) == 1.0
    assert unpack_reward(env.step([1, 0])) == 2.0
    assert unpack_reward(env.step([0, 1])) == 2.0
    assert unpack_reward(env.step([1, 0])) == 4.0


def test_negative_reward(make_ready_env, datums):
    datums.add().rows([1], [1], [0.5], [1], [0.5])
    env = make_ready_env(cash=10)
    assert unpack_reward(env.step([0, 1])) == 1.0
    assert unpack_reward(env.step([1, 0])) == 0.5
    assert unpack_reward(env.step([0, 1])) == 0.5
    assert unpack_reward(env.step([1, 0])) == 0.25


def test_reset_returns_portfolio_to_original_state(make_ready_env, datums):
    datums.add().rows([1], [2])
    env = make_ready_env(cash=10)
    env.step([0, 1])
    assert_portfolio(env.portfolio, np.array([0, 5]))
    env.reset()
    assert_portfolio(env.portfolio, np.array([10, 0]))


def assert_portfolio(actual, expected):
    np.testing.assert_almost_equal(actual.assets, expected)


def test_by_default_use_buy_and_hold_baseline(make_ready_env, datums):
    datums.add().rows([1], [2], [0.5], [4])
    env = make_ready_env(cash=10)
    assert unpack_info(idle_step(env))['baseline'] == 2
    assert unpack_info(idle_step(env))['baseline'] == 0.5
    assert unpack_info(idle_step(env))['baseline'] == 4


def test_buy_and_hold_baseline_with_multiple_assets(make_ready_env, datums):
    datums.add().rows([1], [2], [0.5], [4])
    datums.add().rows([1], [1], [3], [2])
    env = make_ready_env(cash=10)
    assert unpack_info(idle_step(env))['baseline'] == 1.5
    assert unpack_info(idle_step(env))['baseline'] == 3.5 / 2
    assert unpack_info(idle_step(env))['baseline'] == 3


def test_configure_datum_as_baseline(make_ready_env, datums, baseline_datums):
    datums.add().rows([1], [2], [0.5])
    baseline_datums.rows(10, 5, 20)
    env = make_ready_env(cash=10, baseline=baseline_datums)
    assert unpack_info(idle_step(env))['baseline'] == 0.5
    assert unpack_info(idle_step(env))['baseline'] == 2


def test_reset_baseline_datums(make_ready_env, datums, baseline_datums):
    datums.add().rows([1], [2], [0.5])
    baseline_datums.rows(10, 5, 20)
    env = make_ready_env(cash=10, baseline=baseline_datums)
    assert unpack_info(idle_step(env))['baseline'] == 0.5
    assert unpack_info(idle_step(env))['baseline'] == 2
    env.reset()
    assert unpack_info(idle_step(env))['baseline'] == 0.5
    assert unpack_info(idle_step(env))['baseline'] == 2


def test_commission_fees_are_deducted(make_ready_env, datums):
    datums.add().rows([1], [1], [2], [1], [2])
    env = make_ready_env(cash=10, commission=0.1)
    assert unpack_reward(env.step([0, 1])) == 0.9
    assert unpack_reward(env.step([1, 0])) == approx((2 * 0.9) * 0.9)
    assert unpack_reward(env.step([0, 1])) == approx((2 * 0.9) * 0.9 ** 2)
    assert unpack_reward(env.step([1, 0])) == approx((2 * (2 * 0.9) * 0.9 ** 2) * 0.9)


def test_commission_fees_are_not_deducted_when_assets_are_not_moved(make_ready_env, datums):
    datums.add().rows([1], [1], [2])
    env = make_ready_env(cash=10, commission=0.1)
    assert unpack_reward(idle_step(env)) == 1.0
    assert unpack_reward(idle_step(env)) == 1.0


def test_commission_fees_are_deducted_on_all_shifted_assets(make_ready_env, datums):
    datums.add().rows([1], [1], [3])
    datums.add().rows([1], [1], [0.5])
    env = make_ready_env(cash=10, commission=0.1)
    assert unpack_reward(env.step([0, 0.6, 0.4])) == 0.9
    assert unpack_reward(env.step([1, 0, 0])) == approx((2 * 0.9) * 0.9)


def test_commission_fees_on_real_example(make_ready_env, datums):
    datums.add().rows([6425.8], [6425.8])
    env = make_ready_env(cash=100, commission=0.0016)
    env.step([0.0997454, 0.9002546])
    assert_portfolio(env.portfolio, np.array([9.97454, 0.01398758]))
