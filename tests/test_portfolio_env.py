import pytest

from gym_datums.envs import PortfolioEnv
from tests.aux import assert_that, follows_contract


@pytest.fixture
def env():
    return PortfolioEnv()


def test_adherence_to_gym_contract(env, gym_interface, gym_properties):
    assert_that(env, follows_contract(gym_interface, gym_properties))
