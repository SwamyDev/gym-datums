import pytest


@pytest.fixture(scope='session')
def gym_interface():
    return [('reset', ()), ('step', (0,)), ('render', ()), ('close', ())]


@pytest.fixture(scope='session')
def gym_properties():
    return ['action_space', 'observation_space']
