import gym


def test_creating_environment():
    env = gym.make('gym_datums:portfolio-v0')
    assert env


def test_pass_configuration_to_environment():
    env = gym.make('gym_datums:portfolio-v0', window_size=10)
    assert env.window_size == 10
