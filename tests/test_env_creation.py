import gym


def test_creating_environment(datums):
    env = gym.make('gym_datums:portfolio-v0', datums=datums.get_list())
    assert env


def test_pass_configuration_to_environment(datums):
    env = gym.make('gym_datums:portfolio-v0', datums=datums.get_list(), window_size=10)
    assert env.window_size == 10
