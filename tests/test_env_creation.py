import gym


def test_creating_environment():
    env = gym.make('gym_datums:portfolio-v0')
    assert env
