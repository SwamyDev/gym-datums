from gym import spaces, Env


class PortfolioEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, window_size=1):
        self._window_size = window_size
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Discrete(2)

    @property
    def window_size(self):
        return self._window_size

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass
