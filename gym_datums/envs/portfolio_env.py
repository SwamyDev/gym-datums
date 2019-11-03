import numpy as np
from gym import spaces, Env
from more_itertools import first, collapse


class PortfolioEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, datums=None, window_size=1):
        self._datums = datums
        self._window_size = window_size
        self.action_space = spaces.Discrete(1)
        high, low = self._datums_minmax()
        shape = self._determine_shape()
        self.observation_space = spaces.Box(low, high, shape, dtype=np.float32)

    def _datums_minmax(self):
        low = min(collapse(self._datums))
        high = max(collapse(self._datums))
        return high, low

    def _determine_shape(self):
        values = first(self._datums).shape[1]
        num_assets = len(self._datums)
        return (values,) + tuple(d for d in (num_assets, self.window_size) if d > 1)

    @property
    def window_size(self):
        return self._window_size

    def reset(self):
        d = first(self._datums)
        return first(d)

    def step(self, action):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass
