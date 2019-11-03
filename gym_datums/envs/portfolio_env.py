import numpy as np
from gym import spaces, Env
from more_itertools import first, collapse, windowed


class PortfolioEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, datums=None, window_size=1, calc_returns=False):
        self._datums = datums
        self._datums_iters = None
        self._window_size = window_size
        self._calc_returns = calc_returns
        self.action_space = spaces.Discrete(1)
        self._calc_shape = self._determine_shape()
        high, low = self._datums_minmax()
        self.observation_space = spaces.Box(low, high, self._obs_shape(), dtype=np.float32)
        self._prv_raw_obs = None

    def _datums_minmax(self):
        low = min(collapse(self._datums))
        high = max(collapse(self._datums))
        return high, low

    def _determine_shape(self):
        values = first(self._datums).shape[1]
        num_assets = len(self._datums)
        return values, num_assets, self.window_size

    def _obs_shape(self):
        return (first(self._calc_shape),) + tuple(d for d in self._calc_shape[1:] if d > 1)

    @property
    def window_size(self):
        return self._window_size

    @property
    def calc_returns(self):
        return self._calc_returns

    def reset(self):
        self._datums_iters = [windowed(d, self.window_size) for d in self._datums]
        obs = self._next_obs()
        return obs

    def _next_obs(self):
        raw = self._read_obs_from_datums()
        if self.calc_returns:
            raw = self._transform_to_return(raw)
        obs = self._shape_to_observation(raw)
        return obs

    def _read_obs_from_datums(self):
        obs = np.empty(shape=self._calc_shape)
        for asset, it in enumerate(self._datums_iters):
            obs[:, asset, :] = np.array(next(it)).transpose()
        return obs

    def _transform_to_return(self, raw):
        if self._prv_raw_obs is None:
            self._prv_raw_obs = raw
            raw = self._read_obs_from_datums()
        raw /= self._prv_raw_obs
        return raw

    def _shape_to_observation(self, obs):
        obs = obs.squeeze()
        if self._calc_shape[0] == 1:
            obs = np.expand_dims(obs, axis=0)
        return obs

    def step(self, action):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass
