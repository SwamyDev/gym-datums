import numpy as np
from gym import spaces, Env
from more_itertools import first, collapse, windowed


class ReturnTransformer:
    def __init__(self):
        self._prv_obs = None

    def __call__(self, obs, get_next_obs):
        if self._prv_obs is None:
            self._prv_obs = obs
            obs = get_next_obs()
        ret = obs / self._prv_obs
        self._prv_obs = obs
        return ret


class IdentityTransformer:
    def __call__(self, obs, get_next_obs):
        return obs


class PortfolioEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, datums=None, window_size=1, calc_returns=False):
        self._datums = datums
        self._datums_iters = None
        self._window_size = window_size
        self._transformer = ReturnTransformer() if calc_returns else IdentityTransformer()
        self.action_space = spaces.Discrete(1)
        self._calc_shape = self._determine_shape()
        high, low = self._datums_minmax()
        self.observation_space = spaces.Box(low, high, self._obs_shape(), dtype=np.float32)
        self._upcoming_obs = None

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

    def reset(self):
        self._datums_iters = [windowed(d, self.window_size) for d in self._datums]
        obs = self._next_obs()
        try:
            self._upcoming_obs = self._next_obs()
        except StopIteration:
            raise DatumsError("Not enough data in the time series to create a single step.")
        return obs

    def _next_obs(self):
        obs = self._read_next_obs()
        obs = self._transformer(obs, self._read_next_obs)
        obs = self._shape_to_observation(obs)
        return obs

    def _read_next_obs(self):
        obs = np.empty(shape=self._calc_shape)
        for asset, it in enumerate(self._datums_iters):
            obs[:, asset, :] = np.array(next(it)).transpose()

        if not obs.all() or not np.isfinite(obs).all():
            raise DatumsError(f'Encountered zero, NaN or inf values in observation data: {obs}')

        return obs

    def _shape_to_observation(self, obs):
        obs = obs.squeeze()
        if self._calc_shape[0] == 1:
            obs = np.expand_dims(obs, axis=0)
        return obs

    def step(self, action):
        ret = self._upcoming_obs
        done = False
        try:
            self._upcoming_obs = self._next_obs()
        except StopIteration:
            done = True

        return ret, None, done, None

    def render(self, mode='human'):
        pass

    def close(self):
        pass


class DatumsError(ValueError):
    pass
