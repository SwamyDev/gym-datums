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


class Portfolio:
    cash_index = 0

    def __init__(self, cash, size):
        self._cash = cash
        self._action_mat = np.identity(size)
        self._assets = np.zeros(size)
        self._assets[self.cash_index] = self._cash
        self._prices = np.empty(size)
        self._prices[self.cash_index] = 1

    @property
    def assets(self):
        return self._assets

    def reset(self):
        self._assets = np.zeros_like(self._assets)
        self._assets[self.cash_index] = self._cash

    def update(self, observation):
        self._prices[1:] = observation[0, :, -1]

    def shift(self, action):
        m = self._make_price_shift_mat(action)
        self._assets = np.matmul(m, self.assets)

    def _make_price_shift_mat(self, action):
        p = np.expand_dims(self._prices, 0)
        p_mat = p * (1 / p).T
        shift = np.matmul((self._action_mat * action), p_mat)
        return shift

    def normalized_value(self):
        return np.matmul(self._prices, self.assets) / self._cash


class PortfolioEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, datums=None, window_size=1, calc_returns=False, cash=1, relative_reward=False):
        self._datums = datums
        self._datums_iters = None
        self._window_size = window_size
        self._transformer = ReturnTransformer() if calc_returns else IdentityTransformer()
        self.action_space = spaces.Box(1, -1, (self.num_assets + 1,), dtype=np.float32)
        self._calc_shape = self._determine_shape()
        high, low = self._datums_minmax()
        self.observation_space = spaces.Box(low, high, self._obs_shape(), dtype=np.float32)
        self._observation = None

        self._portfolio = Portfolio(cash, self.num_assets + 1)

    def _datums_minmax(self):
        low = min(collapse(self._datums))
        high = max(collapse(self._datums))
        return high, low

    def _determine_shape(self):
        values = first(self._datums).shape[1]
        return values, self.num_assets, self.window_size

    def _obs_shape(self):
        return (first(self._calc_shape),) + tuple(d for d in self._calc_shape[1:] if d > 1)

    @property
    def window_size(self):
        return self._window_size

    @property
    def num_assets(self):
        return len(self._datums)

    @property
    def portfolio(self):
        return self._portfolio

    def reset(self):
        self._portfolio.reset()
        self._datums_iters = [windowed(d, self.window_size) for d in self._datums]
        self._move_to_next_datum()
        obs, done = self._move_to_next_datum()
        if done:
            raise DatumsError("Not enough data in the time series to create a single step.")
        return obs

    def _move_to_next_datum(self):
        prv_obs = self._observation
        done = False
        try:
            raw = self._read_next_obs()
            self._portfolio.update(raw)
            self._observation = self._transformer(raw, self._read_next_obs)
            self._observation = self._shape_to_observation(self._observation)
        except StopIteration:
            done = True
        return prv_obs, done

    def _read_next_obs(self):
        obs = np.empty(shape=self._calc_shape)
        for asset, it in enumerate(self._datums_iters):
            obs[:, asset, :] = np.array(next(it)).transpose()

        if not (obs > 0).all() or not np.isfinite(obs).all():
            raise DatumsError(f'Encountered zero, NaN or inf values in observation data: {obs}')

        return obs

    def _shape_to_observation(self, obs):
        obs = obs.squeeze()
        if self._calc_shape[0] == 1:
            obs = np.expand_dims(obs, axis=0)
        return obs

    def step(self, action):
        self._portfolio.shift(action)
        reward = self._portfolio.normalized_value()
        obs, done = self._move_to_next_datum()
        return obs, reward, done, None

    def render(self, mode='human'):
        pass

    def close(self):
        pass


class DatumsError(ValueError):
    pass
