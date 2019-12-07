import numpy as np
from gym import spaces, Env
from gym.error import ResetNeeded
from more_itertools import first, collapse, windowed


def normalize(vector):
    return vector / np.linalg.norm(vector, 1)


class Portfolio:
    cash_index = 0

    def __init__(self, cash, commission, size=None, distribution=None):
        self._cash = cash
        self._commission = commission
        if size:
            self._assets = np.zeros(size)
            self._assets[self.cash_index] = self._cash
        else:
            self._assets = distribution * self._cash
            size = len(self._assets)
        self._init_assets = self.assets
        self._action_mat = np.identity(size)
        self._prices = np.empty(size)
        self._prices[self.cash_index] = 1

    @property
    def assets(self):
        return self._assets

    def reset(self):
        self._assets = self._init_assets

    def update(self, observation):
        self._prices[1:] = observation[0, :, -1]

    def shift(self, action):
        m = self._make_price_shift_mat(action)
        new_a = np.matmul(m, self.assets)
        diff = new_a - self._assets
        diff[diff < 0] = 0
        self._assets = new_a - diff * self._commission

    def _make_price_shift_mat(self, action):
        p = np.expand_dims(self._prices, 0)
        p_mat = p * (1 / p).T
        shift = np.matmul((self._action_mat * action), p_mat)
        return shift

    def normalized_value(self):
        return np.matmul(self._prices, self.assets) / self._cash


class FixedPortfolio:
    def __init__(self, datums):
        self._datums = datums
        self._datums_iter = None
        self._current = None
        self._cash = None

    def reset(self):
        self._datums_iter = iter(self._datums)

    def update(self, _):
        self._current = next(self._datums_iter)
        if self._cash is None:
            self._cash = self._current

    def normalized_value(self):
        return self._current / self._cash


def make_buy_and_hold(action_space):
    dist = np.ones(action_space.shape)
    dist[0] = 0
    return normalize(dist)


class PortfolioEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, datums=None, window_size=1, cash=1, commission=0, baseline=None):
        self._datums = datums
        self._datums_iters = None
        self._window_size = window_size
        self.action_space = spaces.Box(-1, 1, (self.num_assets + 1,), dtype=np.float32)
        self._calc_shape = self._determine_shape()
        high, low = self._datums_minmax()
        self.observation_space = spaces.Box(low, high, self.obs_shape(), dtype=np.float32)
        self._portfolio = Portfolio(cash, commission, size=self.num_assets + 1)
        if baseline is None:
            self._baseline = Portfolio(cash, 0, distribution=make_buy_and_hold(self.action_space))
        else:
            self._baseline = FixedPortfolio(baseline)
        self._observation = None
        self._done = False

    def _datums_minmax(self):
        low = min(collapse(self._datums))
        high = max(collapse(self._datums))
        return high, low

    def _determine_shape(self):
        values = first(self._datums).shape[1]
        return values, self.num_assets, self.window_size

    def obs_shape(self):
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

    @property
    def datums(self):
        return self._datums

    def reset(self):
        self._portfolio.reset()
        self._baseline.reset()
        self._datums_iters = [windowed(d, self.window_size) for d in self._datums]
        self._move_to_next_datum()
        obs = self._move_to_next_datum()
        if self._done:
            raise DatumsError("Not enough data in the time series to create a single step.")
        return obs

    def _move_to_next_datum(self):
        prv_obs = self._observation
        self._done = False
        try:
            raw = self._read_next_obs()
            self._portfolio.update(raw)
            self._baseline.update(raw)
            self._observation = self._shape_to_observation(raw)
        except StopIteration:
            self._done = True
        return prv_obs

    def skip_state(self):
        return self._move_to_next_datum()

    def _read_next_obs(self):
        obs = np.empty(shape=self._calc_shape)
        for asset, it in enumerate(self._datums_iters):
            obs[:, asset, :] = np.array(next(it)).transpose()

        if not (obs > 0).all() or not np.isfinite(obs).all():
            raise DatumsError(f'Encountered zero, NaN or inf values in observation data: {obs}')

        return obs

    def _shape_to_observation(self, obs):
        obs = obs.squeeze()
        if self._calc_shape[0] == 1:  # force observations to arrays because most ANNs don't use scalars directly
            obs = np.expand_dims(obs, axis=0)
        return obs

    def step(self, action):
        if self._done:
            raise PortfolioResetNeeded("Stepping past the end of the time series")

        self._portfolio.shift(action)
        reward = self._portfolio.normalized_value()
        baseline = self._baseline.normalized_value()
        obs = self._move_to_next_datum()
        return obs, reward, self._done, {'baseline': baseline}

    def render(self, mode='human'):
        pass

    def close(self):
        pass


class PortfolioResetNeeded(ResetNeeded):
    pass


class DatumsError(ValueError):
    pass
