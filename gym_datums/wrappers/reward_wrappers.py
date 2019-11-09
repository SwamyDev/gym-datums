import numpy as np
from gym import RewardWrapper, Wrapper


class RelativeReward(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._prv_reward = 1

    def reward(self, reward):
        r = reward / self._prv_reward
        self._prv_reward = reward
        return r

    def reset(self, **kwargs):
        self._prv_reward = 1
        return super().reset(**kwargs)


class OnlyFinalReward(Wrapper):
    def step(self, action):
        o, r, d, i = super().step(action)
        if not d:
            r = 0
        return o, r, d, i


class SharpRatioReward(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._mean = 0
        self._variance = 0
        self._n = 0

    def step(self, action):
        o, reward, d, info = super().step(action)
        excess = reward - info['baseline']
        self._increment_mean_and_variance(excess)
        if self._n == 1:
            return o, excess, d, info
        else:
            return o, self._mean / np.sqrt(self._variance), d, info

    def _increment_mean_and_variance(self, x):
        """
        v ... variance
        m ... mean
        v' = ((n - 2)/(n - 1)) * v + ((x - m)^2) / n
        m' = (x + (n - 1) * m) / n
        """
        self._n += 1
        var_term = 0 if self._n == 1 else ((self._n - 2) / (self._n - 1)) * self._variance
        self._variance = var_term + ((x - self._mean) ** 2) / self._n
        self._mean = (x + (self._n - 1) * self._mean) / self._n
