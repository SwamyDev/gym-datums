import numpy as np
from gym import ObservationWrapper, spaces
from more_itertools import collapse, stagger


class LogReturnsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._previous = None
        high, low = self._wrapped_minmax()
        self.observation_space = spaces.Box(low, high, self.obs_shape(), dtype=np.float32)

    def _wrapped_minmax(self):
        returns = [np.log(nxt / prv) for prv, nxt in stagger(collapse(self.datums), offsets=(0, 1))]
        low = min(returns)
        high = max(returns)
        return high, low

    def observation(self, observation):
        if self._previous is None:
            self._previous = observation
            observation = self.skip_state()

        rt = np.log(observation / self._previous)
        self._previous = observation
        return rt
