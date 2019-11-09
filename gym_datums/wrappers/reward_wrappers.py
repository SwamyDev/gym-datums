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
