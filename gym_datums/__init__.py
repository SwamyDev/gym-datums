from gym.envs.registration import register
from gym_datums._version import __version__

name = "gym_datums"

register(
    id='portfolio-v0',
    entry_point='gym_datums.envs:PortfolioEnv',
)
