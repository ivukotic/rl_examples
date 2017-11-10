import gym
from gym import error, spaces, utils
from gym.utils import seeding

class LinkEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.observation_space = spaces.Box(low=0, high=1, shape=(1))

        self.action_space = spaces.Tuple((spaces.Discrete(3),
                                          spaces.Box(low=0, high=100, shape=1),
                                          spaces.Box(low=-180, high=180, shape=1),
                                          spaces.Box(low=-180, high=180, shape=1),
                                          spaces.Box(low=0, high=100, shape=1),
                                          spaces.Box(low=-180, high=180, shape=1)))

    def _step(self, action):
        reward = 1
        observation = 0.5
        episode_over = True #self.status != hfo_py.IN_GAME
        return observation, reward, episode_over, {}

    def _reset(self):
        ...

    def _render(self, mode='human', close=False):
        pass
