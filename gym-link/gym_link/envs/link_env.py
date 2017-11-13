"""
One network link environment. 
Link has changing base load.
Actions: start 0 to 4 more transfers
Reward: percentage of free rate used. Gets negative if link fully saturated   
Files sizes are normally distributed (absolute values).
"""

import math
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np


class LinkEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.max_link_rate = 10 * 1024 * 1024 * 1024 / 8  # 10 Gigabits - all rates are in B/s
        self.base_rate_min = 0
        self.base_rate_max = self.max_link_rate * 0.9
        self.handshake_duration = 7  # seconds
        self.max_rate_per_file = 25 * 1024 * 1024  # B/s
        self.file_size_mean = 150 * 1024 * 1024
        self.file_size_sigma = 200 * 1024 * 1024

        self.dt = 1  # seconds
        #  key: int, start: int, size: int [bytes], transfered: int[bytes]
        self.transfers = {}
        self.current_base_rate = self.max_link_rate * 0.5 * np.random.ranf()
        self.step = 0
        self.ntransfers = 0
        self.state = None
        self.viewer = None
        self._seed()

        # obesrvation space reports only on files transfered: rate and how many steps ago it started.
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([np.finfo(np.float32).max, np.iinfo(np.int32).max]),
            shape=(1)
        )
        self.action_space = spaces.Discrete(4)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):

        # add transfers if asked for
        for i in action:
            file_size = int(math.fabs(self.file_size_mean + np.random.standard_normal() * self.file_size_sigma))
            self.transfers[self.ntransfers] = [self.step, file_size, 0]

        # find current base rate
            self.current_base_rate += int(np.random.standard_normal(0, 8 * 1024 * 1024))
            if self.current_base_rate > self.base_rate_max:
                self.current_base_rate = self.base_rate_max
            if self.current_base_rate < self.base_rate_min:
                self.current_base_rate = self.base_rate_min

        # find used rate if all the ongoing transfers would be at maximal rate
        active_transfers = 0
        for t in self.transfers.values():
            if self.step < self.handshake_duration + t[0]:
                continue
            active_transfers += 1

        max_rate = self.max_rate_per_file * active_transfers

        # find free bandwidth
        max_free_bandwidth = self.max_link_rate - self.current_base_rate

        reward = max_rate / max_free_bandwidth * 100
        if reward > 100:
            reward = -reward

        if reward < -100:
            episode_over = True
        else:
            episode_over = False

        current_rate_per_file = math.floor(max_free_bandwidth / active_transfers)

        time_of_last_started_finished_transfer = 0
        rate_of_last_started_finished_transfer = 0
        for k, t in self.transfers.items():
            if self.step < self.handshake_duration + t[0]:
                continue
            t[2] += current_rate_per_file
            if t[2] > t[1]:  # it is finished
                if t[0] > time_of_last_started_finished_transfer:  # last started from all finished
                    rate_of_last_started_finished_transfer = t[1] / (self.step - t[0])
                print("transfer ", k, "finished")

        observation = (rate_of_last_started_finished_transfer, time_of_last_started_finished_transfer)

        return observation, reward, episode_over, {}

    def _reset(self):
        ...

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        # world_width = self.x_threshold * 2
        # scale = screen_width / world_width
        # carty = 100  # TOP OF CART
        # polewidth = 10.0
        # polelen = scale * 1.0
        # cartwidth = 50.0
        # cartheight = 30.0

        # if self.viewer is None:
        #     from gym.envs.classic_control import rendering
        #     self.viewer = rendering.Viewer(screen_width, screen_height)
        #     l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        #     axleoffset = cartheight / 4.0
        #     cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        #     self.carttrans = rendering.Transform()
        #     cart.add_attr(self.carttrans)
        #     self.viewer.add_geom(cart)
        #     l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        #     pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        #     pole.set_color(.8, .6, .4)
        #     self.poletrans = rendering.Transform(translation=(0, axleoffset))
        #     pole.add_attr(self.poletrans)
        #     pole.add_attr(self.carttrans)
        #     self.viewer.add_geom(pole)
        #     self.axle = rendering.make_circle(polewidth / 2)
        #     self.axle.add_attr(self.poletrans)
        #     self.axle.add_attr(self.carttrans)
        #     self.axle.set_color(.5, .5, .8)
        #     self.viewer.add_geom(self.axle)
        #     self.track = rendering.Line((0, carty), (screen_width, carty))
        #     self.track.set_color(0, 0, 0)
        #     self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        # x = self.state
        # cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        # self.carttrans.set_translation(cartx, carty)
        # self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
