"""
One network link environment.
Link has changing base load.
Actions: start 0 to 4 more transfers
Reward: percentage of free rate used. Gets negative if link fully saturated
Files sizes are normally distributed (absolute values).
"""

import math
from collections import deque

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.classic_control import rendering

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
        self.current_base_rate = int(self.max_link_rate * 0.5 * np.random.ranf())
        self.tstep = 0
        self.ntransfers = 0
        self.viewer = None
        self.h_base = deque(maxlen=600)
        self.h_added = deque(maxlen=600)
        self._seed()

        # obesrvation space reports only on files transfered: rate and how many steps ago it started.
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0]),
            high=np.array([np.finfo(np.float32).max, np.iinfo(np.int32).max])
        )
        self.action_space = spaces.Discrete(4)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):

        # add transfers if asked for
        for i in range(action):
            file_size = int(math.fabs(self.file_size_mean + np.random.standard_normal() * self.file_size_sigma))
            self.transfers[self.ntransfers] = [self.tstep, file_size, 0]
            self.ntransfers += 1

        # find current base rate
        self.current_base_rate += int(np.random.standard_normal() * 8 * 1024 * 1024)
        if self.current_base_rate > self.base_rate_max:
            self.current_base_rate = self.base_rate_max
        if self.current_base_rate < self.base_rate_min:
            self.current_base_rate = self.base_rate_min

        # find used rate if all the ongoing transfers would be at maximal rate
        active_transfers = 0
        for t in self.transfers.values():
            # print(t)
            if self.tstep < self.handshake_duration + t[0]:
                continue
            active_transfers += 1

        max_rate = self.max_rate_per_file * active_transfers

        # find free bandwidth
        max_free_bandwidth = self.max_link_rate - self.current_base_rate

        reward = max_rate / max_free_bandwidth * 100
        if reward > 100:
            reward = -reward + 100

        episode_over = False
        if (max_rate + self.current_base_rate) > 1.1 * self.max_link_rate:
            episode_over = True

        current_rate_per_file = 0
        if active_transfers > 0:
            current_rate_per_file = min(math.floor(max_free_bandwidth / active_transfers), self.max_rate_per_file)

        time_of_last_started_finished_transfer = 0
        rate_of_last_started_finished_transfer = 0
        finished = 0
        for k, t in self.transfers.items():
            if self.tstep < self.handshake_duration + t[0]:
                continue
            t[2] += current_rate_per_file
            if t[2] > t[1]:  # it is finished
                finished += 1
                if t[0] > time_of_last_started_finished_transfer:  # last started from all finished
                    rate_of_last_started_finished_transfer = t[1] / (self.tstep - t[0])
                    time_of_last_started_finished_transfer = t[0]

        observation = (rate_of_last_started_finished_transfer, time_of_last_started_finished_transfer)
        self.tstep += 1

        self.h_base.append(self.current_base_rate)
        self.h_added.append(max_rate + self.current_base_rate)

        return observation, reward, episode_over, {
            "finished transfers": finished,
            "active transfers": active_transfers,
            "base rate [%]": int(self.current_base_rate / self.max_link_rate * 10000) / 100
        }

    def _reset(self):
        self.tstep = 0
        self.transfers = {}
        self.ntransfers = 0
        return np.array((0, 0))

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 640
        screen_height = 480

        scale = np.max(self.h_added) / 440

        bdata = []  # (screen_width - 20, 20)]  # first point in lower right corner
        y = list(reversed(self.h_base))
        for j, i in enumerate(y):
            bdata.append((screen_width - 20 - j, 20 + int(i / scale)))
        # bdata.append((screen_width - 20 - len(y), 20))

        adata = []  # (screen_width - 20, 20)]
        y = list(reversed(self.h_added))
        for j, i in enumerate(y):
            adata.append((screen_width - 20 - j, 20 + int(i / scale)))
        # adata.append((screen_width - 20 - len(y), 20))
        adata = adata[:self.tstep]
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
        #     l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        #     axleoffset = cartheight / 4.0
        #     cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        #     self.carttrans = rendering.Transform()
        #     cart.add_attr(self.carttrans)
        #     self.viewer.add_geom(cart)
        #     self.poletrans = rendering.Transform(translation=(0, axleoffset))
        #     pole.add_attr(self.poletrans)
        #     pole.add_attr(self.carttrans)
        #     self.axle = rendering.make_circle(polewidth / 2)
        #     self.axle.add_attr(self.poletrans)
        #     self.axle.add_attr(self.carttrans)
            self.xaxis = rendering.Line((20, 20), (screen_width - 20, 20))
            self.xaxis.set_color(0, 0, 0)
            self.yaxis = rendering.Line((20, 20), (20, screen_height - 20))
            self.yaxis.set_color(0, 0, 0)
            self.viewer.add_geom(self.xaxis)
            self.viewer.add_geom(self.yaxis)

        adde = rendering.PolyLine(adata, False)
        adde.set_color(.1, .6, .8)
        self.viewer.add_onetime(adde)

        base = rendering.PolyLine(bdata, False)
        base.set_color(.8, .6, .4)
        self.viewer.add_onetime(base)

        max_line = self.max_link_rate / scale
        ml = rendering.Line((20, max_line + 20), (screen_width - 20, max_line + 20))
        ml.set_color(0.1, 0.9, .1)
        self.viewer.add_onetime(ml)

        # if self.state is None:
        # return None

        # x = self.state
        # cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        # self.carttrans.set_translation(cartx, carty)
        # self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
