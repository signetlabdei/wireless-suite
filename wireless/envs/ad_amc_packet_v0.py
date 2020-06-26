"""
© 2020, University of Padova, Department of Information Engineering, SIGNET Lab.
"""
import os
import random
import numpy as np
from gym import spaces, Env
import pandas as pd
from wireless.utils import misc
from wireless.utils.dmg_error_model import DmgErrorModel


class AdAmcPacketV0(Env):
    """An OpenAIGym-based environment to simulate link adaptation in IEEE 802.11ad"""
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        super().__init__()

        # Define action and observation space
        # They must be gym.spaces objects

        # Example when using discrete actions:
        self.action_space = spaces.Discrete(0)

        # Example for using image as input:
        self.observation_space = spaces.Discrete(0)

    def _next_observation(self):
        return []

    def _take_action(self, action):
        return

    def _calculate_reward(self):
        return 0

    def reset(self):
        return self._next_observation()

    def step(self, action):
        self._take_action(action)

        # update internal state

        observation = self._next_observation()
        reward = self._calculate_reward()
        done = self._is_done()
        info = self._get_info()
        return observation, reward, done, info

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass

    def seed(self, seed=0):
        return

    def _get_initial_timestep(self):
        return []

    def _is_done(self):
        return []

    def _get_info(self):
        return {}
