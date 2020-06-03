"""
© 2020, University of Padova, Department of Information Engineering, SIGNET Lab.
"""
import random
from math import floor, ceil
import numpy as np
from gym import spaces, Env

# Load data from SNR-vs-Time trace
# Load BER-vs-SNR curves

MAX_STEPS = 10  # Max number of time steps for the environment


class AdLinkAdaptationV0(Env):
    """An OpenAIGym-based environment to simulate link adaptation in IEEE 802.11ad"""
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, n_stas=1):
        super().__init__()
        self._seed = None
        self.n_stas = n_stas  # Number of STAs

        # Define: Observation space and Action space

        # Variables of the observation vector

        # Internal variables
        self.t = 0  # Time step

        self.seed()  # Seed the environment
        self.reset()  # Reset the environment

    def _next_observation(self):
        # Get an observation from the environment
        # An observation could be an history of the last n SNR values at the receiver
        # An observation could be a mix between past SNR values and Success/Not Success (ACK/NACK)
        # The values of an observation should be scaled between 0-1
        obs = []
        return obs

    def _take_action(self, action):
        # Execute action
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"
        # Compute the success rate of the current packet based on: current SNR, selected MCS, BER-SNR curves.

    def _calculate_reward(self):
        # Compute the reward associated with the action taken
        # The reward could be the spectral efficiency if the packet is successfully transmitted
        # Whereas, it should be a negative value (e.g. -1) if the packet is NOT successfully transmitted
        # Define specific reward function
        rwd = -1
        return rwd

    def reset(self):
        # Reset the state of the environment to an initial state
        self.t = 0
        # Pick up a random/deterministic new starting point in the SNR-vs-Time trace
        # Random choice of a particular scenario
        # Define the length of the SNR-vs-Time chunk to consider (number of steps in the episode)
        # Try different size of this chunk
        return self._next_observation()

    def step(self, action):
        # Execute one time step within the environment
        # In this environment one time step occurs every 5 ms of IEEE 802.11ad network time
        self._take_action(action)

        # TBD: How to generate a packet? What's our notion of packet?
        # Consider full buffer transmission?
        # Consider CBR or VBR traffic?

        self.t += 1  # Update time step
        reward = self._calculate_reward()
        done = bool(self.t >= MAX_STEPS)
        obs = self._next_observation()

        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass

    def seed(self, seed=0):
        # Seed each component for results replication
        random.seed(seed)
        np.random.seed(seed)
        self._seed = seed
