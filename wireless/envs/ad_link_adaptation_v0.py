"""
© 2020, University of Padova, Department of Information Engineering, SIGNET Lab.
"""
import random
from math import floor, ceil
import numpy as np
from gym import spaces, Env
import pandas as pd
from wireless.utils import misc

# Load BER-vs-SNR curves


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
        self.initial_timestep = 0  # Initial timestep of an observation
        self.current_timestep = 0  # Current timestep of an observation
        self.scenario = pd.DataFrame()  # DataFrame containing ['t', 'SINR']
        self.observation_duration = 1  # Observation duration [s]

        self.seed()  # Seed the environment
        self.reset()  # Reset the environment

    def _next_observation(self):
        # Get an observation from the environment
        # An observation could be an history of the last n SNR values at the receiver
        # An observation could be a mix between past SNR values and Success/Not Success (ACK/NACK)
        # The values of an observation should be scaled between 0-1
        obs = []
        self.current_timestep += 1
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
        """
        Reset the state of the environment to an initial state
        """
        # Random choice of a particular scenario
        filepath = "../../scenarios/lroom.csv"
        self.scenario = misc.import_scenario(filepath)

        # Pick up a random new starting point in the SNR-vs-Time trace
        self.initial_timestep = self._get_initial_timestep()
        self.current_timestep = self.initial_timestep

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
        reward = self._calculate_reward()

        done = self._is_done()

        obs = self._next_observation()  # even if (done)?

        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass

    def seed(self, seed=0):
        # Seed each component for results replication
        random.seed(seed)
        np.random.seed(seed)
        self._seed = seed

    def _get_initial_timestep(self):
        """
        Pick up a random new starting point in the SNR-vs-Time trace
        """
        # Assuming the scenario is ordered in time
        simulation_duration = self.scenario['t'].iloc[-1]
        leftover_duration = simulation_duration - self.observation_duration
        assert leftover_duration >= 0, f'The observation duration ({self.observation_duration} s) is longer than the ' \
                                       f'scenario duration ({simulation_duration} s)'

        max_init_timestep = np.count_nonzero(self.scenario['t'] <= leftover_duration)
        initial_timestep = random.randrange(0, max_init_timestep)  # last excluded

        return initial_timestep

    def _is_done(self):
        """
        Check if the observation is over
        """
        initial_time = self.scenario['t'].iloc[self.initial_timestep]
        current_time = self.scenario['t'].iloc[self.current_timestep]
        elapsed_time = current_time - initial_time

        done = elapsed_time >= self.observation_duration
        if not done:
            assert self.current_timestep <= len(self.scenario), f'Current timestep ({self.current_timestep}) over' \
                                                                f' scenario length ({len(self.scenario)})'

        return done
