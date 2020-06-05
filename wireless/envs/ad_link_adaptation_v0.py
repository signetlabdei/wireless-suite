"""
© 2020, University of Padova, Department of Information Engineering, SIGNET Lab.
"""
import random
from math import floor, ceil
import numpy as np
from gym import spaces, Env
import pandas as pd
from wireless.utils import misc
from wireless.utils.dmg_error_model import DmgErrorModel


class AdLinkAdaptationV0(Env):
    """An OpenAIGym-based environment to simulate link adaptation in IEEE 802.11ad"""
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, scenarios_list, obs_duration, n_stas=1, n_mcs=13, dmg_path="../../dmg_files/"):
        super().__init__()
        self._seed = None
        self.scenarios_list = scenarios_list  # List with the scenarios to be randomly picked up in the env
        self.n_stas = n_stas  # Number of STAs
        self.n_mcs = n_mcs  # By default load curves for DMG Control and SC MCSs
        self.dmg_path = dmg_path  # Path to the folder containing DMG files

        # Define: Observation space and Action space
        self.observation_space = spaces.Box(low=-50, high=80, shape=(1, 0), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_mcs)

        # Internal variables
        self.network_timestep = 0.005  # The current network timestep is 5 ms
        self.msdu_size = 7935*8  # Max MSDU aggregation size in bits
        self.tx_pkts_list = None  # List containing the size of the packets to tx
        self.rnd_list = None  # List of random values between [0,1)
        self.psr_list = None  # List of packet success rates
        self.succ_list = None  # List with the status for each txed packet
        self.initial_timestep = None  # Initial timestep of an observation
        self.current_timestep = None  # Current timestep of an observation
        self.scenario = pd.DataFrame()  # DataFrame containing ['t', 'SINR']
        self.observation_duration = obs_duration  # Observation duration [s]
        self.error_model = DmgErrorModel(self.dmg_path + "/error_model/LookupTable_1458.txt",
                                         self.n_mcs)  # Create DMG error model

        self.seed()  # Seed the environment
        self.reset()  # Reset the environment

    def _get_observation(self):
        # Get an observation from the environment
        # An observation could be an history of the last n SNR values at the receiver
        # An observation could be a mix between past SNR values and Success/Not Success (ACK/NACK)
        # The values of an observation should be scaled between 0-1

        # Return the current SNR value
        obs = self.scenario['SINR'].iloc[self.current_timestep]
        self.current_timestep += 1
        return obs

    def _take_action(self, action):
        # Execute action
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"

        # Create packets based on the current MCS (i.e. the action taken)
        current_snr = self.scenario['SINR'].iloc[self.current_timestep]
        mcs_rate = misc.get_mcs_data_rate(action)
        assert mcs_rate is not None, f"{action} is not a valid MCS or the format is wrong"
        data_rate = int(mcs_rate * self.network_timestep)
        n_packets = data_rate // self.msdu_size
        self.tx_pkts_list = [self.msdu_size] * n_packets
        # Compute the success rate of the current packet based on: current SNR, selected MCS, BER-SNR curves.
        self.psr_list = [self.error_model.get_packet_success_rate(current_snr, action, self.msdu_size)] * n_packets
        last_pkt = data_rate % self.msdu_size
        if last_pkt != 0:
            self.tx_pkts_list.append(last_pkt)
            self.psr_list.append(self.error_model.get_packet_success_rate(current_snr, action, last_pkt))
        self.rnd_list = np.random.rand(len(self.psr_list),)
        self.succ_list = self.rnd_list <= self.psr_list

    def _calculate_reward(self):
        # Compute the reward associated with the action taken
        # The reward could be the spectral efficiency if the packet is successfully transmitted
        # Whereas, it should be a negative value (e.g. -1) if the packet is NOT successfully transmitted
        # Define specific reward function

        # Return the number of bits successfully transmitted
        succ_pkts = np.array(self.tx_pkts_list)[np.array(self.succ_list)]
        return np.sum(succ_pkts)

    def reset(self):
        """
        Reset the state of the environment to an initial state
        """
        # Random choice of a particular scenario
        scenario = random.choice(self.scenarios_list)
        filepath = self.dmg_path + "/qd_scenarios/" + scenario
        self.scenario = misc.import_scenario(filepath)

        # Pick up a random new starting point in the SNR-vs-Time trace
        self.initial_timestep = self._get_initial_timestep()
        self.current_timestep = self.initial_timestep
        # Define the length of the SNR-vs-Time chunk to consider (number of steps in the episode)
        # Try different size of this chunk

        # TEMPORARILY consider the entire trace
        self.initial_timestep = 0
        self.current_timestep = 0

        return self._get_observation()

    def step(self, action):
        """
        Execute one time step within the environment
        """
        self._take_action(action)

        # TBD: How to generate a packet? What's our notion of packet?
        # Consider full buffer transmission?
        # Consider CBR or VBR traffic?
        reward = self._calculate_reward()
        done = self._is_done()
        obs = self._get_observation()  # even if (done)?

        return obs, reward, done, {"tx_pkts_list": self.tx_pkts_list, "rnd_list": self.rnd_list,
                                   "psr_list": self.psr_list, "succ_list": self.succ_list}

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
        Check if the observation duration is over
        """
        initial_time = self.scenario['t'].iloc[self.initial_timestep]
        current_time = self.scenario['t'].iloc[self.current_timestep]
        elapsed_time = current_time - initial_time

        done = elapsed_time >= self.observation_duration
        if not done:
            assert self.current_timestep <= len(self.scenario), f'Current timestep ({self.current_timestep}) over' \
                                                                f' scenario length ({len(self.scenario)})'
        return done
