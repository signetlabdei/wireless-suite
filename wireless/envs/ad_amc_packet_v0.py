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
    """
    An OpenAIGym-based environment to simulate link adaptation in IEEE 802.11ad.

    This environment is packet-based.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, campaign, net_timestep, scenarios_list=None, dmg_path="../../dmg_files/", obs_duration=1,
                 history_length=5, n_mcs=13, packet_size=7935 * 8):
        """
        Initialize the environment.

        Parameters
        ----------
        campaign : str
            The campaign name in the dmg_path.
        net_timestep : float
            The timestep of the scenario(s).
        scenarios_list : list of str
            Specific scenarios taken from the campaign.
        dmg_path : str
            The path to the folder containing the channel simulation campaigns.
        obs_duration : float
            The duration in [s] of an observation.
        history_length : int
            The length of the lists returned as observation. See step(self, action) for more information.
        n_mcs : int
            The number of MCSs to be used, going [0, n_mcs).
        packet_size : int
            Packet size in [b].
        """
        super().__init__()

        # Member variables
        self._dmg_path = dmg_path
        self._campaign = campaign
        self._scenarios_list = scenarios_list
        self._n_mcs = n_mcs  # the number of MCS considered [0, n_mcs)
        self._history_length = history_length  # the number of past steps visible as observations
        self._network_timestep = net_timestep  # The network timestep of the environment
        self._observation_duration = obs_duration  # temporal duration of an observation
        self._packet_size = packet_size  # The size of a packet. Default: max A-MSDU size in bits (7935 * 8 b)

        self._seed = None  # the seed for RNGs
        self._scenario_duration = None  # The duration of the current scenario
        self._initial_time = None  # Initial time of an observation
        self._current_time = None  # Current time of an observation
        self._scenario = pd.DataFrame()  # DataFrame containing ['t', 'SNR']

        self._error_model = DmgErrorModel(self._dmg_path + "/error_model/LookupTable_1458.txt",
                                          self._n_mcs)  # Create DMG error model
        self._qd_scenarios_path = os.path.join(self._dmg_path, "qd_scenarios", self._campaign)

        if self._scenarios_list is None:
            self._scenarios_list = [file for file in os.listdir(self._qd_scenarios_path) if file.endswith(".csv")]

        self._snr_history = [None] * self._history_length  # The history of observed SNRs
        self._pkt_succ_history = [None] * self._history_length  # The history of pkts success (1) or failure (0)
        self._mcs_history = [None] * self._history_length  # The history of chosen MCSs

        # Define: Observation space and Action space
        snr_space = spaces.Box(low=-50, high=80, shape=(self._history_length,), dtype=np.float32)
        pkt_succ_space = spaces.Discrete(2)
        mcs_space = spaces.Discrete(self._n_mcs)

        self._observation_space = spaces.Tuple((snr_space, pkt_succ_space, mcs_space))
        self._action_space = mcs_space

        self.seed()  # Seed the environment
        self.reset()  # Reset the environment

    # Get-only attributes
    @property
    def error_model(self):
        return self._error_model

    @property
    def network_timestep(self):
        return self._network_timestep

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    # Public methods
    def reset(self):
        # Random choice of a particular scenario
        scenario = random.choice(self._scenarios_list)
        self._import_scenario(scenario)

        # Pick up a random new starting point in the SNR-vs-Time trace
        self._initial_time = random.uniform(0, self._scenario_duration - self._observation_duration)
        self._current_time = self._initial_time

        return self._get_observation()

    def initialize(self):
        """
        Initialize the environment's history.

        For the moment, a number of packets equal to the history length are sent with minimum MCS.
        """
        for i in range(self._history_length):
            self.step(0)

    def step(self, action):
        """
        OpenAI Gym step method.

        Parameters
        ----------
        action : int
            The action to be taken.

        Returns
        -------
        observation : tuple
            The observation. See _get_observation.
        reward : float
        done: bool
        info: dict
        """
        self._take_action(action)

        # update internal state
        observation = self._get_observation()
        reward = self._calculate_reward()
        done = self._is_done()
        info = {}
        return observation, reward, done, info

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass

    def seed(self, seed=0):
        # Seed each component for results replication
        random.seed(seed)
        np.random.seed(seed)
        self._seed = seed

    # Private methods
    def _import_scenario(self, scenario):
        """
        Import the scenario, setting all the related attributes.

        Parameters
        ----------
        scenario : str
        """
        filepath = os.path.join(self._qd_scenarios_path, scenario)
        self._scenario = misc.import_scenario(filepath)
        # The 't' column indicates the start of each timestep
        self._scenario_duration = self._scenario['t'].iloc[-1] + self._network_timestep

        assert self._scenario_duration >= self._observation_duration, "Observation duration should be less than the " \
                                                                      "scenario duration "

    def _get_observation(self):
        """
        Get the current observation.

        Assumes the latest event to be in position [0].

        Returns
        -------
        snr_history : list of float
        pkts_success_history : list of int
            List of [0,1] values, where 1 indicates a successful transmission, and 0 a failed transmission.
        mcs_history : list of int
        """
        return self._snr_history, self._pkt_succ_history, self._mcs_history

    def _take_action(self, mcs):
        """
        Take the action, sending a packet with the given MCS.

        Also update the internal state and history.

        Parameters
        ----------
        mcs : int
            The MCS in the action space used to send the packet.
        """
        assert self.action_space.contains(mcs), f"{mcs} ({type(mcs)}) invalid"

        # Get current SNR
        timestep = misc.get_timestep(self._current_time, self._network_timestep)
        current_snr = self._scenario['SNR'].iloc[timestep]

        # Check packet success
        psr = self._error_model.get_packet_success_rate(current_snr, mcs, self._packet_size)
        success = random.random() <= psr

        # Roll results: [0] regards the most recent packet
        self._snr_history = [current_snr] + self._snr_history[:-1]
        self._pkt_succ_history = [int(success)] + self._pkt_succ_history[:-1]
        self._mcs_history = [mcs] + self._mcs_history[:-1]

        # Update current time for next packet
        self._current_time += misc.get_packet_duration(self._packet_size, mcs)

    def _calculate_reward(self):
        """
        Compute the reward based on the latest packet sent.

        The reward is equal to the packet size in bits, if the packet was successfully transmitted, or 0 if the packet
        failed to be transmitted.

        Returns
        -------
        reward: float
        """
        if self._pkt_succ_history[0] == 1:
            return self._packet_size
        elif self._pkt_succ_history[0] == 0:
            return 0
        else:
            raise ValueError(f"Unexpected value for _succ_pkts_history: {self._pkt_succ_history}")

    def _is_done(self):
        """
        Check if the observation ended.

        Returns
        -------
        is_done : bool
        """
        return self._current_time - self._initial_time >= self._observation_duration


# Simple test
if __name__ == "__main__":
    env = AdAmcPacketV0("scenarios_v2", 5e-3)
    env.initialize()
    env.step(1)
