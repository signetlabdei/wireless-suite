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
from wireless.utils import dot11ad_constants


class AdAmcPacketV0(Env):
    """
    An OpenAIGym-based environment to simulate link adaptation in IEEE 802.11ad.

    This environment is packet-based.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, campaign, net_timestep, scenarios_list=None, dmg_path="../../dmg_files/", obs_duration=1,
                 n_mcs=13, packet_size=dot11ad_constants.maxAmsduSize, harq_retx=2, reward_type="rx_bits"):
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
            The duration in [s] of an observation. If None, it corresponds to the entire scenario duration.
        n_mcs : int
            The number of MCSs to be used, going [0, n_mcs).
        packet_size : int
            Packet size in [b].
        harq_retx : int
            The number of HARQ retransmission after the first transmission. This affects the maximum delay.
        reward_type : str
            Type of reward yielded by the environment. Possible choices are ["rx_bits", "negative_delay"].
        """
        super().__init__()

        # Member variables
        self._dmg_path = dmg_path
        self._campaign = campaign
        self._scenarios_list = scenarios_list
        self._n_mcs = n_mcs  # The number of MCS considered [0, n_mcs)
        self._network_timestep = net_timestep  # The network timestep of the environment
        self._observation_duration = obs_duration  # Temporal duration of an observation
        self._packet_size = packet_size  # The size of a packet. Default: max A-MSDU size in bits (7935 * 8 b)
        self._harq_retx = harq_retx  # The number of HARQ retransmission after the first transmission
        self._reward_type = reward_type  # The type of reward for the environment

        self._seed = None  # The seed for RNGs
        self._scenario_duration = None  # The duration of the current scenario
        self._initial_time = None  # Initial time of an observation
        self._scenario = pd.DataFrame()  # DataFrame containing ['t', 'SNR']

        self._current_time = None  # Current time of an observation [s]
        self._current_snr = None  # SNR at current time [dB]
        self._current_success = None  # Success of the current packet. Success (1), failure (0), or retransmission (2).
        self._current_retx = 0  # The number of retransmissions experienced by the current packet
        self._current_pkt_delay = 0  # The delay of the current packet [s]
        self._current_mcs = None  # The MCS used for the current packet

        self._error_model = DmgErrorModel(self._dmg_path + "/error_model/LookupTable_1458.txt",
                                          self._n_mcs)  # Create DMG error model
        self._qd_scenarios_path = os.path.join(self._dmg_path, "qd_scenarios", self._campaign)

        if self._scenarios_list is None:
            self._scenarios_list = [file for file in os.listdir(self._qd_scenarios_path) if file.endswith(".csv")]

        # Set the requested reward
        if reward_type == "rx_bits":
            self._get_reward = self._rx_bits_reward
        elif reward_type == "negative_delay":
            self._get_reward = self._negative_delay_reward
        else:
            raise ValueError(f"Reward type '{reward_type}' not recognized.")

        # Define: Observation space and Action space
        time_space = spaces.Box(low=0, high=np.inf,
                                shape=(1,),
                                dtype=np.float32)
        snr_space = spaces.Box(low=-np.inf, high=np.inf,
                               shape=(1,),
                               dtype=np.float32)
        pkt_succ_space = spaces.Discrete(3)
        pkt_retx_space = spaces.Discrete(self._harq_retx + 1)  # harq_retx=0 means just a single transmission
        pkt_delay_space = spaces.Box(low=0, high=np.inf,
                                     shape=(1,),
                                     dtype=np.float32)
        mcs_space = spaces.Discrete(self._n_mcs)

        self._observation_space = spaces.Dict({"time": time_space,
                                               "snr": snr_space,
                                               "pkt_succ": pkt_succ_space,
                                               "pkt_retx": pkt_retx_space,
                                               "pkt_delay": pkt_delay_space,
                                               "mcs": mcs_space})
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

    @property
    def scenario_duration(self):
        return self._scenario_duration

    # Public methods
    def reset(self):
        """
        Reset the environment.

        Since no packets have been sent yet, the current success and mcs are None.

        Returns
        -------
        observation : dict
            See _take_action
        info : dict
            See _get_info
        """
        # Reset variables
        self._current_success = None
        self._current_retx = 0
        self._current_pkt_delay = 0
        self._current_mcs = None

        # Random choice of a particular scenario
        scenario = random.choice(self._scenarios_list)
        self._import_scenario(scenario)

        # Pick up a random new starting point in the SNR-vs-Time trace
        self._initial_time = self._get_initial_time()
        self._current_time = self._initial_time

        self._current_snr = self._get_snr()

        # Create initial observation
        obs = {"time": self._current_time,
               "snr": self._current_snr,
               "pkt_succ": self._current_success,
               "pkt_retx": self._current_retx,
               "pkt_delay": self._current_pkt_delay,
               "mcs": self._current_mcs}

        return obs, self._get_info()  # TODO: can we also output the info?

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

        # update internal state
        observation = self._take_action(action)
        reward = self._get_reward()
        done = self._is_done()
        info = self._get_info()
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
    def _get_initial_time(self):
        """
        Get the initial time for the observation.

        Returns
        -------
        initial_time : float
        """
        if self._observation_duration is not None:
            return np.random.uniform(0, self._scenario_duration - self._observation_duration)
        else:
            return 0

    def _get_snr(self):
        timestep = misc.get_timestep(self._current_time, self._network_timestep)
        return self._scenario['SNR'].iloc[timestep]

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

        if self._observation_duration is not None:
            assert self._scenario_duration >= self._observation_duration, "Observation duration should be less than " \
                                                                          "the scenario duration "

    def _take_action(self, mcs):
        """
        Take the action, sending a packet with the given MCS.

        Also update the internal state and history.

        Parameters
        ----------
        mcs : int
            The MCS in the action space used to send the packet.

        Returns
        -------
        obs : dict
            "time" : float
                Time when the packet was sent [s]
            "snr" : float
                SNR observed for the sent packet [dB]
            "pkt_succ" : int
                List of [0,2] values, where 1 indicates a successful transmission, 0 a failed transmission,
                and 2 a retransmission.
            "pkt_retx" : int
                Packet retransmissions. Only retransmissions are accounted for, thus a packet which is
                successfully transmitted at the first try will show a pkt_succ=1 and pkt_retx=0.
                Packets only fail (pkt_succ=0) when pkt_retx == harq_retx.
            "pkt_delay" : float
                Packet delay [s]. The delay includes the packet transmission time since different MCSs will
                also affect the total packet delay. The delay is incremented at each retransmission.
            "mcs" : int
        """
        assert self.action_space.contains(mcs), f"{mcs} ({type(mcs)}) invalid"
        self._current_mcs = mcs

        # Get current SNR
        self._current_snr = self._get_snr()

        # Check packet success
        psr = self._error_model.get_packet_success_rate(self._current_snr, self._current_mcs, self._packet_size)
        success = np.random.rand() <= psr

        # Update retransmission counter
        if success:
            self._current_success = 1  # Packet successfully transmitted
        elif self._current_retx < self._harq_retx:
            self._current_success = 2  # Packet retransmitted
        else:
            self._current_success = 0  # Packet failed

        # Update current packet duration
        duration = dot11ad_constants.get_total_tx_time(self._packet_size, self._current_mcs)
        self._current_pkt_delay += duration

        # Observation for the action taken
        obs = {"time": self._current_time,
               "snr": self._current_snr,
               "pkt_succ": self._current_success,
               "pkt_retx": self._current_retx,
               "pkt_delay": self._current_pkt_delay,
               "mcs": self._current_mcs}

        # Update time for next packet transmission (back to back)
        self._current_time += duration

        # Update retransmission counter
        if self._current_success == 1 or self._current_success == 0:
            # Packet successfully transmitted or failed: reset
            self._current_retx = 0
            self._current_pkt_delay = 0

        elif self._current_success == 2:
            # Packet retransmitted
            self._current_retx += 1

        else:
            raise ValueError(f"success={self._current_success} not recognized")

        return obs

    def _rx_bits_reward(self):
        """
        Compute the reward based on the latest packet sent.

        The reward is equal to the packet size in bits, if the packet was successfully transmitted, or 0 if the packet
        failed to be transmitted.

        Returns
        -------
        reward : float
        """

        if self._current_success == 1:
            return self._packet_size
        elif self._current_success == 0 or self._current_success == 2:
            return 0
        else:
            raise ValueError(f"Unexpected value for _current_success: {self._current_success}")

    def _negative_delay_reward(self):
        """
        Compute the reward based on the latest packet sent.

        The reward is equal to the negative delay of the packet if it was successfully transmitted, or scaled by a
        factor alpha if the transmission failed.
        The gold of this reward is to minimize the packet delay while discouraging packet loss.

        Returns
        -------
        reward : float
        """

        assert self._current_pkt_delay is not None, "Last delay is None"

        if self._current_success == 1:
            # Packet sent successfully transmitted
            return -1 * self._current_pkt_delay
        elif self._current_success == 2:
            # Retransmissions are just transitory
            return 0
        elif self._current_success == 0:
            # Failed transmissions should be discouraged
            alpha = 2
            return -alpha * self._current_pkt_delay
        else:
            raise ValueError(f"Unexpected value for _current_success: {self._current_success}")

    def _is_done(self):
        """
        Check if the observation ended.

        Returns
        -------
        is_done : bool
        """
        if self._observation_duration is None:
            return self._current_time >= self._scenario_duration
        else:
            return self._current_time - self._initial_time >= self._observation_duration

    def _get_info(self):
        """
        Get debug information from the environment.

        Returns
        -------
        info : dict
        """
        info = {"pkt_size": self._packet_size,
                "current_time": self._current_time}
        return info


# Simple test
if __name__ == "__main__":
    env = AdAmcPacketV0("scenarios_v2", 5e-3)
    env.step(1)
