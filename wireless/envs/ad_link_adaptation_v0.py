        self.t += 1  # Update time step
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


class AdLinkAdaptationV0(Env):
    """An OpenAIGym-based environment to simulate link adaptation in IEEE 802.11ad"""
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, net_timestep, campaign, scenarios_list=None, obs_duration=None, snr_history=5,
                 n_mcs=13, dmg_path="../../dmg_files/"):
        """Initialize the IEEE 802.11ad link adaptation environment (v0).

        Parameters
        ----------
        net_timestep: Network timestep in [s].
        campaign: Folder containing a set of scenario with specific beamforming and antenna configurations.
        scenarios_list: List with the scenarios' filenames from dmg_path to be randomly picked up in the env. If None,
         import all files in the folder. Default: None.
        obs_duration: Duration in [s] of a single observation. If None, the obs_duration equals the length of the
         selected scenario. Default: None.
        snr_history: The number of SNR values in the past to consider in the state. Default: 5.
        n_mcs: Number of possible MCSs supported, starting from 0. Default: 13.
        dmg_path: Path to the folder containing DMG files. Default: "../../dmg_files/".
        """
        super().__init__()

        self.dmg_path = dmg_path
        self.campaign = campaign
        self.qd_scenarios_path = os.path.join(self.dmg_path, "qd_scenarios",  self.campaign)
        self.obs_duration = obs_duration

        if scenarios_list is None:
            scenarios_list = [file for file in os.listdir(self.qd_scenarios_path) if file.endswith(".csv")]
        self.scenarios_list = scenarios_list

        self._seed = None
        self.n_mcs = n_mcs
        self.snr_history = snr_history

        # Define: Observation space and Action space
        self.snr_space = spaces.Box(low=-50, high=80, shape=(self.snr_history,), dtype=np.float32)
        self.mcs_space = spaces.Discrete(self.n_mcs)
        self.observation_space = spaces.Dict({"snr": self.snr_space,
                                              "mcs": self.mcs_space})
        self.action_space = self.mcs_space

        # Internal variables
        self.done = None  # Flag to signal the end of the current episode
        self.network_timestep = net_timestep  # The network timestep of the environment
        self.amsdu_size = 7935 * 8  # Max MSDU aggregation size in bits (7935 bytes is the max A-MSDU size)
        self.mcs = None  # The current MCS to be used
        self.tx_pkts_list = None  # List containing the size of the packets to tx
        self.rnd_list = None  # List of random values between [0,1)
        self.psr_list = None  # List of packet success rates
        self.succ_list = None  # List with the status for each txed packet
        self.scenario_duration = None  # The duration of the current scenario
        self.initial_timestep = None  # Initial timestep of an observation
        self.current_timestep = None  # Current timestep of an observation
        self.scenario = pd.DataFrame()  # DataFrame containing ['t', 'SNR']
        self.current_obs_duration = None  # The current observation duration
        self.future_snr = None  # The future snr to return in debug information
        self.error_model = DmgErrorModel(self.dmg_path + "/error_model/LookupTable_1458.txt",
                                         self.n_mcs)  # Create DMG error model

        # self.seed()  # Seed the environment
        self.reset()  # Reset the environment

    def _get_observation(self):
        # Get an observation from the environment
        # An observation could be an history of the last n SNR values at the receiver
        # An observation could be a mix between past SNR values and Success/Not Success (ACK/NACK)
        # The values of an observation should be scaled between 0-1

        # Return history of SNR and the current MCS
        snr_list = self.scenario['SNR'].iloc[self.current_timestep - self.snr_history:self.current_timestep].tolist()
        # Get one-step-ahead SNR
        self.future_snr = None
        if not self.done:
            self.future_snr = self.scenario['SNR'].iloc[self.current_timestep]

        # n_succ_pkts = np.count_nonzero(self.succ_list == True)
        return {"snr": snr_list,
                "mcs": self.mcs}

    def _take_action(self, action):
        # Execute action
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"
        self.mcs = action
        # Create packets based on the current MCS (i.e. the action taken)
        current_snr = self.scenario['SNR'].iloc[self.current_timestep]
        n_packets, last_pkt, self.tx_pkts_list = misc.get_tx_pkt_size_list(self.mcs,
                                                                           self.network_timestep,
                                                                           self.amsdu_size)
        # Compute the success rate of each packet based on: current SNR, selected MCS, BER-SNR curves.
        self.psr_list = [self.error_model.get_packet_success_rate(current_snr, self.mcs, self.amsdu_size)] * n_packets
        if last_pkt != 0:
            self.psr_list.append(self.error_model.get_packet_success_rate(current_snr, self.mcs, last_pkt))

        self.rnd_list = np.random.rand(len(self.psr_list), )
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
        # Reset internal state variables
        self.done = False
        self.succ_list = []
        self.mcs = 0

        # Random choice of a particular scenario
        scenario = random.choice(self.scenarios_list)
        filepath = os.path.join(self.qd_scenarios_path, scenario)
        self.scenario = misc.import_scenario(filepath)
        self.scenario_duration = self.scenario['t'].iloc[-1]
        # Retrieve the observation length
        if self.obs_duration is None:
            self.current_obs_duration = self.scenario_duration
        else:
            self.current_obs_duration = self.obs_duration
        # Pick up a random new starting point in the SNR-vs-Time trace
        self.initial_timestep = self._get_initial_timestep()
        self.current_timestep = self.initial_timestep

        return self._get_observation()

    def step(self, action):
        """
        Execute one time step within the environment
        """

        self._take_action(action)
        reward = self._calculate_reward()
        self.done = self._is_done()

        self.current_timestep += 1  # Need to increment the timestep before getting the observation
        obs = self._get_observation()  # even if (done)?

        return obs, reward, self.done, {"tx_pkts_list": self.tx_pkts_list, "rnd_list": self.rnd_list,
                                        "psr_list": self.psr_list, "succ_list": self.succ_list,
                                        "future_snr": self.future_snr}

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
        leftover_duration = self.scenario_duration - self.current_obs_duration
        assert leftover_duration >= 0, f"The observation duration ({self.current_obs_duration} s) is longer than the " \
                                       f"scenario duration ({self.scenario_duration} s)"

        max_n_init_timestep = np.count_nonzero(self.scenario['t'] <= leftover_duration)
        if self.snr_history >= max_n_init_timestep:
            initial_timestep = self.snr_history
        else:
            initial_timestep = random.randrange(self.snr_history, max_n_init_timestep)  # last excluded

        return initial_timestep

    def _is_done(self):
        """
        Check if the observation duration is over
        """
        initial_time = self.scenario['t'].iloc[self.initial_timestep]
        current_time = self.scenario['t'].iloc[self.current_timestep]
        elapsed_time = current_time - initial_time
        done = elapsed_time >= self.current_obs_duration or current_time == self.scenario_duration
        if not done:
            assert self.current_timestep <= len(self.scenario), f"Current timestep ({self.current_timestep}) over" \
                                                                f" scenario length ({len(self.scenario)})"
        return done
