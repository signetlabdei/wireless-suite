"""
© 2020, University of Padova, Department of Information Engineering, SIGNET Lab.
"""
import numpy as np
import tensorflow as tf
from wireless.utils import misc
from wireless.utils.misc import get_mcs_data_rate
import pandas as pd
import math
from wireless.utils import misc, dot11ad_constants


class SimpleNN(tf.keras.Model):
    def __init__(self, n_states, hidden_units, n_actions):
        super(SimpleNN, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(n_states,))
        self.hidden_layers = []
        for n_units in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(n_units, activation="relu",
                                                            kernel_initializer="random_normal"))
        self.output_layer = tf.keras.layers.Dense(n_actions, activation="linear", kernel_initializer="random_normal")

    @tf.function
    def call(self, inputs):
        y = self.input_layer(inputs)
        for hidden_layer in self.hidden_layers:
            y = hidden_layer(y)
        return self.output_layer(y)


class ReplayMemory:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.experience = [None] * self.memory_size
        self.current_idx = 0
        self.current_size = 0

    def store(self, state, action, reward, new_state, done):
        self.experience[self.current_idx] = (state, action, reward, new_state, done)
        self.current_idx += 1
        self.current_size = min(self.current_size + 1, self.memory_size)
        if self.current_idx >= self.memory_size:
            self.current_idx -= self.memory_size

    def sample(self, batch_size):
        assert self.current_size >= batch_size, f"Buffer size {self.current_size} smaller than batch size {batch_size}"
        sample_idxs = np.random.randint(0, self.current_size, size=batch_size)
        states, actions, rewards, new_states, dones = ([None] * batch_size for _ in range(5))
        for i, sample_idx in enumerate(sample_idxs):
            exp = self.experience[sample_idx]
            states[i] = exp[0]
            actions[i] = exp[1]
            rewards[i] = exp[2]
            new_states[i] = exp[3]
            dones[i] = exp[4]

        return np.asarray(states), np.asarray(actions), np.asarray(rewards), np.asarray(new_states), np.asarray(dones)


class DQNAgent:
    def __init__(self, action_space, n_states, hidden_units, memory_size=2000,
                 min_experiences=32, target_update=50, discount=0.99, batch_size=32, lr=0.001):
        self.action_space = action_space
        self.n_states = n_states
        self.discount_factor = discount
        self.batch_size = batch_size
        self.min_experiences = min_experiences
        self.target_update = target_update
        self.optimizer = tf.optimizers.Adam(lr)
        self.loss_fn = tf.losses.MeanSquaredError()
        self.model = SimpleNN(self.n_states, hidden_units, self.action_space.n)
        self.model.compile(optimizer=self.optimizer, loss="mse")
        self.target_model = SimpleNN(self.n_states, hidden_units, self.action_space.n)
        self.memory = ReplayMemory(memory_size)

    def predict(self, inputs, target=False):
        if target:
            return self.target_model(np.atleast_2d(inputs.astype('float32')))
        return self.model(np.atleast_2d(inputs.astype('float32')))

    def train(self):
        if self.memory.current_size < self.min_experiences:
            return 0
        # Sample batch of experiences from memory buffer
        states, actions, rewards, new_states, dones = self.memory.sample(self.batch_size)
        '''next_values = np.max(self.predict(new_states, target=True), axis=1)
        target_values = np.where(dones, rewards, rewards + self.discount_factor * next_values)
        # print(f"Target values {target_values}")
        with tf.GradientTape() as tape:
            selected_values = tf.math.reduce_sum(self.predict(states) * tf.one_hot(actions, self.action_space.n),
                                                 axis=1)
            # print(f"selected values: {selected_values}")
            # loss = tf.math.reduce_mean(tf.math.square(target_values - selected_values))
            loss = self.loss_fn(selected_values, target_values)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))'''

        target_q = rewards + self.discount_factor * np.amax(self.predict(new_states, target=True), axis=1) * (1 - dones)
        # print(f"target q-values: {target_q}, shape: {target_q.shape}")
        target_f = np.array(self.predict(states))
        # print(f"target f: {target_f}, shape: {target_f.shape}")
        for i, ac in enumerate(actions):
            target_f[i][ac] = target_q[i]
        # print(f"new target f: {target_f}")
        loss = self.model.train_on_batch(states, target_f)
        return loss

    def act(self, state, epsilon):
        if np.random.rand() < epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.predict(state)[0])

    def add_experience(self, state, action, reward, new_state, done):
        self.memory.store(state, action, reward, new_state, done)

    def copy_weights(self):
        self.target_model.set_weights(self.model.get_weights())

    def seed(self, seed=0):
        self.action_space.seed(seed)
        np.random.seed(seed)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path, compile=False)


class ConstantRateAgent:
    """
    This agent uses the same MCS for the entire simulation
    (i.e. no adaptation is performed)
    """

    def __init__(self, mcs):
        self.mcs = mcs

    def act(self, state):
        return self.mcs


class TargetBerAgent:
    """
    This agent computes the MCS based on a target BER
    """

    def __init__(self, action_space, error_model, target_ber=1e-6):
        self.action_space = action_space
        self.error_model = error_model
        self.target_ber = target_ber

    def act(self, snr):
        bers = [self.error_model.get_ber(snr, mcs) for mcs in range(self.action_space.n) if
                self.action_space.contains(mcs)]
        target_mcs_list = np.where(np.array(bers) <= self.target_ber)[0]
        # return lowest MCS if the target mcs list is empty
        selected_mcs = 0
        if target_mcs_list.size != 0:
            selected_mcs = np.amax(target_mcs_list)
        return selected_mcs


class OptimalAgent:
    """
    This agent computes the best MCS based on the future SNR, i.e., the SNR at one timestep ahead with respect to the
    current timestep.
    The MCS is chosen to be the one that maximizes the total average number of received packets, hence its optimality.
    """

    def __init__(self, action_space, error_model, timestep, pkt_size):
        self.action_space = action_space
        self.error_model = error_model
        self.timestep = timestep
        self.pkt_size = pkt_size

    def act(self, snr):
        rx_bits = np.zeros((self.action_space.n,))
        for action_idx in range(self.action_space.n):
            n_packets, last_pkt, pkt_size_list = misc.get_tx_pkt_size_list(action_idx, self.timestep, self.pkt_size)
            pkt_psr = self.error_model.get_packet_success_rate(snr, action_idx, self.pkt_size)
            last_pkt_psr = self.error_model.get_packet_success_rate(snr, action_idx, last_pkt)

            avg_rx_bits = self.pkt_size * pkt_psr * n_packets + last_pkt * last_pkt_psr
            rx_bits[action_idx] = avg_rx_bits

        selected_mcs = np.argmax(rx_bits)
        return selected_mcs


class TabularAgent:
    """
        This agent perform the choice of the best MCS based on the Q-value associated to the current state.
        At any time the agent can return the full Q-table or the current best policy.
    """

    def __init__(self, obs_space_dim, action_space_dim, method, q_table=None,
                 epsilon=1.0, alpha=0.1, gamma=0.95, smart_init=False, eps_update='e-greedy', sigma=0.04):
        """
        Initialize the agent.

        Parameters:
            obs_space_dim (int): The size of the observation space.
            action_space_dim (int): The size of the action space.
            method (str): The method used to update the Q_values (available: sarsa, q_learning, exSarsa).
            q_table (numpy array): Q_table to update.
            epsilon (float): Initial value of epsilon.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            smart_init (bool): Flag for smart initialization of the Q_table (default: False).
            eps_update (str): The method used for the update of epsilon.
            sigma (float): parameter for VDBE-Boltzmann algorithm

        """
        self.obs_space_dim = obs_space_dim
        self.action_space_dim = action_space_dim
        self.method = method

        if q_table is None:
            if smart_init:
                mcs_list = np.arange(action_space_dim)
                action_q_values = np.array([-get_mcs_data_rate(mcs) / 1e6 * alpha for mcs in mcs_list])

                self.q_table = np.tile(action_q_values, (obs_space_dim, 1))
            else:
                self.q_table = np.zeros((obs_space_dim, action_space_dim))
        else:
            self.q_table = q_table

        self.eps_update = eps_update
        self.epsilon = epsilon
        self.epsilon_values = [self.epsilon]

        if self.eps_update == "vdbe":
            self.epsilons_state = np.repeat(self.epsilon, obs_space_dim)
            self.delta_vdbe = 1.0 / self.action_space_dim
            self.sigma = sigma
            self.old_q = None
            self.new_q = None
            self.epsilon_values = [[self.epsilon] for _ in range(self.obs_space_dim)]

        self.alpha = alpha
        self.gamma = gamma
        self.state = 0
        self.action = 0

    def set_state(self, state):
        self.state = state

    def set_action(self, action):
        self.action = action

    def get_qtable(self):
        return self.q_table

    def act(self, state):
        if self.eps_update == "vdbe":
            self.epsilon = self.epsilons_state[state]
        return misc.epsilon_greedy(self.q_table[state, :], self.epsilon)

    def train_step(self, next_state, next_action, reward):
        self.old_q = self.q_table[self.state, self.action]
        if self.method == "sarsa":
            self.q_table[self.state, self.action] += self.alpha * \
                                                     (reward + self.gamma * self.q_table[next_state, next_action]
                                                      - self.q_table[self.state, self.action])
        elif self.method == "q_learning":
            self.q_table[self.state, self.action] += self.alpha \
                                                     * (reward + self.gamma * np.max(self.q_table[next_state, :])
                                                        - self.q_table[self.state, self.action])
        elif self.method == "exSarsa":
            expected_q = (1 - self.epsilon) * np.max(self.q_table[next_state, :]) \
                         + self.epsilon * np.mean(self.q_table[next_state, :])
            self.q_table[self.state, self.action] += self.alpha * (reward + self.gamma * expected_q
                                                                   - self.q_table[self.state, self.action])
        else:
            raise NotImplemented

        if self.eps_update == "vdbe":
            self.new_q = self.q_table[self.state, self.action]
            self.update_epsilon()

        self.state, self.action = next_state, next_action

    def update_epsilon(self, decrease_factor=0.999):
        if self.eps_update == "e-greedy":
            self.epsilon = max(0.1, self.epsilon * decrease_factor)
            self.epsilon_values.append(self.epsilon)
        elif self.eps_update == "vdbe":
            self.epsilons_state[self.state] = (1 - self.delta_vdbe) * \
                                              self.epsilons_state[self.state] + self.delta_vdbe * \
                                              misc.vdbe_function(np.abs(self.new_q - self.old_q), self.sigma)
            self.epsilon_values[self.state].append(self.epsilons_state[self.state])
        else:
            raise NotImplemented

    def generate_policy(self):
        policy = np.argmax(self.q_table, axis=1)
        value = np.amax(self.q_table, axis=1)
        return policy, value


class ArfAgent:
    """
    This agent implements the classic Automatic Rate Fallback (ARF) algorithm found in
    Kamerman, A. and Monteban, L. (1997), WaveLAN®‐II: a high‐performance wireless LAN for the unlicensed band.
    Bell Labs Tech. J., 2: 118-133. doi:10.1002/bltj.2069

    Implementation taken from
    Mathieu Lacage, Mohammad Hossein Manshaei, and Thierry Turletti. 2004. IEEE 802.11 rate adaptation: a practical
    approach. In Proceedings of the 7th ACM international symposium on Modeling, analysis and simulation of wireless
    and mobile systems (MSWiM ’04). Association for Computing Machinery, New York, NY, USA, 126–134.
    DOI:https://doi.org/10.1145/1023663.1023687

    NOTE: The agent is intended to be used with AdAmcPacket envs.
    """

    def __init__(self, action_space):
        self._max_mcs = action_space.n - 1
        self._pkt_succ_count = 0

    def act(self, state, info=None):
        """
        Perform action given the state.

        Parameters
        ----------
        state : dict
            "mcs" : int
                MCS used for the previous packet
            "pkt_succ" : int
                Last packet was successful (1), failed (0), or was retx'd (2). If None, the communication just started.
        info : dict
            Not used, kept to maintain the same signature for all agents.

        Returns
        -------
        mcs : int
        """
        success = state["pkt_succ"]
        if success is None:
            # Initialize with highest MCS
            return self._max_mcs

        mcs = state["mcs"]

        if success == 1:
            # If transmission is successful
            self._pkt_succ_count += 1

            if self._pkt_succ_count == 10:
                # Reset counter
                self._pkt_succ_count = 0
                # Increase MCS if possible
                return min(self._max_mcs, mcs + 1)

            else:
                return mcs

        else:
            # If transmission fails: fall back
            self._pkt_succ_count = 0
            return max(0, mcs - 1)


class AarfAgent:
    """
    This agent implements the Adaptive Automatic Rate Fallback (AARF) algorithm found in
    Mathieu Lacage, Mohammad Hossein Manshaei, and Thierry Turletti. 2004. IEEE 802.11 rate adaptation: a practical
    approach. In Proceedings of the 7th ACM international symposium on Modeling, analysis and simulation of wireless
    and mobile systems (MSWiM ’04). Association for Computing Machinery, New York, NY, USA, 126–134.
    DOI:https://doi.org/10.1145/1023663.1023687

    NOTE: The agent is intended to be used with AdAmcPacket envs.
    """

    def __init__(self, action_space):
        self._max_mcs = action_space.n - 1

        self._pkt_succ_count = 0
        self._pkt_fail_count = 0

        self._succ_to_advance = 10

    def act(self, state, info=None):
        """
        Perform action given the state.

        Parameters
        ----------
        state : dict
            "mcs" : int
                MCS used for the previous packet
            "pkt_succ" : int
                Last packet was successful (1), failed (0), or was retx'd (2). If None, the communication just started.
        info : dict
            Not used, kept to maintain the same signature for all agents.

        Returns
        -------
        mcs : int
        """
        success = state["pkt_succ"]
        if success is None:
            # Initialize with highest MCS
            return self._max_mcs

        mcs = state["mcs"]

        if success == 1:
            # If transmission is successful
            self._pkt_succ_count += 1
            self._pkt_fail_count = 0

            if self._pkt_succ_count == self._succ_to_advance:
                # Reset counter
                self._pkt_succ_count = 0
                # Increase MCS if possible
                return min(self._max_mcs, mcs + 1)

            else:
                return mcs

        else:
            # If transmission fails
            self._pkt_succ_count = 0
            self._pkt_fail_count += 1

            if self._pkt_fail_count == 1:
                self._succ_to_advance = min(50, 2 * self._succ_to_advance)
            else:
                self._succ_to_advance = 10

            return max(0, mcs - 1)


class OnoeAgent:
    """
    This agent implements the Onoe protocol for link rate adaptation, used in some 802.11 drivers such as MadWiFi.
    No papers on the actual implementations were found. This implementation is based on
    He, J., Tang, Z., Chen, H.‐H. and Wang, S. (2012), Performance analysis of ONOE protocol—an IEEE 802.11 link
    adaptation algorithm. Int. J. Commun. Syst., 25: 821-831. doi:10.1002/dac.1290

    NOTE: The agent is intended to be used with AdAmcPacket envs.
    """

    def __init__(self, action_space, theta_r=0.5, theta_c=0.1, n_c=10, window=100e-3):
        self._max_mcs = action_space.n - 1

        self._theta_r = theta_r
        self._theta_c = theta_c
        self._n_c = n_c
        self._window = window

        self._window_start = None
        self._pkt_succ_count = 0
        self._pkt_fail_count = 0
        self._credit = 0

        self._mcs = self._max_mcs

    def act(self, state, info):
        """
        Perform action given the state.

        Parameters
        ----------
        state : dict
            "pkt_succ" : int
                Last packet was successful (1), failed (0), or was retx'd (2). If None, the communication just started.
        info : dict
            "current_time" : float
                The absolute time when the last packet was sent.

        Returns
        -------
        mcs : int
        """
        success = state["pkt_succ"]
        if success is None:
            # Initialize with highest MCS
            return self._max_mcs

        time = info["current_time"]

        if self._window_start is None:
            self._window_start = time

        if time - self._window_start < self._window:
            # Same window: updated counters and keep the same MCS
            self._update_counters(success)

        else:
            # New window
            self._window_start += self._window
            tot_packets = self._pkt_succ_count + self._pkt_fail_count

            if self._pkt_fail_count / tot_packets > self._theta_r:
                # (1)
                self._credit = 0
                self._mcs = max(0, self._mcs - 1)

            elif self._pkt_fail_count / tot_packets > self._theta_c:
                # (2)
                self._credit = max(0, self._credit - 1)

            else:
                # (3)
                self._credit += 1

            if self._credit == self._n_c:
                # (4)
                self._credit = 0
                self._mcs = min(self._max_mcs, self._mcs + 1)

            # New window: reset counters
            self._pkt_succ_count = 0
            self._pkt_fail_count = 0

        return self._mcs

    def _update_counters(self, success):
        if success == 1:
            self._pkt_succ_count += 1
        else:
            self._pkt_fail_count += 1


class MinstrelAgent:
    """
    This agent implements the Minstrel link adaptation algorithm as described at:
    https://wireless.wiki.kernel.org/en/developers/documentation/mac80211/ratecontrol/minstrel
    """

    def __init__(self, action_space, harq_retx, window=1e-1, ewma_weight=.25, lookaround_rate=0.1):
        self.n_mcs = action_space.n
        self.max_mcs = action_space.n - 1
        self.window = window
        self.ewma_weight = ewma_weight
        self.lookaround_rate = lookaround_rate
        self.tot_pkt_attempts = harq_retx + 1  # first transmission + retransmissions

        assert self.tot_pkt_attempts % 4 == 0, "the total number of attempts must be a multiple of 4 (retry chain " \
                                               "length) "

        self.window_start = None
        self.retry_window_start = None
        self.pkt_stats = np.zeros((self.n_mcs, 2))  # 1st column: successful pkts; 2nd column: failed/retransmitted pkts
        self.p_success = np.zeros((self.n_mcs,))  # p_success of each MCS
        self.estimated_thr = np.zeros((self.n_mcs,))  # estimated throughput for each MCS
        self.mcs_s = [mcs for mcs in range(self.n_mcs)]
        self.retry_chain = [None] * 4  # Minstrel's retry chain consists of four MCSs
        self.attempt_per_mcs = self.tot_pkt_attempts // 4
        self.attempt_number = 1

    def act(self, state, info):
        success = state["pkt_succ"]

        if success == 2:
            # retransmission
            self.attempt_number += 1
        else:
            # reset number of attempts for the next packet transmission
            self.attempt_number = 1
            # compute the retry chain for the next packet
            self.retry_chain = self._get_retry_chain()

        current_time = info["current_time"]
        if self.window_start is None:
            self.window_start = current_time
        if self.retry_window_start is None:
            self.retry_window_start = current_time

        if current_time - self.window_start < self.window:
            self._collect_stats(state["mcs"], success)
        else:
            self.window_start += self.window
            # reset packet statistics
            self._update_stats(info["pkt_size"])

        assert self.attempt_number <= self.tot_pkt_attempts, f"Attempt number={self.attempt_number} exceeds the max " \
                                                             f"number of attempts per packet: {self.tot_pkt_attempts}"
        # Choose the MCS based on the attempt number and the retry chain
        if self.attempt_number <= self.attempt_per_mcs:
            selected_mcs = self.retry_chain[0]
        elif self.attempt_number <= self.attempt_per_mcs * 2:
            selected_mcs = self.retry_chain[1]
        elif self.attempt_number <= self.attempt_per_mcs * 3:
            selected_mcs = self.retry_chain[2]
        elif self.attempt_number <= self.attempt_per_mcs * 4:
            selected_mcs = self.retry_chain[3]
        else:
            raise ValueError(f"No MCS is available for attempt number={self.attempt_number}")
        # print(f"attempt: {self.attempt_number}, with mcs={selected_mcs}")
        return selected_mcs

    def _get_retry_chain(self):
        if np.random.rand() <= self.lookaround_rate:
            # sampling transmission: pick-up random MCS (look around)
            available_mcs_s = [mcs for mcs in self.mcs_s if mcs not in self.retry_chain]
            random_mcs = np.random.choice(available_mcs_s)
            if self.estimated_thr[random_mcs] >= np.amax(self.estimated_thr):
                # first rate is the randomly selected MCS, best will be the second
                return [random_mcs, np.argmax(self.estimated_thr), np.argmax(self.p_success), 0]
            else:
                # first rate is best, the randomly selected MCS will be the second
                return [np.argmax(self.estimated_thr), random_mcs, np.argmax(self.p_success), 0]
        else:
            # normal transmission: exploit Minstrel's default retry chain
            sorted_mcs_s = np.argsort(self.estimated_thr)
            return [sorted_mcs_s[-1], sorted_mcs_s[-2], np.argmax(self.p_success), 0]

    def _update_stats(self, pkt_size):
        tot_pkts = self.pkt_stats[:, 0] + self.pkt_stats[:, 1]
        pkt_size_bytes = pkt_size / 8
        for mcs, tot in enumerate(tot_pkts):
            if tot != 0:
                new_p_success = self.pkt_stats[mcs][0] / tot
                self.p_success[mcs] = self.p_success[mcs] * self.ewma_weight + new_p_success * (1.0 - self.ewma_weight)
                self.estimated_thr[mcs] = self.p_success[mcs] * (pkt_size_bytes /
                                                                 dot11ad_constants.get_TXTIME(pkt_size_bytes, mcs))

        # reset stats
        self.pkt_stats = np.zeros((self.n_mcs, 2))

    def _collect_stats(self, mcs, success):
        if success == 1:
            self.pkt_stats[mcs, 0] += 1  # increment of 1 the # of packets successful with that MCS
        else:
            self.pkt_stats[mcs, 1] += 1  # increment of 1 the # of packets failed/retransmitted with that MCS


class PredictiveTargetBerAgent:
    """
    This agent computes the MCS based on a target BER and a prediction of the next packet's SNR
    """

    def __init__(self, action_space, error_model, history_length, prediction_func, target_ber=1e-6):
        self._max_mcs = action_space.n - 1
        self._error_model = error_model
        self._prediction_func = prediction_func
        self._target_ber = target_ber

        assert history_length > 0, "history_length must be strictly positive"
        self._snr_history = [None] * history_length
        self._time_history = [None] * history_length

    def act(self, state, info):
        """
        Perform action given the state.

        Parameters
        ----------
        state : dict
            "snr" : float
                The SNR probed by the previous packet [dB]
            "time" : float
                The departure time of the previous packet [s]
        info : dict
            "current_time" : float
                The current time, when the next packet is being sent [s]

        Returns
        -------
        mcs : int
        """
        # Update internal history
        self._snr_history = self._snr_history[1:] + [state["snr"]]
        self._time_history = self._time_history[1:] + [state["time"]]

        # Remove leaning None's at the beginning of the observation
        times = [t for s, t in zip(self._snr_history, self._time_history) if s is not None]
        snrs = [s for s in self._snr_history if s is not None]

        # Predict SNR
        current_time = info["current_time"]
        predicted_snr = self._prediction_func(times, snrs, current_time)

        bers = [self._error_model.get_ber(predicted_snr, mcs)
                for mcs in range(self._max_mcs + 1)]
        valid_mcs_list = np.nonzero(np.array(bers) <= self._target_ber)[0]

        if valid_mcs_list.size == 0:
            # If no valid MCS: return the lowest one
            return 0
        else:
            # Return the highest valid MCS
            return valid_mcs_list[-1]


class RraaAgent:
    """
    This agent implements the Robust Rate Adaptation Algorithm (RRAA) for link rate adaptation.
    The reference paper is
    Wong, Starsky & Lu, Songwu & Yang, H. & Bharghavan, Vaduvur. (2006). Robust rate adaptation for 802.11 wireless
    networks. Proceedings of the Annual International Conference on Mobile Computing and Networking, MOBICOM. 2006.
    pp. 146-157. 10.1145/1161089.1161107.

    NOTE: Only the loss estimation and the rate change are implemented. The A-RTS filter has been excluded from this
    first implementation.
    NOTE: The agent is intended to be used with AdAmcPacket envs.
    """

    def __init__(self, action_space, target_pkt_size, alpha=1.25, beta=2):
        self._max_mcs = action_space.n - 1
        self._target_pkt_size = target_pkt_size
        self._alpha = alpha
        self._beta = beta

        # Algorithm parameters (described in the paper)
        self._critical_loss_ratio = [None] * (self._max_mcs + 1)
        self._p_ori = [0] * (self._max_mcs + 1)  # Opportunistic Rate Increase threshold
        self._p_mtl = [1] * (self._max_mcs + 1)  # Maximum Tolerable Loss threshold
        self._ewnd = [80] * (self._max_mcs + 1)  # Estimation WiNDow

        # Algorithm counters
        self._lost_frames_list = []  # list of boolean: True=lost frame, False=success frame
        self._mcs = self._max_mcs  # initialize with max MCS

        self._setup_rraa_parameters()

    def act(self, state, info=None):
        """
        Perform action given the state.

        Parameters
        ----------
        state : dict
            "pkt_succ" : int
                Last packet was successful (1), failed (0), or was retx'd (2). If None, the communication just started.
        info : dict
            Not used, kept to maintain the same signature for all agents.

        Returns
        -------
        mcs : int
        """
        # Append new packet lost to list
        self._lost_frames_list.append(state["pkt_succ"] != 1)
        lost_frames = self._lost_frames_list.count(True)
        tx_frames = len(self._lost_frames_list)

        assert tx_frames <= self._ewnd[self._mcs], "Something went wrong with the _lost_frames_list"

        if tx_frames == self._ewnd[self._mcs]:
            # Reached the end of the window
            p = lost_frames / tx_frames

            # Check if change of MCS is needed
            if p > self._p_mtl[self._mcs]:
                self._update_mcs(self._mcs - 1)
            elif p < self._p_ori[self._mcs]:
                self._update_mcs(self._mcs + 1)
            else:
                self._update_mcs(self._mcs)

        else:
            # Optimizing responsiveness
            # Best case: assume the remaining packets to be all received
            p = lost_frames / self._ewnd[self._mcs]
            if p > self._p_mtl[self._mcs]:
                self._update_mcs(self._mcs - 1)

            # Worst case: assume the remaining packets to be all lost
            pkts_left = self._ewnd[self._mcs] - tx_frames
            p = (lost_frames + pkts_left) / self._ewnd[self._mcs]
            if p < self._p_ori[self._mcs]:
                self._update_mcs(self._mcs + 1)

        return self._mcs

    def _setup_rraa_parameters(self):
        # Compute critical loss ration
        tx_time = [dot11ad_constants.get_total_tx_time(self._target_pkt_size, mcs)
                   for mcs in range(self._max_mcs + 1)]

        for mcs in range(1, self._max_mcs + 1):
            # [0]: None
            self._critical_loss_ratio[mcs] = 1 - tx_time[mcs] / tx_time[mcs - 1]
            # [0]: 1
            self._p_mtl[mcs] = self._alpha * self._critical_loss_ratio[mcs]
        for mcs in range(self._max_mcs):
            # [max_mcs]: 0
            self._p_ori[mcs] = self._p_mtl[mcs + 1] / self._beta
            # [max_mcs]: 80
            self._ewnd[mcs] = math.ceil(1 / self._p_ori[mcs])

        assert len(self._critical_loss_ratio) == self._max_mcs + 1, 'RRAA parameters are inconsistent'
        assert len(self._p_ori) == self._max_mcs + 1, 'RRAA parameters are inconsistent'
        assert len(self._p_mtl) == self._max_mcs + 1, 'RRAA parameters are inconsistent'
        assert len(self._ewnd) == self._max_mcs + 1, 'RRAA parameters are inconsistent'

    def _update_mcs(self, new_mcs):
        """
        Try to update the MCS, clipping it between the minimum and maximum accepted value.

        As described in the paper, if the MCS is updated, the counters are reset.
        If not, the window slides forward.

        Parameters
        ----------
        new_mcs : int
        """
        clipped_mcs = max(0, min(self._max_mcs, new_mcs))

        if clipped_mcs != self._mcs:
            self._mcs = clipped_mcs
            # Reset counters
            self._lost_frames_list = []
        else:
            # Slide window
            self._lost_frames_list = self._lost_frames_list[1:]


class HrcAgent:
    """
    This agent implements the Hybrid Rate Control (HRC) algorithm for link rate adaptation.
    The reference paper is
    Ivaylo Haratcherev, Koen Langendoen, Reginald Lagendijk, and Henk Sips. 2004. Hybrid rate control for IEEE 802.11.
    In Proceedings of the second international workshop on Mobility management & wireless access protocols
    (MobiWac ’04). Association for Computing Machinery, New York, NY, USA, 10–18.
    DOI:https://doi.org/10.1145/1023783.1023787

    NOTE: The agent is intended to be used with AdAmcPacket envs.
    NOTE: The paper discusses and shows measurements regarding SNR measurements being symmetric between TX and RX
    NOTE: The max number of retries of the chipset used for the experiment is set to 10 by default
    NOTE: Some procedures/initialization were not described or were unclear
    """

    def __init__(self, action_space, error_model, target_pkt_size, a=0.1, b=1, c=3, d=1, e=10, f=5, window=1,
                 ssia_diff_thresh=2):
        self._max_mcs = action_space.n - 1
        self._error_model = error_model
        self._target_pkt_size = target_pkt_size
        self._a = a
        self._b = b
        self._c = c
        self._d = d
        self._e = e
        self._f = f
        self._window = window
        self._ssia_diff_thresh = ssia_diff_thresh  # No value was ever mentioned in the paper [dB/ms]

        # Algorithm variables (described in the paper)
        self._window_start = None
        self._rs_opt = self._max_mcs  # Start from the max MCS
        self._may_upscale = None
        self._try_upscale = None
        self._rs_lo_bound = None
        self._rs_up_bound = None
        self._ssia_tbl = None

        # Algorithm counters
        self._ssia_history = [None] * 3  # History of the last 3 Signal Strength Indicator of the Acknowledged frames
        self._ssia_time = [None] * 3  # History of the last 3 SSIA acquisition times
        self._pkt_succ_window = np.array([], dtype=np.int)  # Success code for packets in a window
        self._mcs_used_window = np.array([], dtype=np.int)  # MCS used for packets in a window
        self._ssia_window = np.array([])  # SSIA for the packets in a window

        self._rs_curr = self._max_mcs  # Start from max MCS

        self._create_ssia_tbl()

    def act(self, state, info):
        """
        Perform action given the state.

        Parameters
        ----------
        state : dict
            "mcs" : int
                MCS used for the previous packet
            "pkt_succ" : int
                Last packet was successful (1), failed (0), or was retx'd (2). If None, the communication just started.
            "snr" : float
                The SNR probed by the previous packet [dB]
            "time" : float
                The departure time of the previous packet [s]
        info : dict
            "current_time" : float
                The current time, when the next packet is being sent [s]

        Returns
        -------
        mcs : int
        """
        # Collect window stats
        if state["mcs"] is not None:
            # MCS is None for the first packet
            self._pkt_succ_window = np.append(self._pkt_succ_window, state["pkt_succ"])
            self._mcs_used_window = np.append(self._mcs_used_window, state["mcs"])
            self._ssia_window = np.append(self._ssia_window, state["snr"])

        # Check SSIA dynamics
        curr_ssia = state["snr"]  # TODO should retx/failed packet be taken into account?
        self._ssia_history = self._ssia_history[1:] + [curr_ssia]
        self._ssia_time = self._ssia_time[1:] + [state["time"]]

        is_dynamic = self._check_is_dynamic()
        self._calculate_bounds(curr_ssia, is_dynamic)

        # roughly corresponds to the function for_each_packet() described in the paper
        if self._window_start is None or info["current_time"] - self._window_start > self._window:
            self._once_per_decision_window(info["current_time"])

        if self._try_upscale:
            if state["pkt_succ"] == 1:
                self._rs_opt = self._rs_curr
            else:
                self._may_upscale = False
            self._try_upscale = False

        self._rs_curr = self._probe_or_not()

        if self._rs_curr > self._rs_up_bound:
            self._rs_curr = self._rs_up_bound
        elif self._rs_curr < self._rs_lo_bound and self._may_upscale:
            self._rs_curr = self._rs_lo_bound
            self._try_upscale = True

        return self._rs_curr

    def _create_ssia_tbl(self):
        # Initialize the SSIA table
        # As the paper does not say how to initialize it, the following heuristic has been created
        # For a given MCS, the low threshold is based on the minimum SNR that ensures at least 10% of the achievable
        # throughput. High and dynamic low thresholds are computed similarly to the table update function
        snr = np.arange(-14, 30, 0.25)
        pkt_size = self._target_pkt_size

        lo_thld = np.zeros((self._max_mcs + 1,))
        for mcs in range(self._max_mcs + 1):
            tx_time = dot11ad_constants.get_total_tx_time(pkt_size, mcs)
            psr = np.array([self._error_model.get_packet_success_rate(s, mcs, pkt_size) for s in snr])
            thr = (pkt_size / tx_time) * psr
            # find the lowest SNR that ensures at least 10% of the achievable throughput
            lo_thld_idx = np.nonzero(thr > (thr[-1] * 0.1))[0][0]
            lo_thld[mcs] = snr[lo_thld_idx]

        self._ssia_tbl = pd.DataFrame({"lo_thld": lo_thld,
                                       "hi_thld": lo_thld + self._e,
                                       "lo_thld_dyn": lo_thld + self._f})
        self._fix_ssia_tbl()

    def _once_per_decision_window(self, current_time):
        if self._window_start is None:
            self._window_start = current_time
        else:
            self._update_ssia_tbl()
            # Allow possible jumps of multiple windows in extreme cases
            while current_time - self._window_start > self._window:
                self._window_start += self._window

        self._rs_opt = self._find_opt_rate_by_status()
        self._may_upscale = True
        self._pkt_succ_window = np.array([], dtype=np.int)
        self._mcs_used_window = np.array([], dtype=np.int)
        self._ssia_window = np.array([])

    def _find_opt_rate_by_status(self):
        if len(self._pkt_succ_window) == 0:
            # No data on the previous window: keep the same MCS
            return self._rs_opt

        else:
            mcs_used = np.unique(self._mcs_used_window)
            thr = np.zeros((self._max_mcs + 1,))
            for mcs in mcs_used:
                mask = self._mcs_used_window == mcs
                # compute empirical throughput
                empirical_psr = np.count_nonzero(self._pkt_succ_window[mask] == 1) / len(self._pkt_succ_window[mask])
                tx_time = dot11ad_constants.get_total_tx_time(self._target_pkt_size, mcs)
                thr[mcs] = self._target_pkt_size / tx_time * empirical_psr

            return np.argmax(thr)

    def _probe_or_not(self):
        p = np.random.rand()
        if p > 0.1:
            return self._rs_opt
        # 10% of the data is sent at the adjacent rates to the current optimal
        if p < 0.05:
            return max(0, self._rs_opt - 1)
        return min(self._max_mcs, self._rs_opt + 1)

    def _calculate_bounds(self, curr_ssia, is_dyn):
        if curr_ssia is None:
            # First packet, keep large bounds (not described in the paper)
            self._rs_lo_bound = 0
            self._rs_up_bound = self._max_mcs
            return

        if is_dyn:
            lo_thld_col = "lo_thld_dyn"
        else:
            lo_thld_col = "lo_thld"

        # Find highest MCS which respects the lower bound
        valid_mcs = np.nonzero(curr_ssia >= self._ssia_tbl[lo_thld_col].to_numpy())[0]
        if len(valid_mcs) > 0:
            self._rs_lo_bound = valid_mcs[-1]
        else:
            # curr_ssia too small: use lowest MCS anyway
            self._rs_lo_bound = 0

        # Find lowest MCS which respects the upper bound
        valid_mcs = np.nonzero(curr_ssia <= self._ssia_tbl["hi_thld"].to_numpy())[0]
        if len(valid_mcs) > 0:
            self._rs_up_bound = valid_mcs[0]
        else:
            # curr_ssia too large: use highest MCS
            self._rs_up_bound = self._max_mcs

    def _update_ssia_tbl(self):
        mcs_used = np.unique(self._mcs_used_window)
        for mcs in mcs_used:
            lo_thld_old = self._ssia_tbl.iloc[mcs]["lo_thld"]
            pkt_succ = self._pkt_succ_window[self._mcs_used_window == mcs]

            n_retx = np.sum(pkt_succ == 2)
            n_good = np.sum(pkt_succ == 1)
            fer = n_retx / (n_retx + n_good) * 100

            if fer > 5:
                lo_thld_new = lo_thld_old + (fer - 5) * self._a + self._b
            elif fer < 1 and np.mean(self._ssia_window[self._mcs_used_window == mcs]) < lo_thld_old:
                lo_thld_new = lo_thld_old - (1 - fer) * self._c + self._d
            else:
                lo_thld_new = lo_thld_old

            self._update_thlds(mcs, lo_thld_new)

        self._fix_ssia_tbl()

    def _check_is_dynamic(self):
        if any([ssia is None for ssia in self._ssia_history]):
            # Assume non-dynamic channel at the beginning
            return False

        # NOTE: it is extremely unlikely that both differences will be larger than 0 with the current RT timestep
        diff = np.diff(self._ssia_history) / (np.diff(self._ssia_time) * 1e3)  # [dB/ms]

        if np.prod(diff) > 0 and np.all(diff > self._ssia_diff_thresh):
            # "If both differences have the same sign, and both exceed a certain threshold, then the dynamic low
            # threshold will be used, since conditions are changing fast and consistently."
            return True
        else:
            return False

    def _update_thlds(self, mcs, lo_thld):
        self._ssia_tbl.iloc[mcs]["lo_thld"] = lo_thld
        self._ssia_tbl.iloc[mcs]["hi_thld"] = lo_thld + self._e
        self._ssia_tbl.iloc[mcs]["lo_thld_dyn"] = lo_thld + self._f

    def _fix_ssia_tbl(self):
        # fix ranges of unused MCSs to make them monotonically increasing
        for mcs in range(1, self._max_mcs + 1):
            prev_lo_thld = self._ssia_tbl.iloc[mcs - 1]["lo_thld"]
            if self._ssia_tbl.iloc[mcs]["lo_thld"] < prev_lo_thld:
                self._update_thlds(mcs, prev_lo_thld)
