"""
© 2020, University of Padova, Department of Information Engineering, SIGNET Lab.
"""
import numpy as np
import tensorflow as tf
from wireless.utils import misc
from wireless.utils.misc import get_mcs_data_rate


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
                MCS used for the previous packet(s)
            "pkt_succ" : int
                Last packet(s) were successful (1) or not. If None, the communication just started.
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
                MCS used for the previous packet(s)
            "pkt_succ" : int
                Last packet(s) were successful (1) or not (0). If None, the communication just started.
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
                Last packet(s) were successful (1) or not (0). If None, the communication just started.
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

    def act(self, state, info=None):
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
