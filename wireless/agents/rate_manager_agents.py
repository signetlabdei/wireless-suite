"""
© 2020, University of Padova, Department of Information Engineering, SIGNET Lab.
"""
import numpy as np
import math
from wireless.utils import misc, dot11ad_constants


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

    def act(self, state, info):
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
