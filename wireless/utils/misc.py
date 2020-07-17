"""
© 2020 Nokia
Licensed under the BSD 3 Clause license
SPDX-License-Identifier: BSD-3-Clause
"""
import os
import pandas as pd
import numpy as np
import math
from scipy import constants
from scipy.interpolate import interp1d


def clip(value, min_value, max_value):
    """
    Clip value between min and max.

    Parameters
    ----------
    value : float
    min_value : float
    max_value : float

    Returns
    -------
    clipped_value : float
    """
    return max(min(value, max_value), min_value)


def calculate_thermal_noise(bw_mhz):
    """
    Compute thermal noise at T0=290 K for the given bandwidth.

    Parameters
    ----------
    bw_mhz : float

    Returns
    -------
    N : float
    """
    t0_kelvin = 290
    return constants.Boltzmann * t0_kelvin * (bw_mhz * 1E6) * 1000


def import_scenario(filepath):
    """
    Import the channel scenario from the given path.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    df : pd.DataFrame
    """
    path = os.path.abspath(filepath)
    return pd.read_csv(path)


def get_mcs_data_rate(mcs_idx):
    """
    The bitrate of a given MCS index.

    Parameters
    ----------
    mcs_idx : int
        IEEE 802.11ad MCS index between [0,12] (Control and SC PHY)

    Returns
    -------
    bitrate : float
    """
    switcher = {
        0: 27.5 * 1e6,
        1: 385 * 1e6,
        2: 770 * 1e6,
        3: 962.5 * 1e6,
        4: 1155 * 1e6,
        5: 1251.25 * 1e6,
        6: 1540 * 1e6,
        7: 1925 * 1e6,
        8: 2310 * 1e6,
        9: 2502.5 * 1e6,
        10: 3080 * 1e6,
        11: 3850 * 1e6,
        12: 4620 * 1e6,
    }
    return switcher.get(mcs_idx, None)


def get_tx_pkt_size_list(mcs_idx, time, packet_size):
    """
    List of packet sizes for a given time interval, MCS index and target packet size.

    The last packet might be small the the others to fit the time.

    Parameters
    ----------
    mcs_idx : int
    time : float
        Time interval where packets are sent.
    packet_size : float
        Target packet size in [b].

    Returns
    -------
    n_packets : int
        The number of packets that fit in the interval.
    last_pkt : float
        Size of the last packet in [b].
    tx_pkts_list : list of float
        List containing the size of each packet.
    """
    mcs_rate = get_mcs_data_rate(mcs_idx)
    assert mcs_rate is not None, f"{mcs_idx} is not a valid MCS or the format is wrong"

    data_rate = int(mcs_rate * time)
    n_packets = data_rate // int(packet_size)
    tx_pkts_list = [packet_size] * n_packets

    last_pkt = data_rate % packet_size
    if last_pkt != 0:
        tx_pkts_list.append(last_pkt)

    return n_packets, last_pkt, tx_pkts_list


def generate_policy(q_table):
    policy = np.argmax(q_table, axis=1)
    value = np.amax(q_table, axis=1)
    return policy, value


def get_timestep(time, timestep):
    """
    Get the timestep index given a time and a timestep duration.

    Parameters
    ----------
    time : float
    timestep : float
        The timestep duration.

    Returns
    -------
    idx : int
    """
    return math.floor(time / timestep)


def epsilon_greedy(q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(0, len(q_values))  # to change if only a subset of actions is allowed
    else:
        max_q_value = np.max(q_values)
        return np.random.choice([action_ for action_, value_ in enumerate(q_values) if value_ == max_q_value])


def vdbe_function(td_error_abs, sigma):
    return (1 - np.exp(-td_error_abs / sigma)) / (1 + np.exp(-td_error_abs / sigma))


def get_packet_duration(packet_size, mcs):
    """
    Transmission duration for a packet of a given size with the given MCS index.

    Parameters
    ----------
    packet_size : int
        Packet size in [b].
    mcs : int
        MCS index.

    Returns
    -------
    duration : float
    """
    rate = get_mcs_data_rate(mcs)
    return packet_size / rate


def predict_snr(t, snr, t_next, kind="previous"):
    assert len(t) == len(snr), "x and y must have the same size"
    if len(t) == 0:
        # Nothing to interpolate, assume inf SNR
        return np.inf

    if len(t) == 1:
        # Assume constant SNR
        return snr

    f = interp1d(t, snr,
                 kind=kind,
                 fill_value="extrapolate")
    return f(t_next)


def get_windowed_throughput(t, bits, window):
    """
    Compute the throughput over contiguous non-overlapping time windows.

    Parameters
    ----------
    t : list of float
        Packet arrival time [s]
    bits : list of int
        Packet size [b]
    window : float
        Duration of the time window [s]

    Returns
    -------
    t_window : list of float
        The initial time of each window [s]
    thr_window : list of float
        The average throughput in each window [Mbps]
    """
    t = np.array(t)
    bits = np.array(bits)

    n = math.ceil((t[-1] - t[0]) / window)
    t0 = t[0]

    t_window = np.full((n,), np.nan)
    thr_window = np.full((n,), np.nan)
    for i in range(n):
        t_min = t0 + i * window
        t_max = t0 + min(t[-1], (i + 1) * window)
        dt = t_max - t_min
        mask = np.bitwise_and(t_min <= t, t < t_max)

        t_window[i] = t_min
        thr_window[i] = np.sum(bits[mask]) / dt / 1e6

    return t_window, thr_window
