"""
© 2020 Nokia
Licensed under the BSD 3 Clause license
SPDX-License-Identifier: BSD-3-Clause
"""
import os
import pandas as pd
from scipy import constants


def clip(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def calculate_thermal_noise(bw_mhz):
    t0_kelvin = 290
    return constants.Boltzmann * t0_kelvin * bw_mhz * 1E6 * 1000


def import_scenario(filepath):
    path = os.path.abspath(filepath)
    print("scenario_abs_path=", path)
    df = pd.read_csv(path)
    return df


def get_mcs_data_rate(mcs_idx):
    switcher = {
        0: 27.5*1e6,
        1: 385*1e6,
        2: 770*1e6,
        3: 962.5*1e6,
        4: 1155*1e6,
        5: 1251.25*1e6,
        6: 1540*1e6,
        7: 1925*1e6,
        8: 2310*1e6,
        9: 2502.5*1e6,
        10: 3080*1e6,
        11: 3850*1e6,
        12: 4620*1e6,
    }
    return switcher.get(mcs_idx, None)


def get_tx_pkt_size_list(mcs_idx, time, packet_size):
    mcs_rate = get_mcs_data_rate(mcs_idx)
    assert mcs_rate is not None, f"{mcs_idx} is not a valid MCS or the format is wrong"

    data_rate = int(mcs_rate * time)
    n_packets = data_rate // packet_size
    tx_pkts_list = [packet_size] * n_packets

    last_pkt = data_rate % packet_size
    if last_pkt != 0:
        tx_pkts_list.append(last_pkt)

    return n_packets, last_pkt, tx_pkts_list
