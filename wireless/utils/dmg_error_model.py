"""
© 2020, University of Padova, Department of Information Engineering, SIGNET Lab.
"""
import os
import numpy as np
from scipy.interpolate import interp1d


def is_valid(ber):
    if ber < 0.0 or ber > 1.0:
        return False
    else:
        return True


class DmgErrorModel:
    """Dmg Error Model class"""

    def __init__(self, path, n_mcs):
        self.path = os.path.abspath(path)  # Absolute path to the file with BER-vs-SNR curves
        print("error_model_abs_path=", self.path)
        self.n_mcs = n_mcs  # The number of MCSs to load for the error model
        self.bersnr_curves = [None] * self.n_mcs
        self._load_bersnr_curves()

    def _load_bersnr_curves(self):
        assert self.bersnr_curves == [None] * self.n_mcs, "Error tables have already been loaded"
        file = open(self.path, "r")
        total_mcs = int(file.readline())
        assert self.n_mcs <= total_mcs, f"The max number of available MCSs is {total_mcs}"
        next(file)  # Skip line with the snr-dec-places value
        next(file)  # Skip line with the snr-step value
        for mcs in range(self.n_mcs):
            mcs_idx = int(file.readline())
            assert mcs == mcs_idx, "MCS indexes differ"
            next(file)  # Skip line with the min value for the SNR
            next(file)  # Skip line with the max value for the SNR
            ber_max = float(file.readline())
            ber_min = float(file.readline())
            next(file)  # Skip line with the number of data points value
            snr_values = np.array(file.readline().split(","), dtype=float)
            ber_values = np.array(file.readline().split(","), dtype=float)
            self.bersnr_curves[mcs] = interp1d(snr_values, ber_values, kind="linear",
                                               bounds_error=False, fill_value=(ber_max, ber_min))
        file.close()

    def get_ber(self, snr, mcs):
        assert mcs in range(self.n_mcs), f"{mcs} is not a valid MCS. Max MCS={self.n_mcs-1}"
        ber = self.bersnr_curves[mcs](snr)
        assert is_valid(ber), f"{ber} is not a probability value"
        return ber

    def get_packet_success_rate(self, snr, mcs, n_bits=0):
        assert n_bits != 0, "The number of bits must be greater than zero"
        return pow(1 - self.get_ber(snr, mcs), n_bits)
