"""
© 2020, University of Padova, Department of Information Engineering, SIGNET Lab.
"""
import numpy as np


class ConstantRateAgent:
    """
    This agent uses the same MCS for the entire simulation
    (i.e. no adaptation is performed)
    """

    def __init__(self, mcs):
        self.mcs = mcs

    def act(self, state):
        return self.mcs


class OptimalRateAgent:
    """
    This agent computes the MCS based on the future SNR, hence its optimality
    (the future SNR is the SNR at one timestep ahead with respect to the current timestep)
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
