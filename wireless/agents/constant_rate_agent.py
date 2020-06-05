"""
© 2020, University of Padova, Department of Information Engineering, SIGNET Lab.
"""


class ConstantRateAgent:
    """
    This agent uses the same MCS for the entire simulation
    (i.e. no adaptation is performed)
    """
    def __init__(self, mcs):
        self.mcs = mcs

    def act(self, state, reward, done):
        return self.mcs
