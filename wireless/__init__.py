from gym.envs.registration import register

register(
    id='TimeFreqResourceAllocation-v0',
    entry_point='wireless.envs:TimeFreqResourceAllocationV0',
)

register(
    id='AdLinkAdaptation-v0',
    entry_point='wireless.envs:AdLinkAdaptationV0',
)
