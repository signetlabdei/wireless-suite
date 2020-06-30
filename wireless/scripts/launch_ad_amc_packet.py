"""
© 2020, University of Padova, Department of Information Engineering, SIGNET Lab.
"""
import gym
import numpy as np
import matplotlib.pyplot as plt
from wireless.agents import rate_manager_agents

# Env params
CAMPAIGN = "scenarios_v1"
SCENARIOS_LIST = ["Journal1Lroom_1.csv"]  # List of scenarios for the environment
HISTORY_LENGTH = 1  # The number of past SNR values to consider for the state
NET_TIMESTEP = 5e-3  # The real network timestep [s]
OBS_DURATION = None  # Observation duration [s]; If None, the whole scenario is the observation

# Agents params
AGENT_TYPE = "ONOE"


def main():
    assert len(SCENARIOS_LIST) == 1, "Exactly 1 scenario should be evaluated with this script"

    env = gym.make("AdAmcPacket-v0", campaign=CAMPAIGN, scenarios_list=SCENARIOS_LIST, obs_duration=OBS_DURATION,
                   history_length=HISTORY_LENGTH, net_timestep=NET_TIMESTEP)

    if AGENT_TYPE == "ARF":
        agent = rate_manager_agents.ArfAgent(env.action_space)
    elif AGENT_TYPE == "AARF":
        agent = rate_manager_agents.AarfAgent(env.action_space)
    elif AGENT_TYPE == "ONOE":
        agent = rate_manager_agents.OnoeAgent(env.action_space)
    else:
        raise ValueError(f"Agent '{AGENT_TYPE}' not recognized")

    # Init stats
    reward_t = []
    time = []
    done = False
    tot_pkts = 0
    tot_mb_generated = 0
    mcs_t = []

    # Run agent
    state = env.reset()
    info = {}

    while not done:
        if AGENT_TYPE == "ARF":
            action = agent.act(state, info)
        elif AGENT_TYPE == "AARF":
            action = agent.act(state, info)
        elif AGENT_TYPE == "ONOE":
            action = agent.act(state, info)
        else:
            raise ValueError(f"Agent '{AGENT_TYPE}' not recognized")

        state, reward, done, info = env.step(action)

        tot_mb_generated += info["pkt_size"] / 1e6
        reward_t.append(reward)
        time.append(info["current_time"])
        mcs_t.append(state["mcs"][0])

        tot_pkts += 1

        env.close()

    # Stats
    avg_data_rate_mbps = tot_mb_generated / env.scenario_duration
    avg_thr = np.sum(reward_t, dtype=np.int64) / env.scenario_duration
    print(f"Avg data rate [Mbps]: {avg_data_rate_mbps}")
    print(f"Avg throughput [Mbps]: {avg_thr / 1e6}")
    print(f"Average reward: {np.mean(reward_t)}")
    print(f"Total packets: {tot_pkts}")

    # Plot results
    plt.figure()
    plt.plot(time, mcs_t, label=AGENT_TYPE)
    plt.title(SCENARIOS_LIST[0])
    plt.ylabel("MCS [idx]")
    plt.xlabel("Time [s]")
    plt.legend(loc="lower left")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
