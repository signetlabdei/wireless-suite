"""
© 2020, University of Padova, Department of Information Engineering, SIGNET Lab.
"""
import gym
import numpy as np
import matplotlib.pyplot as plt
from wireless.agents import rate_manager_agents

# Env params
CAMPAIGN = "scenarios_v3"
SCENARIOS_LIST = ["Journal1Lroom_1.csv"]  # List of scenarios for the environment
HISTORY_LENGTH = 1  # The number of past SNR values to consider for the state
NET_TIMESTEP = 5e-3  # The real network timestep [s]
OBS_DURATION = None  # Observation duration [s]; If None, the whole scenario is the observation
HARQ_RETX = 2  # The number of HARQ retransmission after the first transmission
REWARD_TYPE = "rx_bits"


def main(agent_type="ARF"):
    assert len(SCENARIOS_LIST) == 1, "Exactly 1 scenario should be evaluated with this script"

    env = gym.make("AdAmcPacket-v0", campaign=CAMPAIGN, scenarios_list=SCENARIOS_LIST, obs_duration=OBS_DURATION,
                   history_length=HISTORY_LENGTH, net_timestep=NET_TIMESTEP, harq_retx=HARQ_RETX,
                   reward_type=REWARD_TYPE)

    if agent_type == "ARF":
        agent = rate_manager_agents.ArfAgent(env.action_space)
    elif agent_type == "AARF":
        agent = rate_manager_agents.AarfAgent(env.action_space)
    elif agent_type == "ONOE":
        agent = rate_manager_agents.OnoeAgent(env.action_space)
    else:
        raise ValueError(f"Agent '{agent_type}' not recognized")

    # Init stats
    done = False
    tot_sent_pkts = 0  # Excluding retransmissions
    tot_mb_generated = 0  # Excluding retransmissions
    tot_mb_rx = 0
    tot_pkts = 0  # Including retransmissions
    tot_rx_pkts = 0

    rx_pkts_retx = []
    rx_pkts_delay = []

    time = []
    mcs_t = []
    reward_t = []
    retx_t = []
    delay_t = []

    # Run agent
    state = env.reset()
    info = {}

    while not done:
        if agent_type == "ARF":
            action = agent.act(state, info)
        elif agent_type == "AARF":
            action = agent.act(state, info)
        elif agent_type == "ONOE":
            action = agent.act(state, info)
        else:
            raise ValueError(f"Agent '{agent_type}' not recognized")

        state, reward, done, info = env.step(action)

        if state["pkt_retx"][0] == 0:
            # If retx == 0: new packet, irrespective of its success
            tot_sent_pkts += 1
            tot_mb_generated += info["pkt_size"] / 1e6

        tot_pkts += 1

        if state["pkt_succ"][0] == 1:
            tot_rx_pkts += 1
            tot_mb_rx += info["pkt_size"] / 1e6
            rx_pkts_retx.append(state["pkt_retx"][0])
            rx_pkts_delay.append(state["pkt_delay"][0])

        reward_t.append(reward)
        time.append(info["current_time"])
        mcs_t.append(state["mcs"][0])
        retx_t.append(state["pkt_retx"][0])
        delay_t.append(state["pkt_delay"][0])

    env.close()

    # Stats
    avg_generated_rate_mbps = tot_mb_generated / env.scenario_duration
    avg_thr_mbps = tot_mb_rx / env.scenario_duration
    avg_reward = np.mean(reward_t)
    psr = tot_rx_pkts / tot_sent_pkts
    avg_succ_pkt_retx = np.mean(rx_pkts_retx)
    avg_succ_pkt_delay = np.mean(rx_pkts_delay)

    print("***")
    print(f"Agent type: {agent_type}")
    print(f"Avg generated data rate [Mbps]: {avg_generated_rate_mbps}")
    print(f"Avg throughput [Mbps]: {avg_thr_mbps}")
    print(f"Average reward: {avg_reward}")
    print(f"Total packets sent: {tot_sent_pkts}, transmitted (including RETXs): {tot_pkts}")
    print(f"Average packet success rate: {psr}")
    print(f"Average successful packet RETX: {avg_succ_pkt_retx}")
    print(f"Average successful packet delay [us]: {avg_succ_pkt_delay * 1e6}")

    # Plot results
    plt.figure()
    plt.plot(time, mcs_t, label=agent_type)
    plt.title(SCENARIOS_LIST[0])
    plt.ylabel("MCS [idx]")
    plt.xlabel("Time [s]")
    plt.legend(loc="lower left")
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(time, retx_t, label=agent_type)
    plt.title(SCENARIOS_LIST[0])
    plt.ylabel("Retransmissions")
    plt.xlabel("Time [s]")
    plt.legend(loc="lower left")
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(time, [d * 1e3 for d in delay_t], label=agent_type)
    plt.title(SCENARIOS_LIST[0])
    plt.ylabel("Delay [ms]")
    plt.xlabel("Time [s]")
    plt.legend(loc="lower left")
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(time, np.cumsum(reward_t, dtype=np.float) / np.arange(1, len(reward_t) + 1), label=agent_type)
    plt.title(SCENARIOS_LIST[0])
    plt.ylabel("Average Cumulative Reward")
    plt.xlabel("Time [s]")
    plt.legend(loc="lower left")
    plt.grid()
    plt.show()

    return


if __name__ == '__main__':
    main(agent_type="ARF")
    main(agent_type="AARF")
    main(agent_type="ONOE")
