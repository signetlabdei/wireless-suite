"""
© 2020, University of Padova, Department of Information Engineering, SIGNET Lab.
"""
import gym
import numpy as np
import matplotlib.pyplot as plt
from wireless.utils.misc import get_mcs_data_rate
from wireless.agents.rate_manager_agents import ConstantRateAgent

MCS_MODE = 12  # MCS for the Constant Rate Agent
SCENARIOS_LIST = ["Journal1Lroom_1.csv"]  # List of scenarios for the environment
SNR_HISTORY = 5  # The number of past SNR values to consider for the state
NET_TIMESTEP = 0.005  # The real network timestep [s]
OBS_DURATION = 19.15  # Observation duration [s]; 19.15 s is the duration of LRoom scenario
MCS_LIST = np.arange(13)
THR_STEP = 20
START_TIME = NET_TIMESTEP*(SNR_HISTORY-1)
TIME = np.arange(START_TIME, OBS_DURATION, NET_TIMESTEP * THR_STEP)


def main():

    thr_t_mcs = [None] * len(MCS_LIST)
    env = gym.make("AdLinkAdaptation-v0", scenarios_list=SCENARIOS_LIST, obs_duration=OBS_DURATION,
                   snr_history=SNR_HISTORY, net_timestep=NET_TIMESTEP)
    for mcs in MCS_LIST:

        agent = ConstantRateAgent(mcs)
        reward_t = []
        tot_bits_t = []
        done = False
        state = env.reset()
        tot_steps = 0
        thr_t = []
        thr_values = []
        thr_step = THR_STEP  # timestep for thr calculation

        while not done:
            action = agent.act(state)
            new_state, reward, done, debug = env.step(action)

            tot_bits_generated = sum(debug["tx_pkts_list"])
            tot_bits_t.append(tot_bits_generated)
            reward_t.append(reward)
            thr_values.append(reward / NET_TIMESTEP)
            thr_step -= 1
            if thr_step == 0:
                thr_t.append(np.mean(thr_values))
                thr_values = []
                thr_step = THR_STEP

            old_state = state
            state = new_state
            tot_steps += 1

        env.close()
        if len(thr_values) != 0:
            thr_t.append(np.mean(thr_values))
        avg_data_rate = np.sum(tot_bits_t, dtype=np.int64) / (OBS_DURATION - START_TIME) / 1e6
        avg_thr = np.sum(reward_t, dtype=np.int64) / (OBS_DURATION - START_TIME) / 1e6
        print(f"Avg data rate [Mbps]: {avg_data_rate}")
        print(f"Avg throughput [Mbps]: {avg_thr}")
        print(f"Average reward: {np.mean(reward_t)}")
        print(f"Total timesteps: {tot_steps}")
        thr_t_mcs[mcs] = np.array(thr_t) / 1e6

    thr_mcs = np.array(thr_t_mcs)
    mcs_max = np.argmax(thr_mcs, axis=0)
    thr_max = np.amax(thr_mcs, axis=0)
    print(f"Avg Optimal throughput [Mbps]: {np.mean(thr_max)}")

    # Plot results
    plt.figure(1)
    plt.plot(TIME, mcs_max, marker="x", linewidth=0.2, markersize=7.0, label="MCS index")
    plt.title("Best MCS for each time value in the simulation", fontsize=18)
    plt.ylabel("MCS index", fontsize=18)
    plt.xlabel("Time [s]", fontsize=18)
    plt.yticks(MCS_LIST, fontsize=12)
    plt.legend(loc="lower left", fontsize=12)
    plt.grid()
    plt.show()

    plt.figure(2)
    for mcs in MCS_LIST:
        label = "MCS " + str(mcs)
        plt.plot(TIME, thr_t_mcs[mcs], linewidth=2.0, label=label)

    plt.title("MCS data rate vs Scenario throughput", fontsize=18)
    plt.ylabel("Rate [Mbps]", fontsize=18)
    plt.xlabel("Time [s]", fontsize=18)
    plt.legend(loc="lower left", fontsize=12)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
