"""
© 2020, University of Padova, Department of Information Engineering, SIGNET Lab.
"""
import gym
import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
from wireless.agents.rate_manager_agents import TargetBerAgent
from wireless.utils.misc import generate_policy
from sklearn import linear_model


TARGET_BER = 1e-6
SNR_HISTORY = 2  # The number of past SNR values to consider for the state
NET_TIMESTEP = 0.005  # The real network timestep [s]
OBS_DURATION = 19.15  # Observation duration [s]; 19.15 s is the duration of LRoom scenario
THR_STEP = 20
START_TIME = NET_TIMESTEP * (SNR_HISTORY - 1)
TIME = np.arange(START_TIME, OBS_DURATION, NET_TIMESTEP * THR_STEP)
state_space = 'v2'
regr_x = np.arange(0, NET_TIMESTEP * SNR_HISTORY, NET_TIMESTEP)

campaign = "scenarios_v3"
SCENARIO = ["Journal1ParkingLot_1.csv"]  # List of scenarios for the environment


def get_baseline(env):

    agent = TargetBerAgent(env.action_space, env.error_model, target_ber=TARGET_BER)
    reward_t = []
    tot_bits_t = []
    done = False
    state = env.reset()
    tot_steps = 0
    thr_t = []
    thr_values = []
    selected_mcs_t = []
    thr_step = THR_STEP  # timestep for thr calculation

    while not done:
        action = agent.act(state["snr"][-1])
        new_state, reward, done, debug = env.step(action)

        tot_bits_generated = sum(debug["tx_pkts_list"])
        tot_bits_t.append(tot_bits_generated)
        reward_t.append(reward)
        thr_values.append(reward / NET_TIMESTEP)
        thr_step -= 1
        if thr_step == 0:
            selected_mcs_t.append(new_state[2])
            thr_t.append(np.mean(thr_values) / 1e6)
            thr_values = []
            thr_step = THR_STEP

        state = new_state
        tot_steps += 1

    env.close()
    if len(thr_values) != 0:
        selected_mcs_t.append(state[2])
        thr_t.append(np.mean(thr_values) / 1e6)

    avg_data_rate = np.sum(tot_bits_t, dtype=np.int64) / (OBS_DURATION - START_TIME) / 1e6
    avg_thr = np.sum(reward_t, dtype=np.int64) / (OBS_DURATION - START_TIME) / 1e6
    print(f"OPTIMAL policy")
    print(f"Avg data rate [Mbps]: {avg_data_rate}")
    print(f"Avg throughput [Mbps]: {avg_thr}")
    print(f"Average reward: {np.mean(reward_t)}")
    print(f"Total timesteps: {tot_steps}")

    return thr_t, selected_mcs_t


def main():
    # Loading tables and returns
    folder = "./"
    qtable_sarsa = loadtxt(folder + "qtable_sarsa.csv", delimiter=",")
    qtable_ql = loadtxt(folder + "qtable_ql.csv", delimiter=",")
    # qtable_exSarsa = loadtxt(folder + "qtable_exSarsa.csv", delimiter=",")

    env = gym.make("AdLinkAdaptation-v0",
                   campaign=campaign,
                   scenarios_list=SCENARIO,
                   obs_duration=OBS_DURATION,
                   snr_history=SNR_HISTORY,
                   net_timestep=NET_TIMESTEP)

    baseline_thr, baseline_mcs = get_baseline(env)

    regr = linear_model.LinearRegression()

    policy_list = ["SARSA", "Q-learning"]
    fig_index = 1

    for p in policy_list:
        reward_t = []
        tot_bits_t = []
        done = False
        observation = env.reset()
        tot_steps = 0
        thr_t = []
        thr_values = []
        mcs_list = []
        sinr_list = []
        thr_step = THR_STEP  # timestep for thr calculation
        action = None

        if p == "SARSA":
            policy_sarsa, value_sarsa = generate_policy(qtable_sarsa)
            policy = policy_sarsa
        elif p == "Q-learning":
            policy_ql, value_ql = generate_policy(qtable_ql)
            policy = policy_ql
        else:
            raise NotImplemented

        bins = np.arange(-5, 16, 1)

        while not done:
            if state_space == 'v2':
                regr.fit(regr_x.reshape(-1, 1), observation["snr"])
                if regr.coef_ > 0:
                    state = np.digitize(observation["snr"][-1], bins) + len(bins) + 1
                else:
                    state = np.digitize(observation["snr"][-1], bins)
            else:
                state = np.digitize(observation["snr"][-1], bins)

            action = policy[state]
            next_observation, reward, done, debug = env.step(action)
            sinr_list.append(np.mean(observation["snr"]))
            tot_bits_generated = sum(debug["tx_pkts_list"])
            tot_bits_t.append(tot_bits_generated)
            reward_t.append(reward)
            thr_values.append(reward / NET_TIMESTEP)
            thr_step -= 1
            if thr_step == 0:
                mcs_list.append(action)
                thr_t.append(np.mean(thr_values) / 1e6)
                thr_values = []
                thr_step = THR_STEP

            observation = next_observation
            tot_steps += 1

        env.close()
        if len(thr_values) != 0:
            mcs_list.append(action)
            thr_t.append(np.mean(thr_values) / 1e6)

        avg_data_rate = np.sum(tot_bits_t, dtype=np.int64) / (OBS_DURATION - START_TIME) / 1e6
        avg_thr = np.sum(reward_t, dtype=np.int64) / (OBS_DURATION - START_TIME) / 1e6
        print(f"{p}")
        print(f"Avg data rate [Mbps]: {avg_data_rate}")
        print(f"Avg throughput [Mbps]: {avg_thr}")
        print(f"Average reward: {np.mean(reward_t)}")
        print(f"Total timesteps: {tot_steps}")

        # Plot results
        fig = plt.figure(fig_index)
        plt.subplot(121)
        plt.plot(TIME, thr_t, marker="x", linewidth=0.2, markersize=7.0, label=p)
        plt.plot(TIME, baseline_thr, linewidth=2, label='OPT')
        # plt.plot(TIME, thr_max, linewidth=2)
        plt.ylabel("Rate [Mbps]", fontsize=18)
        plt.xlabel("Time [s]", fontsize=18)
        plt.legend()
        plt.grid(True)

        plt.subplot(122)
        plt.plot(TIME, mcs_list, marker="x", linewidth=0.2, markersize=7.0, label=p)
        plt.plot(TIME, baseline_mcs, linewidth=2, label='OPT')
        # plt.plot(TIME, mcs_max, linewidth=2)
        plt.ylabel("MCS INDEX", fontsize=18)
        plt.xlabel("Time [s]", fontsize=18)
        plt.legend()
        plt.grid(True)
        fig.suptitle(p, fontsize=16)

        plt.savefig(folder + '/' + p + '.png')

        fig_index += 1

    plt.show()


if __name__ == '__main__':
    main()
