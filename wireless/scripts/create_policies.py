"""
© 2020, University of Padova, Department of Information Engineering, SIGNET Lab.
"""
import os
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from gym import wrappers
from numpy import savetxt
from sklearn import linear_model
from wireless.agents.rate_manager_agents import TabularAgent

# Need the following snippet to avoid problems when launching this script to BLADE cluster
matplotlib.use("agg")

CAMPAIGN = "scenarios_v3"
SCENARIOS_LIST = ["Journal1ParkingLot_2.csv",
                  "Journal1ParkingLot_3.csv",
                  "Journal1Lroom_1.csv",
                  "Journal1Lroom_3.csv"]
EVAL_SCENARIO = ["Journal1ParkingLot_1.csv"]
EVAL_DURATION = None
DMG_PATH = "../../dmg_files/"  # use this path when running locally
# DMG_PATH = "/nfsd/signet3/dragomat/wireless-suite/dmg_files"  # use this path when running on cluster
SNR_HISTORY = 2  # The number of past SNR values to consider for the state
NET_TIMESTEP = 0.005  # The real network timestep [s]
N_EPISODES = 10
THR_STEP = 20
START_TIME = NET_TIMESTEP * (SNR_HISTORY - 1)


def create_policy(hyperparameters,
                  method="sarsa",
                  eps_update="e-greedy",
                  state_space=None,
                  n_episodes=N_EPISODES):
    hypar = hyperparameters.copy()

    policy_env = gym.make("AdLinkAdaptation-v0",
                          campaign=CAMPAIGN,
                          scenarios_list=SCENARIOS_LIST,
                          obs_duration=hypar["obs_duration"],
                          snr_history=SNR_HISTORY,
                          net_timestep=NET_TIMESTEP,
                          dmg_path=DMG_PATH)

    # Need the following snippet to avoid problems when launching this script to BLADE cluster
    # temp = pathlib.Path(cwd + '/monitor/' + str(random.randrange(1e6)))  # use this path when running on cluster
    temp = os.getcwd()  # use this path when running locally
    policy_env = wrappers.Monitor(policy_env, temp, video_callable=False, force=True)

    # initialization of hyperparameters
    epsilon = hypar["epsilon"]
    gamma = hypar["gamma"]
    alpha = hypar["alpha"]
    bins = hypar["bins"]
    x = np.arange(0, NET_TIMESTEP * SNR_HISTORY, NET_TIMESTEP)
    regr = None

    if state_space == "v2":
        if SNR_HISTORY <= 1:
            ValueError("Impossible to do regression on a single sample of SNR history")
        regr = linear_model.LinearRegression()
        obs_space_dim, action_space_dim = 2 * (len(bins) + 1), policy_env.action_space.n
    else:
        obs_space_dim, action_space_dim = len(bins) + 1, policy_env.action_space.n

    # instances for debug purposes
    count_table = np.zeros((obs_space_dim, action_space_dim))
    actions_counting = np.zeros(action_space_dim)
    history_returns = np.zeros(n_episodes)
    average_returns = np.zeros(n_episodes)

    print(f"Starting {method} with n_episodes={n_episodes}")

    agent = TabularAgent(obs_space_dim,
                         action_space_dim,
                         method,
                         epsilon=epsilon,
                         gamma=gamma,
                         alpha=alpha,
                         eps_update=eps_update)

    for n in range(n_episodes):
        ep_cumulative_reward = 0
        observation = policy_env.reset()

        if state_space == "v2":
            regr.fit(x.reshape(-1, 1), observation["snr"])
            if regr.coef_ > 0:
                state = np.digitize(observation["snr"][-1], bins) + len(bins) + 1
            else:
                state = np.digitize(observation["snr"][-1], bins)
        else:
            state = np.digitize(observation["snr"][-1], bins)

        if eps_update == "e-greedy":
            agent.update_epsilon()

        action = agent.act(state)
        agent.set_state(state)
        agent.set_action(action)

        actions_counting[action] += 1
        count_table[state, action] += 1

        done = False

        while not done:
            observation, reward, done, debug = policy_env.step(action)
            reward = reward / 1e6
            if state_space == "v2":
                regr.fit(x.reshape(-1, 1), observation["snr"])
                if regr.coef_ > 0:
                    state = np.digitize(observation["snr"][-1], bins) + len(bins) + 1
                else:
                    state = np.digitize(observation["snr"][-1], bins)
            else:
                state = np.digitize(observation["snr"][-1], bins)

            action = agent.act(state)
            agent.train_step(state, action, reward)

            ep_cumulative_reward += reward
            actions_counting[action] += 1
            count_table[state, action] += 1

        history_returns[n] = ep_cumulative_reward
        average_returns[n] = np.mean(history_returns[0:n + 1])
        if n % 100 == 0:
            print(f"completed episode: {n}, avg reward: {average_returns[n]}")

    print(f"{method} qtable created!")
    tot_actions = np.sum(actions_counting)
    frequency = [x / tot_actions for x in actions_counting]

    policy_env.close()
    policy, _ = agent.generate_policy()
    info = {
        "history_returns": history_returns,
        "average_returns": average_returns,
        "frequency": frequency,
        "count_table": count_table,
        "epsilon_values": agent.epsilon_values
    }

    return agent.get_qtable(), policy, info


def main():
    bins = np.arange(-5, 16, 1)
    param_grid = {
        "obs_duration": 8.0,
        "epsilon": 1.0,
        "alpha": 0.01,
        "gamma": 0.95,
        "bins": bins
    }
    greedy_update = "vdbe"
    state_space = ""

    qtable_sarsa, policy_sarsa, info_sarsa = create_policy(param_grid, "sarsa", greedy_update, state_space)
    savetxt("qtable_sarsa.csv", qtable_sarsa, delimiter=",")
    savetxt("avgReturn_sarsa.csv", info_sarsa["average_returns"], delimiter=",")
    savetxt("count_sarsa.csv", info_sarsa["count_table"], delimiter=",")

    qtable_ql, policy_ql, info_ql = create_policy(param_grid, "q_learning", greedy_update, state_space)
    savetxt("qtable_ql.csv", qtable_ql, delimiter=",")
    savetxt("avgReturn_ql.csv", info_ql["average_returns"], delimiter=",")
    savetxt("count_ql.csv", info_ql["count_table"], delimiter=",")

    # AVERAGE CUMULATIVE RETURN PLOT COMPARISON

    fig = plt.figure(1, figsize=(14, 14))
    plt.subplot(111)
    plt.plot(info_sarsa["average_returns"], label="SARSA")
    plt.plot(info_ql["average_returns"], label="Q-Learning")
    plt.xlabel("# EPISODES")
    plt.ylabel("Average Cumulative Reward")
    plt.grid()
    plt.legend()
    plt.savefig("avgReturn_comparison.png")

    # HEATMAPS PLOT

    fig = plt.figure(2, figsize=(14, 14))
    ax = plt.subplot(111)
    plt.title("SARSA")
    plt.imshow(qtable_sarsa, origin="lower")
    plt.xlabel("MCS")
    plt.ylabel("SINR [dB]")
    ax.tick_params(labelsize=8)
    ax.set_yticks(np.arange(len(bins)))
    ax.set_yticklabels(bins)
    for i in range(len(policy_sarsa)):
        text = ax.text(policy_sarsa[i], i, "X",
                       ha="center", va="center", color="w")
    plt.colorbar()
    plt.savefig("sarsa-heatmap.png", bbox_inches="tight")

    fig = plt.figure(3, figsize=(14, 14))
    ax = plt.subplot(111)
    plt.title("Q_learning")
    plt.imshow(qtable_ql, origin="lower")
    plt.xlabel("MCS")
    plt.ylabel("SINR [dB]")
    ax.tick_params(labelsize=8)
    ax.set_yticks(np.arange(len(bins)))
    ax.set_yticks(np.arange(len(bins)))
    ax.set_yticklabels(bins)
    for i in range(len(policy_ql)):
        text = ax.text(policy_ql[i], i, "X",
                       ha="center", va="center", color="w")
    plt.colorbar()
    plt.savefig("ql-heatmap.png", bbox_inches="tight")

    # ACTIONS COUNTING HEATMAP

    fig = plt.figure(5, figsize=(14, 14))
    ax = plt.subplot(111)
    plt.title("SARSA")
    plt.imshow(info_sarsa["count_table"], origin='lower')
    plt.xlabel("MCS")
    plt.ylabel("SINR [dB]")
    ax.tick_params(labelsize=8)
    ax.set_yticks(np.arange(len(bins)))
    ax.set_yticklabels(bins)
    for j in range(len(info_sarsa["count_table"][0, :])):
        for i in range(len(info_sarsa["count_table"][:, 0])):
            if info_sarsa["count_table"][i, j] > 0:
                text = ax.text(j, i, np.int64(info_sarsa["count_table"][i, j]),
                               ha="center", va="center", color="w", size=5)
    plt.colorbar()
    plt.savefig("count-sarsa.png", bbox_inches="tight")

    fig = plt.figure(6, figsize=(14, 14))
    ax = plt.subplot(111)
    plt.title("Q_learning")
    plt.imshow(info_ql["count_table"], origin="lower")
    plt.xlabel("MCS")
    plt.ylabel("SINR [dB]")
    ax.tick_params(labelsize=8)
    ax.set_yticks(np.arange(len(bins)))
    ax.set_yticklabels(bins)
    for j in range(len(info_ql["count_table"][0, :])):
        for i in range(len(info_ql["count_table"][:, 0])):
            if info_ql["count_table"][i, j] > 0:
                text = ax.text(j, i, np.int64(info_ql["count_table"][i, j]),
                               ha="center", va="center", color="w", size=5)
    plt.colorbar()
    plt.savefig("count-ql.png", bbox_inches="tight")

    '''fig = plt.figure(2, figsize=(14, 14))
    markerline1, stemlines, _ = plt.stem(
        freq_sarsa, label='SARSA', use_line_collection=True)
    plt.setp(markerline1, 'markerfacecolor', 'b')
    markerline2, stemlines, _ = plt.stem(
        freq_ql, label='Q-learning', use_line_collection=True)
    plt.setp(markerline2, 'markerfacecolor', 'r')
    plt.xlabel('MCS')
    plt.ylabel('Frequency')
    plt.title('Actions frequency')
    plt.grid(True)
    plt.legend()
    plt.savefig('act-freq.png')'''

    # EPSILON VALUE RETURN PLOT COMPARISON

    if greedy_update == "e-greedy":
        fig = plt.figure(7, figsize=(14, 14))
        plt.subplot(111)
        plt.plot(info_sarsa["epsilon_values"], label="e-greedy")
        plt.xlabel("# EPISODES")
        plt.ylabel("Epsilon")
        plt.grid()
        plt.legend()
        plt.savefig("e-greedy.png")
    elif greedy_update == "vdbe":
        # plot subset of states [from low SNR (less visited) to high SNR (most visited)
        subset_states = [0, 2, 5, 21]
        fig = plt.figure(7, figsize=(14, 14))
        plt.subplot(111)
        for state in subset_states:
            plt.plot(info_sarsa["epsilon_values"][state], label=str(state))
        plt.xlabel("Training steps")
        plt.ylabel("Epsilon")
        plt.grid()
        plt.legend()
        plt.savefig("vdbe-sarsa.png")

        fig = plt.figure(8, figsize=(14, 14))
        plt.subplot(111)
        for state in subset_states:
            plt.plot(info_ql["epsilon_values"][state], label=str(state))
        plt.xlabel("Training steps")
        plt.ylabel("Epsilon")
        plt.grid()
        plt.legend()
        plt.savefig("vdbe-ql.png")


if __name__ == '__main__':
    main()
