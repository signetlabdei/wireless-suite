"""
© 2020, University of Padova, Department of Information Engineering, SIGNET Lab.
"""
import gym
import os
import sys
import ast
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from wireless.agents.rate_manager_agents import TabularAgent

# Need the following snippet to avoid problems when launching this script to BLADE cluster
matplotlib.use('agg')
from numpy import savetxt
from gym import wrappers


CAMPAIGN = "scenarios_v1"
SCENARIOS_LIST = ["Journal1ParkingLot_2.csv",
                  "Journal1ParkingLot_3.csv",
                  "Journal1Lroom_1.csv",
                  "Journal1Lroom_3.csv"]
EVAL_SCENARIO = ["Journal1ParkingLot_1.csv"]
EVAL_DURATION = 19.15
DMG_PATH = "/nfsd/signet3/dragomat/wireless-suite/dmg_files"
SNR_HISTORY = 1  # The number of past SNR values to consider for the state
NET_TIMESTEP = 0.005  # The real network timestep [s]
N_EPISODES = 1000
THR_STEP = 20
START_TIME = NET_TIMESTEP * (SNR_HISTORY - 1)


def create_policy(hyperparameters, method='sarsa'):
    hypar = hyperparameters.copy()

    policy_env = gym.make("AdLinkAdaptation-v0",
                          campaign=CAMPAIGN,
                          scenarios_list=SCENARIOS_LIST,
                          obs_duration=hypar['obs_duration'],
                          snr_history=SNR_HISTORY,
                          net_timestep=NET_TIMESTEP,
                          dmg_path=DMG_PATH)

    # Need the following snippet to avoid problems when launching this script to BLADE cluster
    cwd = "wireless/scripts/"+method+"/"
    policy_env = wrappers.Monitor(policy_env, cwd, video_callable=False, force=True)

    epsilon = 1
    gamma = hypar['gamma']
    alpha = hypar['alpha']

    bins = np.arange(-5, 16, 1)
    obs_space_dim, action_space_dim = len(bins) + 1, policy_env.action_space.n

    count_table = np.zeros((obs_space_dim, action_space_dim))
    actions_counting = np.zeros(action_space_dim)
    history_returns = np.zeros(N_EPISODES)
    average_returns = np.zeros(N_EPISODES)

    agent = TabularAgent(obs_space_dim, action_space_dim, method, epsilon=epsilon, gamma=gamma, alpha=alpha)

    print(f"Run for combination: {hypar}")

    for n in range(N_EPISODES):

        cumulative_reward = 0
        observation = policy_env.reset()
        state = np.digitize(observation["snr"][-1], bins)

        agent.update_epsilon(epsilon)
        action = agent.act(state)
        agent.set_state(state)
        agent.set_action(action)

        done = False

        while not done:
            observation, reward, done, debug = policy_env.step(action)
            reward = reward / 1e6
            state = np.digitize(observation["snr"][-1], bins)

            action = agent.act(state)
            agent.train_step(state, action, reward)

            actions_counting[action] += 1
            count_table[state, action] += 1
            cumulative_reward += reward

        history_returns[n] = cumulative_reward
        average_returns[n] = np.mean(history_returns[0:n + 1])

    print("Q-table created!")
    frequency = [x / np.sum(actions_counting) for x in actions_counting]

    policy_env.close()
    policy, _ = agent.generate_policy()
    return policy, agent.get_qtable(), history_returns, average_returns, frequency, count_table


def main():
    folder = "wireless/scripts/"+sys.argv[1]+"/"
    data = pd.read_csv(folder+"search_results.csv")
    bins = np.arange(-5, 16, 1)

    data.sort_values('score', ascending=False, inplace=True)
    data.reset_index(inplace=True)

    print('The highest score was {:.5f} found on iteration {}.'.format(data.loc[0, 'score'], data.loc[0, 'iteration']))

    # Use best hyperparameters to create a model
    hyperparameters = ast.literal_eval(data.loc[0, 'params'])

    policy_sarsa, qtable_sarsa, hReturn_sarsa, avgReturn_sarsa, freq_sarsa, count_sarsa = create_policy(hyperparameters,
                                                                                                        'sarsa')
    savetxt(folder+"qtable_sarsa.csv", qtable_sarsa, delimiter=",")
    savetxt(folder+"avgReturn_sarsa.csv", avgReturn_sarsa, delimiter=",")
    savetxt(folder+"count_sarsa.csv", count_sarsa, delimiter=",")

    policy_exSarsa, qtable_exSarsa, hReturn_exSarsa, avgReturn_exSarsa, freq_exSarsa, count_exSarsa = create_policy(
        hyperparameters, 'exSarsa')
    savetxt(folder+"qtable_exSarsa.csv", qtable_exSarsa, delimiter=",")
    savetxt(folder+"avgReturn_exSarsa.csv", avgReturn_exSarsa, delimiter=",")
    savetxt(folder+"count_exSarsa.csv", count_exSarsa, delimiter=",")

    policy_ql, qtable_ql, hReturn_ql, avgReturn_ql, freq_ql, count_ql = create_policy(hyperparameters, 'q_learning')
    savetxt(folder+"qtable_ql.csv", qtable_ql, delimiter=",")
    savetxt(folder+"avgReturn_ql.csv", avgReturn_ql, delimiter=",")
    savetxt(folder+"count_ql.csv", count_ql, delimiter=",")

    # AVERAGE CUMULATIVE RETURN PLOT COMPARISON

    fig = plt.figure(1, figsize=(14, 14))
    plt.subplot(111)
    plt.plot(avgReturn_sarsa, label="SARSA")
    plt.plot(avgReturn_ql, label="Q-Learning")
    plt.plot(avgReturn_exSarsa, label="Expected SARSA")
    plt.xlabel("# EPISODES")
    plt.ylabel("Average Cumulative Reward")
    plt.title("gamma=0.95 alpha=0.1")
    plt.grid()
    plt.legend()
    plt.savefig(folder+'avgReturn_comparison.png')

    # HEATMAPS PLOT

    fig = plt.figure(2, figsize=(14, 14))
    ax = plt.subplot(111)
    plt.title("SARSA")
    plt.imshow(qtable_sarsa, origin='lower')
    plt.xlabel("MCS")
    plt.ylabel("SINR [dB]")
    ax.tick_params(labelsize=8)
    ax.set_yticks(np.arange(len(bins)))
    ax.set_yticklabels(bins)
    for i in range(len(policy_sarsa)):
        text = ax.text(policy_sarsa[i], i, 'X',
                       ha="center", va="center", color="w")
    plt.colorbar()
    plt.savefig(folder+'ql-sarsa.png', bbox_inches='tight')

    fig = plt.figure(3, figsize=(14, 14))
    ax = plt.subplot(111)
    plt.title("Q_learning")
    plt.imshow(qtable_ql, origin='lower')
    plt.xlabel("MCS")
    plt.ylabel("SINR [dB]")
    ax.tick_params(labelsize=8)
    ax.set_yticks(np.arange(len(bins)))
    ax.set_yticks(np.arange(len(bins)))
    ax.set_yticklabels(bins)
    for i in range(len(policy_ql)):
        text = ax.text(policy_ql[i], i, 'X',
                       ha="center", va="center", color="w")
    plt.colorbar()
    plt.savefig(folder+'ql-heatmap.png', bbox_inches='tight')

    fig = plt.figure(8, figsize=(14, 14))
    ax = plt.subplot(111)
    plt.title("Expected SARSA")
    plt.imshow(qtable_exSarsa, origin='lower')
    plt.xlabel("MCS")
    plt.ylabel("SINR [dB]")
    ax.tick_params(labelsize=8)
    ax.set_yticks(np.arange(len(bins)))
    ax.set_yticks(np.arange(len(bins)))
    ax.set_yticklabels(bins)
    for i in range(len(policy_exSarsa)):
        text = ax.text(policy_exSarsa[i], i, 'X',
                       ha="center", va="center", color="w")
    plt.colorbar()
    plt.savefig(folder+'ex-heatmap.png', bbox_inches='tight')

    # ACTIONS FREQUENCY COMPARISON

    fig = plt.figure(4, figsize=(14, 14))
    markerline1, stemlines, _ = plt.stem(
        freq_sarsa, label='SARSA', use_line_collection=True)
    plt.setp(markerline1, 'markerfacecolor', 'b')
    markerline2, stemlines, _ = plt.stem(
        freq_ql, label='Q-learning', use_line_collection=True)
    plt.setp(markerline2, 'markerfacecolor', 'r')
    markerline3, stemlines, _ = plt.stem(
        freq_exSarsa, label='Expected SARSA', use_line_collection=True)
    plt.setp(markerline3, 'markerfacecolor', 'g')
    plt.xlabel('MCS')
    plt.ylabel('Frequency')
    plt.title('Actions frequency')
    plt.grid(True)
    plt.legend()
    plt.savefig(folder+'act-freq.png')

    # ACTIONS COUNTING HEATMAP

    fig = plt.figure(5, figsize=(14, 14))
    ax = plt.subplot(111)
    plt.title("SARSA")
    plt.imshow(count_sarsa, origin='lower')
    plt.xlabel("MCS")
    plt.ylabel("SINR [dB]")
    ax.tick_params(labelsize=8)
    ax.set_yticks(np.arange(len(bins)))
    ax.set_yticklabels(bins)
    for j in range(len(count_sarsa[0, :])):
        for i in range(len(count_sarsa[:, 0])):
            if count_sarsa[i, j] > 0:
                text = ax.text(j, i, np.int64(count_sarsa[i, j]),
                               ha="center", va="center", color="w", size=5)
    plt.colorbar()
    plt.savefig(folder+'count-sarsa.png', bbox_inches='tight')

    fig = plt.figure(6, figsize=(14, 14))
    ax = plt.subplot(111)
    plt.title("Q_learning")
    plt.imshow(count_ql, origin='lower')
    plt.xlabel("MCS")
    plt.ylabel("SINR [dB]")
    ax.tick_params(labelsize=8)
    ax.set_yticks(np.arange(len(bins)))
    ax.set_yticklabels(bins)
    for j in range(len(count_ql[0, :])):
        for i in range(len(count_ql[:, 0])):
            if count_ql[i, j] > 0:
                text = ax.text(j, i, np.int64(count_ql[i, j]),
                               ha="center", va="center", color="w", size=5)
    plt.colorbar()
    plt.savefig(folder+'count-ql.png', bbox_inches='tight')

    fig = plt.figure(7, figsize=(14, 14))
    ax = plt.subplot(111)
    plt.title("Expected SARSA")
    plt.imshow(count_exSarsa, origin='lower')
    plt.xlabel("MCS")
    plt.ylabel("SINR [dB]")
    ax.tick_params(labelsize=8)
    ax.set_yticks(np.arange(len(bins)))
    ax.set_yticklabels(bins)
    for j in range(len(count_exSarsa[0, :])):
        for i in range(len(count_exSarsa[:, 0])):
            if count_exSarsa[i, j] > 0:
                text = ax.text(j, i, np.int64(count_exSarsa[i, j]),
                               ha="center", va="center", color="w", size=5)
    plt.colorbar()
    plt.savefig(folder+'count-exSarsa.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
