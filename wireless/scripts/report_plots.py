"""
© 2020, University of Padova, Department of Information Engineering, SIGNET Lab.
"""

import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt


def generate_policy(q_table):
    dimensions = q_table.shape
    policy = [q_table[i, :].argmax() for i in range(dimensions[0])]
    value = q_table.max(axis=1)
    return policy, value


def main():
    qtable_sarsa = loadtxt("sarsa/qtable_sarsa.csv", delimiter=",")
    qtable_ql = loadtxt("sarsa/qtable_ql.csv", delimiter=",")
    qtable_exSarsa = loadtxt("sarsa/qtable_exSarsa.csv", delimiter=",")

    avgReturn_sarsa = loadtxt("sarsa/avgReturn_sarsa.csv", delimiter=",")
    avgReturn_ql = loadtxt("sarsa/avgReturn_ql.csv", delimiter=",")
    avgReturn_exSarsa = loadtxt("sarsa/avgReturn_exSarsa.csv", delimiter=",")

    policy_sarsa, _ = generate_policy(qtable_sarsa)
    policy_ql, _ = generate_policy(qtable_ql)
    policy_exSarsa, _ = generate_policy(qtable_exSarsa)

    bins = np.arange(-5, 16, 1)

    # AVERAGE CUMULATIVE RETURN PLOT COMPARISON

    fig = plt.figure(1, figsize=(14, 14))
    ax = plt.subplot(111)
    plt.plot(avgReturn_sarsa, label="SARSA")
    plt.plot(avgReturn_ql, label="Q-Learning")
    plt.plot(avgReturn_exSarsa, label="Expected SARSA")
    plt.xlabel("# EPISODES", fontsize=20)
    plt.ylabel("Average Cumulative Reward", fontsize=20)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    plt.grid()
    plt.legend(fontsize=20)
    plt.savefig('avgReturn_comparison.png')

    # HEATMAPS PLOT

    fig = plt.figure(2, figsize=(14, 14))
    ax = plt.subplot(111)
    # plt.title("SARSA")
    plt.imshow(qtable_sarsa, origin='lower')
    plt.xlabel("MCS")
    plt.ylabel("SINR [dB]")
    y_ticks_lab = []
    for i in range(-5, 16, 1):
        y_ticks_lab.append(str(i))
    y_ticks_lab.append('> 15')
    plt.xlabel("MCS", fontsize=20)
    plt.ylabel("SINR [dB]", fontsize=20)
    ax.tick_params(labelsize=8)
    ax.set_xticks(np.arange(0, 13, 1))
    ax.set_xticklabels(np.arange(0, 13, 1), fontsize=16)
    ax.set_yticks(np.arange(len(bins)+1))
    ax.set_yticklabels(y_ticks_lab, fontsize=16)
    for i in range(len(policy_sarsa)):
        text = ax.text(policy_sarsa[i], i, 'X',
                       ha="center", va="center", color="w")
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    plt.savefig('sarsa-heatmap.png', bbox_inches='tight')

    fig = plt.figure(3, figsize=(14, 14))
    ax = plt.subplot(111)
    # plt.title("Q_learning")
    plt.imshow(qtable_ql, origin='lower')
    plt.xlabel("MCS", fontsize=20)
    plt.ylabel("SINR [dB]", fontsize=20)
    ax.set_xticks(np.arange(0, 13, 1))
    ax.set_xticklabels(np.arange(0, 13, 1), fontsize=16)
    ax.set_yticks(np.arange(len(bins)+1))
    ax.set_yticklabels(y_ticks_lab, fontsize=16)
    for i in range(len(policy_ql)):
        text = ax.text(policy_ql[i], i, 'X',
                       ha="center", va="center", color="w")
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    plt.savefig('ql-heatmap.png', bbox_inches='tight')

    fig = plt.figure(4, figsize=(14, 14))
    ax = plt.subplot(111)
    # plt.title("Expected SARSA")
    plt.imshow(qtable_exSarsa, origin='lower')
    plt.xlabel("MCS", fontsize=20)
    plt.ylabel("SINR [dB]", fontsize=20)
    ax.set_xticks(np.arange(0, 13, 1))
    ax.set_xticklabels(np.arange(0, 13, 1), fontsize=16)
    ax.set_yticks(np.arange(len(bins) + 1))
    ax.set_yticklabels(y_ticks_lab, fontsize=16)
    for i in range(len(policy_exSarsa)):
        text = ax.text(policy_exSarsa[i], i, 'X',
                       ha="center", va="center", color="w")
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    plt.savefig('exSarsa-heatmap.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
