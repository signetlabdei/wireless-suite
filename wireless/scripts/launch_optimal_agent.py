"""
© 2020, University of Padova, Department of Information Engineering, SIGNET Lab.
"""
import os
import gym
import json
import numpy as np
import matplotlib.pyplot as plt
from sacred import Experiment
from wireless.utils.misc import get_mcs_data_rate
from wireless.agents.rate_manager_agents import OptimalAgent
from neptunecontrib.monitoring.sacred import NeptuneObserver

# Load agent parameters
with open('../../config/config_agent.json') as f:
    ac = json.load(f)

ex = Experiment(ac["agent"]["agent_type"])
ex.add_config(ac)

# Configure experiment
with open('../../config/config_sacred.json') as f:
    sc = json.load(f)   # Sacred Configuration
    # observer = NeptuneObserver(api_token=os.environ['NEPTUNE_API_TOKEN'], project_name=sc["sacred"]["project_name"])
    # ex.observers.append(observer)  # uncomment to create and setup a Neptune Observer (Neptune API token needed)
    ex.add_config(sc)


@ex.config
def cfg():
    campaign = "scenarios_v1"
    scenario_list = ["Journal1Lroom_1.csv"]  # List of scenarios for the environment
    snr_history = 1  # The number of past SNR values to consider for the state
    net_timestep = 0.005  # The real network timestep [s]
    obs_duration = 19.15  # Observation duration [s]; 19.15 s is the duration of LRoom scenario
    n_episodes = 1  # The number of episodes to run

@ex.automain
def main(campaign, scenario_list, snr_history, obs_duration, net_timestep, _run):

    env = gym.make("AdLinkAdaptation-v0", campaign=campaign, scenarios_list=scenario_list, obs_duration=obs_duration,
                   snr_history=snr_history, net_timestep=net_timestep)
    agent = OptimalAgent(env.action_space, env.error_model, env.network_timestep, env.amsdu_size)
    log_timestep = _run.config["sacred"]["log_timestep"]  # timestep for logging in Neptune
    reward_t = []
    tot_bits_t = []
    done = False
    state = env.reset()
    action_obs = state["snr"][-1]
    tot_steps = 0
    thr_t = []
    thr_values = []
    selected_mcs_t = []
    thr_step = log_timestep

    while not done:
        action = agent.act(action_obs)
        new_state, reward, done, debug = env.step(action)
        action_obs = debug["future_snr"]
        tot_bits_generated = sum(debug["tx_pkts_list"])
        tot_bits_t.append(tot_bits_generated)
        reward_t.append(reward)
        thr_values.append(reward / net_timestep)
        thr_step -= 1
        if thr_step == 0:
            selected_mcs_t.append(new_state["mcs"])
            thr_t.append(np.mean(thr_values))
            thr_values = []
            thr_step = log_timestep

        old_state = state
        state = new_state
        tot_steps += 1

    env.close()
    if len(thr_values) != 0:
        selected_mcs_t.append(state["mcs"])
        thr_t.append(np.mean(thr_values))

    start_time = net_timestep * (snr_history - 1)
    time = np.arange(start_time, obs_duration, net_timestep * log_timestep)
    avg_data_rate = np.sum(tot_bits_t, dtype=np.int64) / (obs_duration - start_time) / 1e6
    avg_thr = np.sum(reward_t, dtype=np.int64) / (obs_duration - start_time) / 1e6
    print(f"Avg data rate [Mbps]: {avg_data_rate}")
    print(f"Avg throughput [Mbps]: {avg_thr}")
    print(f"Average reward: {np.mean(reward_t)}")
    print(f"Total timesteps: {tot_steps}")

    # Plot results
    mcs_list = np.arange(13)
    label = "Optimal Rate Manager"
    plt.figure(1)
    plt.plot(time, selected_mcs_t, marker="x", linewidth=0.2, markersize=7.0, label="MCS index")
    plt.title("MCS for each time value: Optimal Agent", fontsize=18)
    plt.ylabel("MCS index", fontsize=18)
    plt.xlabel("Time [s]", fontsize=18)
    plt.yticks(mcs_list, fontsize=12)
    plt.legend(loc="lower left", fontsize=12)
    plt.grid()
    plt.show()

    plt.figure(2)
    plt.plot(time, thr_t, linewidth=2.0, label=label)
    plt.title("Scenario throughput for the Optimal Agent", fontsize=18)
    plt.ylabel("Throughput [Mbps]", fontsize=18)
    plt.xlabel("Time [s]", fontsize=18)
    plt.legend(loc="lower left", fontsize=12)
    plt.grid()
    plt.show()
