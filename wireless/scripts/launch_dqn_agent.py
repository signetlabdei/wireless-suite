"""
© 2020, University of Padova, Department of Information Engineering, SIGNET Lab.
"""
import os
import gym
import json
import numpy as np
import matplotlib.pyplot as plt
from sacred import Experiment
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from wireless.utils.misc import import_scenario, get_mcs_data_rate
from wireless.agents.rate_manager_agents import DQNAgent
from neptunecontrib.monitoring.sacred import NeptuneObserver

ex_name = "DQN-scaled-state-v2"
ex = Experiment(ex_name)

# Configure experiment
with open('../../config/config_sacred.json') as f:
    sc = json.load(f)   # Sacred Configuration
    observer = NeptuneObserver(api_token=os.environ['NEPTUNE_API_TOKEN'], project_name=sc["sacred"]["project_name"])
    ex.observers.append(observer)  # uncomment to create a Neptune Observer (Neptune API token needed)
    ex.add_config(sc)

@ex.config
def cfg():
    campaign = "scenarios_v1"
    scenario_list_train = None  # List of scenarios of the training phase (If None, all scenarios in the campaign
    # folder are considered)
    scenario_list_test = ["Journal1Lroom_1.csv"]  # Scenario for the evaluation phase
    snr_history = 11  # The number of past SNR values to consider for the state
    net_timestep = 0.005  # The real network timestep [s]
    obs_duration = 8  # Observation duration [s];
    n_episodes = 1000  # The number of episodes to run
    hidden_units = [24, 24]  # The number of units for each hidden layer
    memory_size = 10000
    batch_size = 32
    target_update = 50
    lr = 0.001
    model_path = ex_name
    epsilon = 1.0
    decay = 0.9995
    min_epsilon = 0.1
    train = True

def resize_reward(reward):
    return reward / 1e6

def scale_state(observation, scaler):
    snr = np.atleast_2d(observation["snr"])
    scaled_snr = scaler.transform(snr)[0]
    return np.append(scaled_snr, [observation["mcs"] / 12.0])

def compute_scaler(env):
    scenarios_list = [file for file in os.listdir(env.qd_scenarios_path) if file.endswith(".csv")]
    snr_list = []
    for scn in scenarios_list:
        filepath = os.path.join(env.qd_scenarios_path, scn)
        df = import_scenario(filepath)
        snr_list.append(df['SNR'].tolist())
    snr_list = np.concatenate(snr_list, axis=0)
    scaler = StandardScaler()
    scaler.fit(snr_list.reshape(-1, 1))
    return scaler

def run_episode(env, agent, epsilon, scaler, log_timestep):
    step = 0
    reward_t = []
    loss_t = []
    done = False
    obs = env.reset()
    state = scale_state(obs, scaler)

    while not done:
        action = agent.act(state, epsilon)
        obs, reward, done, _ = env.step(action)
        new_state = scale_state(obs, scaler)  # Select the SNR history from the OBS returned by the environment
        reward = resize_reward(reward)  # Obtain reward in Mbps
        reward_t.append(reward)

        agent.add_experience(state, action, reward, new_state, done)
        state = new_state
        loss = agent.train()
        loss_t .append(loss)
        step += 1
        # update target model
        if step % agent.target_update == 0:
            agent.copy_weights()

    return np.sum(reward_t), np.sum(loss_t), np.mean(reward_t), np.mean(loss_t)

@ex.automain
def main(campaign, scenario_list_train, scenario_list_test, snr_history, obs_duration, net_timestep, hidden_units,
         n_episodes, epsilon, decay, min_epsilon, train, model_path, batch_size, memory_size, target_update, lr, _run):

    log_timestep = _run.config["sacred"]["log_timestep"]  # timestep for logging in Neptune
    scaler = None
    if train:
        env = gym.make("AdLinkAdaptation-v0", campaign=campaign, scenarios_list=scenario_list_train,
                       obs_duration=obs_duration, snr_history=snr_history, net_timestep=net_timestep)
        agent = DQNAgent(env.action_space, snr_history+1, hidden_units, memory_size=memory_size,
                         batch_size=batch_size, target_update=target_update, lr=lr)
        agent.seed()
        scaler = compute_scaler(env)
        reward_ep = np.empty(n_episodes)
        avg_reward_ep = np.empty(n_episodes)
        loss_ep = np.empty(n_episodes)

        for ep in range(n_episodes):
            ep_cum_rwd, ep_cum_loss, _, _ = run_episode(env, agent, epsilon, scaler, log_timestep)
            reward_ep[ep] = ep_cum_rwd
            avg_reward = np.mean(reward_ep[0:ep+1])
            avg_reward_ep[ep] = avg_reward
            loss_ep[ep] = ep_cum_loss

            # Log training information through sacred
            _run.log_scalar("Loss vs episode", ep_cum_loss, ep)
            _run.log_scalar("Reward vs episode", ep_cum_rwd, ep)
            _run.log_scalar("Average reward vs episode", avg_reward, ep)
            _run.log_scalar("Epsilon vs episode", epsilon, ep)

            # print intermediate results
            if ep % 10 == 0:
                print(f"episode: {ep}, episode reward: {ep_cum_rwd}, episode loss: {ep_cum_loss}, "
                      f"avg reward: {avg_reward}, epsilon: {epsilon}")
            # save model every 100 episodes
            if ep % 100 == 0:
                agent.save(model_path)
            # update epsilon decay
            epsilon = max(min_epsilon, epsilon * decay)
            # update target model at the end of the episode
            agent.copy_weights()

        print(f"Training ended after {n_episodes} episodes")
        # save model
        agent.save(model_path)
        env.close()

    # Evaluate the agent in a specific scenario
    if scenario_list_test is not None:
        # At this time, only one scenario must be provided for the testing phase
        assert len(scenario_list_test) == 1, "Testing occurs over one scenario"
        env = gym.make("AdLinkAdaptation-v0", campaign=campaign, scenarios_list=scenario_list_test,
                       snr_history=snr_history, net_timestep=net_timestep)
        agent = DQNAgent(env.action_space, snr_history+1, hidden_units, memory_size=memory_size, batch_size=batch_size,
                         target_update=target_update, lr=lr)
        agent.seed()
        agent.load(model_path)
        if scaler is None:
            scaler = compute_scaler(env)
        done = False
        tot_steps = 0
        tot_bits_t = []
        selected_mcs_t = []
        reward_t = []
        thr_t = []
        thr_values = []
        thr_step = log_timestep
        obs = env.reset()
        state = scale_state(obs, scaler)
        scenario_duration = env.scenario_duration

        while not done:
            action = agent.act(state, 0.0)  # epsilon=0, Only exploiting
            obs, reward, done, debug = env.step(action)
            new_state = scale_state(obs, scaler)
            tot_bits_generated = sum(debug["tx_pkts_list"])
            tot_bits_t.append(tot_bits_generated)
            reward_t.append(reward)
            thr_values.append(reward / net_timestep)
            thr_step -= 1
            if thr_step == 0:
                selected_mcs_t.append(action)
                thr_t.append(np.mean(thr_values))
                thr_values = []
                thr_step = log_timestep

            state = new_state
            tot_steps += 1

        env.close()
        if len(thr_values) != 0:
            selected_mcs_t.append(state[2])
            thr_t.append(np.mean(thr_values))

        start_time = net_timestep * (snr_history - 1)
        time = np.arange(start_time, scenario_duration, net_timestep * log_timestep)
        avg_data_rate = np.sum(tot_bits_t, dtype=np.int64) / (scenario_duration - start_time) / 1e6
        avg_thr = np.sum(reward_t, dtype=np.int64) / (scenario_duration - start_time) / 1e6
        print(f"Avg data rate [Mbps]: {avg_data_rate}")
        print(f"Avg throughput [Mbps]: {avg_thr}")
        print(f"Average reward: {np.mean(reward_t)}")
        print(f"Total timesteps: {tot_steps}")

        # Plot results
        mcs_list = np.arange(13)
        label = "DQN Rate Manager"
        plt.figure(1)
        plt.plot(time, selected_mcs_t, marker="x", linewidth=0.2, markersize=7.0, label="MCS index")
        plt.title("MCS for each time value: DQN Agent", fontsize=18)
        plt.ylabel("MCS index", fontsize=18)
        plt.xlabel("Time [s]", fontsize=18)
        plt.yticks(mcs_list, fontsize=12)
        plt.legend(loc="lower left", fontsize=12)
        plt.grid()
        plt.show()

        plt.figure(2)
        plt.plot(time, thr_t, linewidth=2.0, label=label)
        plt.title("Scenario throughput for the DQN Agent", fontsize=18)
        plt.ylabel("Throughput [Mbps]", fontsize=18)
        plt.xlabel("Time [s]", fontsize=18)
        plt.legend(loc="lower left", fontsize=12)
        plt.grid()
        plt.show()
