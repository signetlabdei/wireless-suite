"""
© 2020, University of Padova, Department of Information Engineering, SIGNET Lab.
"""
import gym
import os
import numpy as np
import pandas as pd
import random
import matplotlib
import pathlib
from gym import wrappers
from wireless.scripts.create_policies import create_policy

# Need the following snippet to avoid problems when launching this script to BLADE cluster
matplotlib.use('agg')
random.seed()

MAX_EVALS = 10
N_EPISODES = 1
CAMPAIGN = "scenarios_v3"
SCENARIOS_LIST = ["Journal1ParkingLot_2.csv",
                  "Journal1ParkingLot_3.csv",
                  "Journal1Lroom_1.csv",
                  "Journal1Lroom_3.csv"]
EVAL_SCENARIO = ["Journal1ParkingLot_1.csv"]
EVAL_DURATION = None
DMG_PATH = "../../dmg_files/"  # use this path when running locally
# DMG_PATH = "/nfsd/signet3/dragomat/wireless-suite/dmg_files"  # use this path when running on cluster
SNR_HISTORY = 1  # The number of past SNR values to consider for the state
NET_TIMESTEP = 0.005  # The real network timestep [s]
THR_STEP = 20
START_TIME = NET_TIMESTEP * (SNR_HISTORY - 1)
OUTPUT_FOLDER = "random_search_output/"
BINS = np.arange(-5, 16, 1)

pathlib.Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
OUTPUT_FILENAME = "random_search_results_" + str(random.randrange(1e6)) + ".csv"


def evaluate_policy(policy):
    env = gym.make("AdLinkAdaptation-v0",
                   campaign=CAMPAIGN,
                   scenarios_list=EVAL_SCENARIO,
                   obs_duration=EVAL_DURATION,
                   snr_history=SNR_HISTORY,
                   net_timestep=NET_TIMESTEP)

    # temp = pathlib.Path(cwd + '/monitor/' + str(random.randrange(1e6)))  # use this path when running on cluster
    temp = os.getcwd()  # use this path when running locally
    env = wrappers.Monitor(env, temp, video_callable=False, force=True)

    reward_t = []
    done = False
    observation = env.reset()

    while not done:
        action = policy[np.digitize(observation["snr"][-1], BINS)]
        next_observation, reward, done, debug = env.step(action)
        reward_t.append(reward)
        observation = next_observation

    env.close()

    avg_thr = np.sum(reward_t, dtype=np.int64) / (env.current_obs_duration - START_TIME) / 1e6

    return avg_thr


def random_search(param_grid, max_evals=MAX_EVALS):
    """Random search for hyperparameter optimization"""

    # Dataframe for results
    results = pd.DataFrame(columns=['score', 'params', 'policy', 'q_table', 'iteration'],
                           index=list(range(max_evals)))

    # Keep searching until reach max evaluations
    for i in range(max_evals):
        # Choose random hyperparameters

        hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        params_tosave = hyperparameters.copy()

        hyperparameters['epsilon'] = 1.0
        hyperparameters['bins'] = BINS

        # Evaluate randomly selected hyperparameters
        q_table, policy, _ = create_policy(hyperparameters,
                                           method="sarsa",
                                           eps_update="e-greedy",
                                           state_space=None,
                                           n_episodes=N_EPISODES)
        evaluation = evaluate_policy(policy)

        results.loc[i, :] = [evaluation, params_tosave, policy, q_table, i]
        results.to_csv(OUTPUT_FOLDER + OUTPUT_FILENAME, index=False)

    return results


def main():
    # parameters grid
    param_grid = {
        'obs_duration': list(np.around(np.linspace(2.0, 8.0, 14), decimals=4)),
        'alpha': list(np.around(np.logspace(np.log10(0.005), np.log10(0.5), base=10, num=20), decimals=4)),
        'gamma': list(np.around(np.linspace(0.8, 0.99, 20), decimals=4)),
    }

    random_results = random_search(param_grid)
    random_results.to_csv(OUTPUT_FOLDER + OUTPUT_FILENAME, index=False)
    print(f"Best policy: {random_results.loc[0, 'policy']}")


if __name__ == '__main__':
    main()
