import numpy as np
from tqdm.auto import tqdm
import nolds
import pandas as pd
from learning_in_games.games import braess_augmented_network
from learning_in_games.running import *
from learning_in_games.agents import *
from pathlib import Path
from learning_in_games import utilities
import math
from dataclasses import dataclass
from typing import Union


@dataclass
class AlphaExperimentConfig:
    user0_alpha: float
    alpha_expectation: float
    alpha_variance: float
    n_agents: int
    n_states: int
    n_actions: int
    n_iter: int
    epsilon: Union[float, str]
    gamma: float
    q_initial: Union[np.ndarray, str]
    qmin: float
    qmax: float
    path: str
    name: str
    repetitions: int


def run_game(config: AlphaExperimentConfig):
    Q = initialize_q_table(config.q_initial, config.n_agents, config.n_states, config.n_actions, config.qmin, config.qmax)
    alpha = np.random.uniform(
        low=config.alpha_expectation-(config.alpha_variance/2),
        high=config.alpha_expectation+(config.alpha_variance/2),
        size=config.n_agents
    )
    alpha[0] = config.user0_alpha
    eps_decay = config.n_iter / 8
    if config.epsilon == "DECAYED":
        eps_start = 1
        eps_end = 0
    else:
        eps_start = config.epsilon
        eps_end = config.epsilon

    all_agent_indices = np.arange(config.n_agents)
    S = np.random.randint(config.n_states, size=config.n_agents)

    data = {}
    for t in range(config.n_iter):
        epsilon = (eps_end + (eps_start - eps_end) * math.exp(-1. * t / eps_decay))  # if t < N_ITER/10 else 0
        A = e_greedy_select_action(Q, S, epsilon)
        R, _, _ = braess_augmented_network(A, config.n_agents, cost=0)

        Q, sum_of_belief_updates = bellman_update_q_table(all_agent_indices, Q, S, A, R, S, alpha, config.gamma)

        ## SAVE PROGRESS DATA
        data[t] = {
                   "R": R,
                   "Qmean": Q.mean(axis=1).mean(axis=0),
                   # "groups": count_groups(Q[ind, S, :], 0.1),
                   "Qvar": Q[all_agent_indices, S, :].var(axis=0),
                   # "A": A,
                   "Q": Q,
                   }
    return data


def main(config: AlphaExperimentConfig):
    all_repetitions = []
    for i in range(config.repetitions):
        M = run_game(config)
        full_path = f"{config.path}/{config.name}"
        Path(full_path).mkdir(parents=True, exist_ok=True)
        all_q_tables = np.stack([M[t]["Q"] for t in M.keys()])
        utilities.save_numpy_array_with_unique_filename(all_q_tables, f"{full_path}/q_tables_{i}.npy")
        all_rewards = np.stack([M[t]["R"] for t in M.keys()])
        utilities.save_numpy_array_with_unique_filename(all_rewards, f"{full_path}/rewards_{i}.npy")
        # all_actions = np.stack([M[t]["A"] for t in M.keys()])
        # utilities.save_numpy_array_with_unique_filename(all_actions, f"{path}/{experiment_name}/actions.npy")

        exclusion_threshold = 0.8
        W = np.array([M[t]["R"].mean() for t in range(0, config.n_iter)])
        # L = nolds.lyap_r(W)
        T = np.mean(W[int(exclusion_threshold * config.n_iter):config.n_iter])
        T_all = np.mean(W)
        median = np.median(W[int(exclusion_threshold * config.n_iter):config.n_iter])
        T_std = np.std(W[int(exclusion_threshold * config.n_iter):config.n_iter])

        user0 = np.array([-M[t]["R"][0].mean() for t in M.keys()])
        deviation_gain = W - user0

        # groups = [M[t]["groups"] for t in range(0, config.n_iter)]
        # groups_mean = np.mean(groups)
        # groups_var = np.var(groups)
        Qvar = [M[t]["Qvar"] for t in range(0, config.n_iter)]
        Qvar_mean = np.mean(Qvar)

        row = {
            "repetition": i,
            "n_agents": config.n_agents,
            "alpha": config.user0_alpha,
            "alpha_expectation": config.alpha_expectation,
            "alpha_variance": config.alpha_variance,
            "epsilon": config.epsilon,
            "deviation_gain": deviation_gain.mean(),
            "T_mean": T,
            "T_mean_all": T_all,
            "T_std": T_std,
            # "Lyapunov": L,
            "median": median,
            "Qvar_mean": Qvar_mean,
        }
        all_repetitions.append(row)

    return all_repetitions


def run_apply_async_multiprocessing(func, argument_list, num_processes):
    pool = mp.Pool(processes=num_processes)

    jobs = [
        pool.apply_async(func=func, args=(*argument,)) if isinstance(argument, tuple) else pool.apply_async(func=func,
                                                                                                            args=(
                                                                                                            argument,))
        for argument in argument_list]
    pool.close()
    result_list_tqdm = []
    for job in tqdm(jobs):
        result_list_tqdm.append(job.get())

    return result_list_tqdm


if __name__ == '__main__':
    import argparse
    import multiprocessing as mp
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    Path(args.path).mkdir(parents=True, exist_ok=True)

    path = args.path
    n_agents = 100
    n_states = 1
    n_actions = 3
    n_iter = 100000
    epsilon = 0.01
    # alpha = 0.1
    gamma = 0
    q_initial = "UNIFORM"
    qmin = -2
    qmax = -1
    repetitions = 40

    num_cpus = int(os.environ.get("SLURM_NTASKS", os.cpu_count()))  # specific for euler cluster
    argument_list = []
    for user0_alpha in np.linspace(0.01, 1, 100):
        for expectation in np.linspace(0.01, 1, 100):
            for variance in [0]:
                experiment_name = f"user0{user0_alpha}_expectation{expectation}_variance{variance}"
                experiment_config = AlphaExperimentConfig(
                    user0_alpha=user0_alpha,
                    alpha_expectation=expectation,
                    alpha_variance=variance,
                    n_agents=n_agents,
                    n_states=n_states,
                    n_actions=n_actions,
                    n_iter=n_iter,
                    epsilon=epsilon,
                    gamma=gamma,
                    q_initial=q_initial,
                    qmin=qmin,
                    qmax=qmax,
                    path=path,
                    name=experiment_name,
                    repetitions=repetitions)
                argument_list.append(experiment_config)
    results = run_apply_async_multiprocessing(main, argument_list=argument_list, num_processes=num_cpus)

    utilities.save_pickle_with_unique_filename(results, f"{path}/results.pkl")
    name = f"results.csv"
    unique_name = utilities.get_unique_filename(base_filename=name)
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{path}/{unique_name}", index=False)
    print(f"saving to {path}/{unique_name}")
