import numpy as np
from tqdm.auto import tqdm
import pickle
import nolds
import pandas as pd
from routing_networks import braess_augmented_network
from run_functions import *
from agent_functions import *
from pathlib import Path
import utilities
import math


def run_game(n_agents, n_states, n_actions, n_iter, epsilon, alpha, gamma, q_initial, qmin, qmax, cost):
    Q = initialize_q_table(q_initial, n_agents, n_states, n_actions, qmin, qmax)
    alpha = initialize_learning_rates(alpha, n_agents)
    eps_decay = n_iter / 8
    if epsilon == "DECAYED":
        eps_start = 1
        eps_end = 0
    else:
        eps_start = epsilon
        eps_end = epsilon

    ind = np.arange(n_agents)
    S = np.random.randint(n_states, size=n_agents)

    data = {}
    for t in range(n_iter):
        epsilon = (eps_end + (eps_start - eps_end) * math.exp(-1. * t / eps_decay))  # if t < N_ITER/10 else 0
        A = e_greedy_select_action(Q, S, epsilon)
        R, _ = braess_augmented_network(A, cost=cost)
        Q, sum_of_belief_updates = bellman_update_q_table(Q, S, A, R, alpha, gamma)

        ## SAVE PROGRESS DATA
        data[t] = {
                   "R": R,
                   "Qmean": Q.mean(axis=1).mean(axis=0),
                   "groups": count_groups(Q[ind, S, :], 0.1),
                   "Qvar": Q[ind, S, :].var(axis=0),
                   # "A": A,
                   # "Q": Q,
                   }
    return data


def main(path, n_agents, n_states, n_actions, n_iter, epsilon, alpha, gamma, q_initial, qmin, qmax, cost):
    M = run_game(n_agents, n_states, n_actions, n_iter, epsilon, alpha, gamma, q_initial, qmin, qmax, cost)
    # experiment_name = f"N{n_agents}_S{n_states}_A{n_actions}_I{n_iter}_e{epsilon}_a{alpha}_g{gamma}_c{cost}"
    # Path(f"{path}/{experiment_name}").mkdir(parents=True, exist_ok=True)
    #
    # all_q_tables = np.stack([M[t]["Q"] for t in M.keys()])
    # utilities.save_numpy_array_with_unique_filename(all_q_tables, f"{path}/{experiment_name}/q_tables.npy")
    # all_rewards = np.stack([M[t]["R"] for t in M.keys()])
    # utilities.save_numpy_array_with_unique_filename(all_rewards, f"{path}/{experiment_name}/rewards.npy")
    # all_actions = np.stack([M[t]["A"] for t in M.keys()])
    # utilities.save_numpy_array_with_unique_filename(all_actions, f"{path}/{experiment_name}/actions.npy")

    exclusion_threshold = 0.8
    W = [M[t]["R"].mean() for t in range(0, n_iter)]
    L = nolds.lyap_r(W)
    T = np.mean(W[int(exclusion_threshold * n_iter):n_iter])
    T_all = np.mean(W)
    T_std = np.std(W[int(exclusion_threshold * n_iter):n_iter])

    groups = [M[t]["groups"] for t in range(0, n_iter)]
    groups_mean = np.mean(groups)
    groups_var = np.var(groups)
    Qvar = [M[t]["Qvar"] for t in range(0, n_iter)]
    Qvar_mean = np.mean(Qvar)

    row = {
        "n_agents": n_agents,
        "alpha": alpha,
        "epsilon": epsilon,
        "cost": cost,
        "T_mean": T,
        "T_mean_all": T_all,
        "T_std": T_std,
        "Lyapunov": L,
        "groups_mean": groups_mean,
        "groups_var": groups_var,
        "Qvar_mean": Qvar_mean,
    }

    return row


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

    # consider just leaving this as main
    # config file can be JSON
    # if using a dataclass you can initialize from JSON and check that variables are correct

    # dict --> dataclass

    # from dataclasses import dataclass
    # import json
    # @dataclass
    # class Config:
    #     path: str
    # d = json.load()
    # data = Config(**d)

    Path(args.path).mkdir(parents=True, exist_ok=True)

    path = args.path
    #n_agents = 100
    n_states = 1
    n_actions = 3
    n_iter = 10000
    #epsilon = "variable"
    #alpha = "variable"
    gamma = 0
    q_initial = "UNIFORM"
    qmin = -2
    qmax = -1
    #cost = "variable"

    num_cpus = int(os.environ.get("SLURM_NTASKS", os.cpu_count()))  # specific for euler cluster
    argument_list = []
    for epsilon in ["DECAYED"]:  # total 30 list(np.linspace(0, 0.2, 21))+list(np.linspace(0.3, 1, 8))+
        for alpha in np.linspace(0.01, 0.2, 11):
            for cost in np.linspace(0, 0.5, 11):
                for n_agents in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
                    for i in range(40):
                        parameter_tuple = (path, n_agents, n_states, n_actions, n_iter, epsilon, alpha, gamma, q_initial, qmin, qmax, cost)
                        argument_list.append(parameter_tuple)
    results = run_apply_async_multiprocessing(main, argument_list=argument_list, num_processes=num_cpus)

    name = f"results.csv"
    unique_name = utilities.get_unique_filename(base_filename=name)
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{path}/{unique_name}", index=False)
    print(f"saving to {path}/{unique_name}")
