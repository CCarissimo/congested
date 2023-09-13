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


def duopoly(a1, a2, n_actions=6):
    p1 = a1 / n_actions
    p2 = a2 / n_actions

    if p1 < p2:
        r1 = (1 - p1) * p1
        r2 = 0
    elif p1 == p2:
        r1 = 0.5 * (1 - p1)
        r2 = r1
    elif p1 > p2:
        r1 = 0
        r2 = (1 - p2) * p2

    R = np.array([r1, r2])
    S = np.array([a2, a1])

    return R, S


def run_game(n_agents, n_states, n_actions, n_iter, epsilon, alpha, gamma, q_initial, qmin, qmax):
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
    action1 = 0
    action2 = 0

    data = {}
    for t in range(n_iter):
        epsilon = (eps_end + (eps_start - eps_end) * math.exp(-1. * t / eps_decay))  # if t < N_ITER/10 else 0
        A = e_greedy_select_action(Q, S, epsilon)

        if t % 2 == 0:
            action1 = A[0]
        else:
            action2 = A[1]
        R, S = duopoly(action1, action2)

        Q, sum_of_belief_updates = bellman_update_q_table(Q, S, A, R, alpha, gamma)

        ## SAVE PROGRESS DATA
        data[t] = {"nA": np.bincount(A, minlength=3),
                   "R": R,
                   "Qmean": Q.mean(axis=1).mean(axis=0),
                   "Qvar": Q[ind, S, :].var(axis=0),
                   "A": A,
                   "Q": Q,
                   }
    return data


def main(path, n_agents, n_states, n_actions, n_iter, epsilon, alpha, gamma, q_initial, qmin, qmax):

    M = run_game(n_agents, n_states, n_actions, n_iter, epsilon, alpha, gamma, q_initial, qmin, qmax)

    experiment_name = f"N{n_agents}_S{n_states}_A{n_actions}_I{n_iter}_e{epsilon}_a{alpha}_g{gamma}"
    Path(f"{path}/{experiment_name}").mkdir(parents=True, exist_ok=True)
    run_name = utilities.get_unique_filename(base_filename="dump_run")
    utilities.save_pickle_with_unique_filename(M, f"{path}/{experiment_name}/{run_name}.pkl")

    exclusion_threshold = 0.8
    W = [M[t]["R"].mean() for t in range(0, n_iter)]
    L = nolds.lyap_r(W, int(n_iter*0.25))
    T = np.mean(W[int(exclusion_threshold * n_iter):n_iter])
    T_all = np.mean(W)
    T_std = np.std(W[int(exclusion_threshold * n_iter):n_iter])

    Qvar = [M[t]["Qvar"] for t in range(0, n_iter)]
    Qvar_mean = np.mean(Qvar)

    row = {
        "n_actions": n_agents,
        "alpha": alpha,
        "epsilon": epsilon,
        "T_mean": T,
        "T_mean_all": T_all,
        "T_std": T_std,
        "Lyapunov": L,
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

    Path(args.path).mkdir(parents=True, exist_ok=True)

    path = args.path
    n_agents = 2
    # n_states = "variable"
    # n_actions = "variable"
    n_iter = 10000
    #epsilon = "variable"
    #alpha = "variable"
    gamma = 0.8
    q_initial = "UNIFORM"
    qmin = 0
    qmax = 1

    num_cpus = int(os.environ.get("SLURM_NTASKS", os.cpu_count()))  # specific for euler cluster
    argument_list = []
    for epsilon in list(np.linspace(0, 0.2, 21)) + list(np.linspace(0.3, 1, 8)) + ["DECAYED"]:  # total 30
        for alpha in np.linspace(0.01, 0.2, 11):
            for n_actions in [6, 9, 12, 18, 24, 36, 48, 72, 96, 100]:
                n_states = n_actions
                for i in range(40):
                    parameter_tuple = (path, n_agents, n_states, n_actions, n_iter, epsilon, alpha, gamma, q_initial, qmin, qmax)
                    argument_list.append(parameter_tuple)
    results = run_apply_async_multiprocessing(main, argument_list=argument_list, num_processes=num_cpus)

    name = f"results.csv"
    unique_name = utilities.get_unique_filename(base_filename=name)
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{path}/{unique_name}", index=False)
    print(f"saving to {path}/{unique_name}")
