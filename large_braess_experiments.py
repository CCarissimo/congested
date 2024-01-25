from tqdm.auto import tqdm
import nolds
import pandas as pd
from learning_in_games import *
from pathlib import Path
from learning_in_games import utilities
import math
import igraph as ig


g = ig.Graph(
    9,
    [(0, 1), (0, 2),
     (1, 3), (1, 4), (1, 2),
     (2, 4), (2, 5),
     (3, 4), (3, 6),
     (4, 6), (4, 7), (4, 5),
     (5, 7),
     (6, 8), (6, 7),
     (7, 8)],
    directed=True
)
weights = ["x", 1,
           "x", 1, 0,
           "x", 1,
           0, 1,
           "x", 1, 0,
           "x",
           1, 0,
           "x"]
g.es["cost"] = weights
adj = g.get_adjacency(attribute="cost")
paths = g.get_all_simple_paths(0, to=8)


def large_braess_network(A, paths, adj, n_agents):
    visits = np.zeros((9, 9))
    for a in A:
        path = paths[a]
        for i in range(len(path) - 1):
            node = path[i]
            next_node = path[i + 1]
            visits[node, next_node] += 1

    costs = np.zeros((9, 9))
    for i in range(9):
        for j in range(9):
            if adj[i, j] == 'x':
                costs[i, j] = visits[i, j] / 100
            elif adj[i, j] == 1:
                costs[i, j] = 1

    R = np.zeros(n_agents)
    for agent, a in enumerate(A):
        path = paths[a]
        cost = 0
        for i in range(len(path) - 1):
            node = path[i]
            next_node = path[i + 1]
            cost += costs[node, next_node]
        R[agent] = cost
    return -R


def run_game(n_agents, n_states, n_actions, n_iter, epsilon, alpha, gamma, q_initial, qmin, qmax, cost):
    gameConfig = RouteConfig(n_agents, n_actions, n_states, cost)
    agentConfig = EpsilonGreedyConfig(alpha, gamma, q_initial, epsilon)

    Q = initialize_q_table(q_initial, gameConfig, qmin, qmax)
    # alpha = initialize_learning_rates(agentConfig, gameConfig)
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
        A = e_greedy_select_action(Q, S, agentConfig)
        R = large_braess_network(A, paths, adj, gameConfig)
        Q, sum_of_belief_updates = bellman_update_q_table(Q, S, A, R, S, agentConfig)

        ## SAVE PROGRESS DATA
        data[t] = {
                   "R": R,
                   "Qmean": Q.mean(axis=1).mean(axis=0),
                   # "groups": count_groups(Q[ind, S, :], 0.1),
                   "Qvar": Q[ind, S, :].var(axis=0),
                   # "A": A,
                   # "Q": Q,
                   }
    return data


def main(path, n_agents, n_states, n_actions, n_iter, repetitions, epsilon, alpha, gamma, q_initial, qmin, qmax, cost):
    all_repetitions = []
    for i in range(repetitions):
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

        # groups = [M[t]["groups"] for t in range(0, n_iter)]
        # groups_mean = np.mean(groups)
        # groups_var = np.var(groups)
        Qvar = [M[t]["Qvar"] for t in range(0, n_iter)]
        Qvar_mean = np.mean(Qvar)

        row = {
            "repetition": i,
            "n_agents": n_agents,
            "alpha": alpha,
            "epsilon": epsilon,
            "cost": cost,
            "T_mean": T,
            "T_mean_all": T_all,
            "T_std": T_std,
            "Lyapunov": L,
            # "groups_mean": groups_mean,
            # "groups_var": groups_var,
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
    n_agents = 100
    n_states = 1
    n_actions = 20
    n_iter = 20000
    # epsilon = "variable"
    alpha = 0.1
    gamma = 0
    q_initial = "UNIFORM"
    qmin = -2
    qmax = -1
    cost = 0
    repetitions = 10

    num_cpus = mp.cpu_count()-1  # int(os.environ.get("SLURM_NTASKS", os.cpu_count()))  # specific for euler cluster
    argument_list = []
    for epsilon in list(np.linspace(0, 0.2, 21))+list(np.linspace(0.3, 1, 8)):  #
        parameter_tuple = (path, n_agents, n_states, n_actions, n_iter, repetitions, epsilon, alpha, gamma, q_initial, qmin, qmax, cost)
        argument_list.append(parameter_tuple)
    results = run_apply_async_multiprocessing(main, argument_list=argument_list, num_processes=num_cpus)

    utilities.save_pickle_with_unique_filename(results, "results.pkl")
    name = f"results.csv"
    unique_name = utilities.get_unique_filename(base_filename=name)
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{path}/{unique_name}", index=False)
    print(f"saving to {path}/{unique_name}")