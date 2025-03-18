from tqdm.auto import tqdm
import nolds
import pandas as pd
from learning_in_games import *
from pathlib import Path
from learning_in_games import utilities
import math
import igraph as ig
from ta_solver.cost_functions import LinearCostFunction

class MultiODCongestionGame(RouteConfig):
    graph: ig.Graph
    mat: np.ndarray # adjacency matrix
    od: dict[tuple[int, int]: int]
    action_agent_map: dict[int: tuple[int, int]]
    od_paths = dict[tuple[int, int]: list[int]]

    def __init__(self, graph, od):
        self.graph = graph
        self.od = od
        self.od_paths = self.build_odpath(od, graph)
        self.n_agents = sum(self.od.values())
        self.action_agent_map= {}
        for k, v in self.od.items():
            offset = len(self.action_agent_map)
            self.action_agent_map.update({i+offset: k for i in range(v)})
                    
    def build_odpath(self, od, graph, costfunc=None, offset=0, max_paths=60, cf_kwargs={}):
        od_paths = {}
        if costfunc:
            graph.es['flow'] = 0
            graph.es['sp_weight'] = costfunc(graph, **cf_kwargs).eval()
        else:
            graph.es['sp_weight'] = 1
        for (o, d), flow in od.items():
            if o==d or flow==0: continue
            from_node = o + offset
            to_node = d + offset
            vertex_paths = graph.get_k_shortest_paths(from_node, to=to_node, k=max_paths, 
                                                    weights=graph.es['sp_weight'])
            od_paths[(o, d)] = []

            for path in vertex_paths:
                edge_ids = []
                for i in range(len(path)-1):
                    edge_ids.append((path[i], path[i+1]))
                s = graph.get_eids(edge_ids)
                od_paths[(o, d)].append(tuple(s))
                
        return od_paths
    
class RouteAgentConfig(EpsilonGreedyConfig):
    od: tuple[int, int]

graph = ig.read('networks/large_braess_network.graphml')
ods = {(0,8): 80,
       (4,8): 20
       }
# g = ig.Graph(
#     9,
#     [(0, 1), (0, 2),
#      (1, 3), (1, 4), (1, 2),
#      (2, 4), (2, 5),
#      (3, 4), (3, 6),
#      (4, 6), (4, 7), (4, 5),
#      (5, 7),
#      (6, 8), (6, 7),
#      (7, 8)],
#     directed=True
# )
# weights = ["x", 1,
#            "x", 1, 0,
#            "x", 1,
#            0, 1,
#            "x", 1, 0,
#            "x",
#            1, 0,
#            "x"]
# g.es["cost"] = weights
# adj = g.get_adjacency(attribute="cost")
# paths = g.get_all_simple_paths(0, to=8)
costfunc = LinearCostFunction(graph)

def initialize_q_tables(q_initial, gameConfig, qmin, qmax):
    Q_tables = {}
    for od, count in gameConfig.od.items():
        n_actions = len(gameConfig.od_paths[od])
        Q_tables[od] = initialize_q_table(q_initial, count, 1, n_actions, qmin, qmax)
    return Q_tables

def bellman_update_q_table(n_agents, Q, S, A, R, S_, alpha, gamma):
    """
    Performs a one-step update using the bellman update equation for Q-learning.
    :param ind: the indices of agents to update for, must match shapes of other arrays
    :param agentConfig:
    :param Q: np.ndarray Q-table indexed by (agents, states, actions)
    :param S: np.ndarray States indexed by (agents)
    :param A: np.ndarray Actions indexed by (agents)
    :param R: np.ndarray Rewards indexed by (agents)
    :param S_: np.ndarray Next States indexed by (agents)
    :return: np.ndarray Q-table indexed by (agents, states, actions)
    """
    ind = list(range(n_agents))
    # print(ind, Q[ind, S[ind], A[ind]], S[ind], A[ind], R[ind], S_[ind], Q[ind, S_[ind]].shape)
    all_belief_updates = alpha * (R[ind] + gamma * Q[ind, S_[ind]].max(axis=1) - Q[ind, S[ind], A[ind]])
    Q[ind, S[ind], A[ind]] = Q[ind, S[ind], A[ind]] + all_belief_updates
    return Q, np.abs(all_belief_updates).sum()

def multi_od_congestion_game(A, agents, game: MultiODCongestionGame):
    graph = game.graph
    graph.es['flow'] = 0

    for agent_idx, a in enumerate(A):
        agent_od = agents[agent_idx]
        path = game.od_paths[agent_od][a]
        graph.es[path]['flow'] += np.ones_like(graph.es[path]['flow'])
    evaluated = costfunc.eval(optimal=False)

    costs = {k: None for k, v in game.od_paths.items()}
    for od, paths in game.od_paths.items():
        path_costs = [np.sum(evaluated[list(path)]) for path in paths]
        costs[od] = np.array(path_costs)

    R = np.zeros((game.n_agents))
    for agent_idx, agent_od in agents.items():
        path_idx = A[agent_idx]
        R[agent_idx] = costs[agent_od][path_idx]

    return -R


def run_game(n_agents, n_states, n_actions, n_iter, epsilon, alpha, gamma, q_initial, qmin, qmax, cost):
    gameConfig = MultiODCongestionGame(graph, ods)
    agentConfig = RouteAgentConfig(alpha, gamma, q_initial, epsilon)

    Q_tables = initialize_q_tables(q_initial, gameConfig, qmin, qmax)
    # alpha = initialize_learning_rates(agentConfig, gameConfig)
    eps_decay = n_iter / 8
    if epsilon == "DECAYED":
        eps_start = 1
        eps_end = 0
    else:
        eps_start = epsilon
        eps_end = epsilon

    ind = np.arange(n_agents)
    S_vals: np.ndarray = np.random.randint(n_states, size=n_agents)
    A = np.zeros(n_agents, dtype=np.int16)


    data = {}
    for t in range(n_iter):
        for od, Q in Q_tables.items():
            agent_idx = [i for i, _od in gameConfig.action_agent_map.items() if od==_od]
            S = S_vals[agent_idx]
            epsilon = (eps_end + (eps_start - eps_end) * math.exp(-1. * t / eps_decay))  # if t < N_ITER/10 else 0
            A_od = e_greedy_select_action(Q, S, epsilon)
            A[agent_idx] = A_od
        R = multi_od_congestion_game(A, gameConfig.action_agent_map, gameConfig)
        # R = np.ones_like(R) * R.mean()
        for od, Q in Q_tables.items():
            agent_idx = [i for i, _od in gameConfig.action_agent_map.items() if od==_od]
            Q_tables[od], sum_of_belief_updates = bellman_update_q_table(len(agent_idx), Q_tables[od], S_vals[agent_idx], A[agent_idx], R[agent_idx], S_vals[agent_idx], alpha, gamma)

            ## SAVE PROGRESS DATA
        data[t] = {
                "R": R,
                # "Qmean": Q.mean(axis=1).mean(axis=0),
                # "groups": count_groups(Q[ind, S, :], 0.1),
                # "Qvar": Q[ind, S, :].var(axis=0),
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
        # L = nolds.lyap_r(W)
        T = np.mean(W[int(exclusion_threshold * n_iter):n_iter])
        T_all = np.mean(W)
        T_std = np.std(W[int(exclusion_threshold * n_iter):n_iter])

        # groups = [M[t]["groups"] for t in range(0, n_iter)]
        # groups_mean = np.mean(groups)
        # groups_var = np.var(groups)
        # Qvar = [M[t]["Qvar"] for t in range(0, n_iter)]
        # Qvar_mean = np.mean(Qvar)

        row = {
            "repetition": i,
            "n_agents": n_agents,
            "alpha": alpha,
            "epsilon": epsilon,
            "cost": cost,
            "T_mean": T,
            "T_mean_all": T_all,
            "T_std": T_std,
            # "Lyapunov": L,
            # "groups_mean": groups_mean,
            # "groups_var": groups_var,
            # "Qvar_mean": Qvar_mean,
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
    parser.add_argument('--path', default='test', type=str)
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
    repetitions = 3

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