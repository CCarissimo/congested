from learning_in_games import *
from dataclasses import dataclass
import numpy as np
import math
import os


@dataclass
class gameConfig:
    n_agents: int
    n_actions: int
    n_states: int
    m: float
    beta: float


@dataclass
class epsilonConfig:
    method: str
    parameter: float


@dataclass
class agentConfig:
    alpha: float
    gamma: float
    qinit: np.ndarray or str
    epsilon: float or str


def public_goods_game(A, config):
    norm_A = A/config.n_actions
    pot = config.m * np.power(norm_A, config.beta).sum()
    R = 1 - norm_A + pot
    return R


def run_public_goods_game(n_iter, game, agent):
    Q = initialize_q_table(agent.qinit, game.n_agents, game.n_states, game.n_actions)
    Q = np.random.random((game.n_agents, game.n_states, game.n_actions))

    if agent.epsilon == "DECAYED":
        EPS_START = 1
        EPS_END = 0
        EPS_DECAY = n_iter / 8
    else:
        EPS_START = agent.epsilon
        EPS_END = agent.epsilon
        EPS_DECAY = n_iter / 8

    M = {}
    ind = np.arange(game.n_agents)
    S = np.random.randint(game.n_states, size=game.n_agents)

    elist = []

    for t in range(n_iter):
        EPSILON = (EPS_END + (EPS_START - EPS_END) * math.exp(-1. * t / EPS_DECAY))  # if t < N_ITER/10 else 0
        elist.append(EPSILON)

        A = e_greedy_select_action(Q, S, EPSILON)

        R = public_goods_game(A, game)
        #         print(A, R)
        Q, sum_of_belief_updates = bellman_update_q_table(Q, S, A, R, agent.alpha, agent.gamma)

        ### SAVE PROGRESS DATA
        M[t] = {"nA": np.bincount(A, minlength=game.n_actions),
                "R": R,
                "Qmean": Q.mean(axis=1).mean(axis=0),
                "Qvar": Q[ind, S, :].var(axis=0),
                "sum_of_belief_updates": sum_of_belief_updates,
                "epsilon": EPSILON
                }
    return M, elist


def parallel_function(path, n_repetitions, n_iter, game_function, game_config, agent):
    precision = 3
    path_to_experiment = f"{path}/m{game_config.m:.{precision}}_beta{game_config.beta:.{precision}}"
    if not os.path.isdir(path_to_experiment):
        os.mkdir(path_to_experiment)

    results = []
    for i in range(n_repetitions):
        M = game_function(n_iter, game_config, agent)

        W = np.array([M[t]["R"].mean() for t in range(0, n_iter)])
        mean = W[int(0.8 * n_iter):n_iter].mean()
        variance = W[int(0.8 * n_iter):n_iter].var()

        filename = get_unique_filename(f"{path_to_experiment}/timeseries.npy")
        np.save(filename, W)

        row = {
            "mean": mean,
            "variance": variance
        }
        row.update(game_config.__dict__)
        row.update(agent.__dict__)
        results.append(row)

    return results


def main(path, n_repetitions, n_iter):
    import pandas as pd
    import multiprocessing as mp

    if not os.path.isdir(path):
        os.mkdir(path)

    num_cpus = mp.cpu_count()  # int(os.environ.get("SLURM_NTASKS", os.cpu_count()))  # specific for euler cluster

    game_function = run_public_goods_game  # references the function specified above to run the game
    x_values = np.linspace(0.5, 1, 21)
    y_values = np.linspace(0.5, 1.5, 21)
    argument_list = []
    for x in tqdm(x_values):
        for y in tqdm(y_values):
            game_config = gameConfig(
                n_agents=2, n_actions=3, n_states=1, m=x, beta=y
            )
            agent_config = agentConfig(
                alpha=0.1, gamma=0, qinit="UNIFORM", epsilon="DECAYED"
            )
            parameter_tuple = (
                path, n_repetitions, n_iter, game_function, game_config, agent_config
            )
            argument_list.append(parameter_tuple)

    results = run_apply_async_multiprocessing(parallel_function, argument_list=argument_list, num_processes=num_cpus)

    save_pickle_with_unique_filename(results, f"{path}/results.pkl")
    df = pd.DataFrame(results)
    df.to_csv(path + "/results.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("n_repetitions", type=int)
    parser.add_argument("n_iter", type=int)
    args = parser.parse_args()

    main(args.path, args.n_repetitions, args.n_iter)
