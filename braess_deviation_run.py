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
class DeviationBraessExperimentConfig:
    n_iter: int
    n_agents: int
    q_initial: Union[np.ndarray, str]
    alpha: Union[float, str]
    alpha_deviator: float
    epsilon: Union[float, str]
    epsilon_deviator: float
    gamma: float
    gamma_deviator: float


def run_deviation_braess(n_iter, n_agents, q_initial, alpha, alpha_deviator, epsilon, epsilon_deviator, gamma, gamma_deviator):
    Q = initialize_q_table(q_initial, n_agents, n_states=1, n_actions=3, qmin=-2, qmax=-1)

    all_agent_indices = np.arange(n_agents)
    S = np.zeros(n_agents).astype(int)

    alphas = np.ones(n_agents) * alpha
    alphas[0] = alpha_deviator

    epsilons = np.ones(n_agents)*epsilon
    epsilons[0] = epsilon_deviator

    gammas = np.ones(n_agents) * gamma
    gammas[0] = gamma_deviator

    data = {}
    for t in range(n_iter):

        A = e_greedy_select_action(Q, S, epsilons)
        R, _, reward_per_action = braess_augmented_network(A, n_agents, cost=0)
        Q, sum_of_belief_updates = bellman_update_q_table(all_agent_indices, Q, S, A, R, S, alphas, gammas)

        ## SAVE PROGRESS DATA
        data[t] = {
                   "R": R,
                   # "reward_per_action": reward_per_action,
                   "A": A,
                   # "Q": Q,
                   }
    return data


if __name__ == '__main__':

    import time
    import tracemalloc
    import linecache
    import os


    def main():
        n_iter = 40000
        n_agents = 100
        q_initial = "UNIFORM"
        alpha = 0.1
        alpha_deviator = 0.5
        epsilon = 0.01
        epsilon_deviator = 0.2
        gamma = 0
        gamma_deviator = 0.1

        params = DeviationBraessExperimentConfig(
            n_iter,
            n_agents,
            q_initial,
            alpha,
            alpha_deviator,
            epsilon,
            epsilon_deviator,
            gamma,
            gamma_deviator
        )

        print(str(params))

        results = run_deviation_braess(
            n_iter,
            n_agents,
            q_initial,
            alpha,
            alpha_deviator,
            epsilon,
            epsilon_deviator,
            gamma,
            gamma_deviator
        )

        print(results)
        return None

    def display_top(snapshot, key_type='lineno', limit=3):
        snapshot = snapshot.filter_traces((
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        ))
        top_stats = snapshot.statistics(key_type)

        print("Top %s lines" % limit)
        for index, stat in enumerate(top_stats[:limit], 1):
            frame = stat.traceback[0]
            # replace "/path/to/module/file.py" with "module/file.py"
            filename = os.sep.join(frame.filename.split(os.sep)[-2:])
            print("#%s: %s:%s: %.1f KiB"
                  % (index, filename, frame.lineno, stat.size / 1024))
            line = linecache.getline(frame.filename, frame.lineno).strip()
            if line:
                print('    %s' % line)

        other = top_stats[limit:]
        if other:
            size = sum(stat.size for stat in other)
            print("%s other: %.1f KiB" % (len(other), size / 1024))
        total = sum(stat.size for stat in top_stats)
        print("Total allocated size: %.1f KiB" % (total / 1024))


    tracemalloc.start()

    t0 = time.time()
    main()
    t1 = time.time()

    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)

    total_n = t1 - t0

    print(total_n)
