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
    number_of_deviators: int


def run_deviation_braess(n_iter, n_agents, q_initial, alpha, alpha_deviator, epsilon, epsilon_deviator, gamma, gamma_deviator, number_of_deviators=1):
    # store_actions = np.zeros((n_iter, n_agents))
    # store_rewards = np.zeros((n_iter, n_agents))

    Q = initialize_q_table(q_initial, n_agents, n_states=1, n_actions=3, qmin=-2, qmax=-1)

    all_agent_indices = np.arange(n_agents)
    S = np.zeros(n_agents).astype(int)

    alphas = np.ones(n_agents) * alpha
    alphas[0:number_of_deviators] = alpha_deviator

    epsilons = np.ones(n_agents)*epsilon
    epsilons[0:number_of_deviators] = epsilon_deviator

    gammas = np.ones(n_agents) * gamma
    gammas[0:number_of_deviators] = gamma_deviator

    data = {}
    for t in range(n_iter):

        A = e_greedy_select_action(Q, S, epsilons)
        R, _, reward_per_action = braess_augmented_network(A, n_agents, cost=0)
        Q, sum_of_belief_updates = bellman_update_q_table(all_agent_indices, Q, S, A, R, S, alphas, gammas)

        ## SAVE PROGRESS DATA
        data[t] = {
                   "R": R,
                   # "reward_per_action": reward_per_action,
                   # "A": A,
                   # "Q": Q,
                   }
        # store_actions[t] = A
        # store_rewards[t] = R

    return data


def increase_decrease_size(W):
    """
    W is a vector of negative values, so diff of two consecutive negative values e.g. -2, -1
    will be -1 - -2 = 1
    or -1, -2,  -2 - -1 = -1
    in the first case social welfare increased and diff is positive
    in the second case social welfare decreased and diff is negative

    so edgeworth cycles which gradually reach Nash, means gradual decrease in social welfare
    and the average size of those steps should be smaller than the ones that rapidly shoot
    towards social optimum
    therefore average decreases should be smaller than average increases.
    """
    differences = np.diff(W)
    increase_indices = np.where(differences >= 0)
    decrease_indices = np.where(differences < 0)

    return differences[increase_indices].mean(), differences[decrease_indices].mean()


def increase_decrease_run_length(W):
    differences = np.diff(W)
    increases = np.where(differences > 0, True, False)
    decreases = np.where(differences < 0, True, False)

    count_increase_len = np.diff(
        np.where(np.concatenate(([increases[0]], increases[:-1] != increases[1:], [True])))[0])[::2]
    count_decrease_len = np.diff(
        np.where(np.concatenate(([decreases[0]], decreases[:-1] != decreases[1:], [True])))[0])[::2]

    return count_increase_len.mean(), count_decrease_len.mean()


def simple_probability_measure(W):
    W = -W
    increases = np.where(np.diff(W) >= 0)[0]
    probability = len(increases)/len(W)
    return probability


def drop_count_measure(W):
    W = -W
    mean = W.mean()
    above_mean = np.where(W > mean, True, False)
    below_mean = np.where(W < mean, True, False)
    below_mean = np.roll(below_mean, -1)
    indices_cross_from_above = np.logical_and(above_mean, below_mean)[1:]
    diff = np.diff(W)
    mean_decrease = np.where(diff < 0, -diff, 0).mean()
    std_decrease = np.where(diff < 0, -diff, 0).std()
    drop_indices = np.where(diff[indices_cross_from_above] > mean_decrease+3*std_decrease, 1, 0)
    num_drops = drop_indices.sum()
    return num_drops


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

        # print(results)
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


    # tracemalloc.start()

    t0 = time.time()
    main()
    t1 = time.time()

    # snapshot = tracemalloc.take_snapshot()
    # display_top(snapshot)

    total_n = t1 - t0

    print(total_n)
