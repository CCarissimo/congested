import numpy as np


def braess_augmented_network(A):
    n_agents = len(A)
    n_up = (A == 0).sum()
    n_down = (A == 1).sum()
    n_cross = (A == 2).sum()

    r_0 = 1 + (n_up + n_cross) / n_agents
    r_1 = 1 + (n_down + n_cross) / n_agents
    r_2 = (n_up + n_cross) / n_agents + (n_down + n_cross) / n_agents
    T = [-r_0, -r_1, -r_2]

    R = np.array([T[a] for a in A])  # -1 * np.vectorize(dict_map.get)(A)
    return R, T


def braess_initial_network(A):
    n_agents = len(A)
    n_up = (A == 0).sum()
    n_down = (A == 1).sum()

    r_0 = 1 + n_up / n_agents
    r_1 = 1 + n_down / n_agents
    T = [-r_0, -r_1]

    R = np.array([T[a] for a in A])
    return R, T


def two_route_game(A):
    n_agents = len(A)
    n_up = (A == 0).sum()

    r_0 = n_up / n_agents
    r_1 = 1
    T = [-r_0, -r_1]

    R = np.array([T[i] for i in A])
    return R, T


def minority_game(A, threshold=0.4):
    n_agents = len(A)
    n_up = (A == 0).sum()
    
    if n_agents * threshold >= n_up: # up is minority
        r_0 = 1
        r_1 = 0
    else:
        r_0 = 0
        r_1 = 1
    
    T = [r_0, r_1]

    R = np.array([T[i] for i in A])
    return R, T


def minority_game_2(A, threshold=0.4):
    n_agents = len(A)
    n_a = (A == 0).sum()
    fraction_a = n_a/n_agents
    fraction_b = 1 - fraction_a
    
    r_a = 1 - 2*fraction_a
    r_b = 1 - 2*fraction_b
    
    T = [r_a, r_b]

    R = np.array([T[i] for i in A])
    return R, T


def el_farol_bar(A):
    n_agents = len(A)
    n_home = (A == 0).sum()
    n_bar = (A == 1).sum()
    pct = n_bar / n_agents

    r_0 = 1
    r_1 = 2 - 4 * pct if (pct < 0) else 4 * pct - 2

    T = [-r_0, -r_1]

    R = np.array([T[a] for a in A])  # -1 * np.vectorize(dict_map.get)(A)
    return R, T


def pigou(A):
    n_agents = len(A)
    n_up = (A == 0).sum()
    n_down = (A == 1).sum()
    pct = n_down / n_agents

    r_0 = 1
    r_1 = pct

    T = [-r_0, -r_1]

    R = np.array([T[a] for a in A])  # -1 * np.vectorize(dict_map.get)(A)
    return R, T


def pigou3(A):
    n_agents = len(A)
    n_up = (A == 0).sum()
    n_down = (A == 1).sum()
    n_cross = (A == 2).sum()
    #     pct = n_down/n_agents

    r_0 = n_up / n_agents
    r_1 = 1
    r_2 = 1  # n_cross/n_agents  # 1

    T = [-r_0, -r_1, -r_2]
    print(A)
    R = np.array([T[a] for a in A])  # -1 * np.vectorize(dict_map.get)(A)
    return R, T

