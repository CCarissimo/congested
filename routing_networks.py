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
    
    if n_up/n_agents < threshold:  # up is minority
        r_0 = 1
        r_1 = 0
        s = 0
    elif (n_agents - n_up)/n_agents < threshold:  # down is minority
        r_0 = 0
        r_1 = 1
        s = 1
    else:
        r_0 = 0
        r_1 = 0
        s = 2
    
    T = [r_0, r_1]

    R = np.array([T[i] for i in A])
    return R, T, np.broadcast_to(s, n_agents)
