import numpy as np
from collections import


class game:
    def __init__(self, n_agents, n_states, n_actions):
        self.n_agents = n_agents
        self.n_states = n_states
        self.n_actions = n_actions
        self.counter = 0

    def step(self, A):
        R = None
        S = None
        return R, S


def duopoly(a1, a2, n_actions=6):
    p1 = a1 / n_actions
    p2 = a2 / n_actions

    if p1 < p2:
        r1 = (1 - p1) * p1
        r2 = 0
    if p1 == p2:
        r1 = 0.5 * (1 - p1)
        r2 = r1
    if p1 > p2:
        r1 = 0
        r2 = (1 - p2) * p2

    R = np.array([r1, r2])
    S = np.array([a2, a1])

    return R, S


def run_duopoly(EPSILON=0):
    # Q = initialize_q_table(QINIT, N_AGENTS, N_STATES, N_ACTIONS)
    Q = np.random.random((N_AGENTS, N_STATES, N_ACTIONS))

    ALPHA = np.random.random_sample(size=N_AGENTS)

    if EPSILON == "UNIFORM":
        EPSILON = np.random.random_sample(size=N_AGENTS) * mask
    else:
        EPSILON = EPSILON * np.ones(N_AGENTS) * mask

    EPS_START = EPSILON
    EPS_END = EPSILON
    EPS_DECAY = N_ITER / 8

    M = {}
    ind = np.arange(N_AGENTS)
    S = np.random.randint(N_STATES, size=N_AGENTS)
    R = np.ones(N_AGENTS) * -2
    A = np.random.randint(N_STATES, size=N_AGENTS)
    action1 = A[0]
    action2 = A[1]

    elist = []

    for t in range(N_ITER):

        EPSILON = (EPS_END + (EPS_START - EPS_END) * math.exp(-1. * t / EPS_DECAY))  # if t < N_ITER/10 else 0
        elist.append(EPSILON)

        A = e_greedy_select_action(Q, S, EPSILON)

        if t % 2 == 0:
            action1 = A[0]
        else:
            action2 = A[1]

        R, S = duopoly(a1=action1, a2=action2, n_actions=N_ACTIONS)

        Q, sum_of_belief_updates = bellman_update_q_table(Q, S, A, R, ALPHA, GAMMA)

        ### SAVE PROGRESS DATA
        M[t] = {"nA": np.bincount(A, minlength=3),
                "R": R,
                "Qmean": Q.mean(axis=1).mean(axis=0),
                # "groups": count_groups(Q[ind, S, :], 0.1),
                "Qvar": Q[ind, S, :].var(axis=0),
                "nA": np.bincount(A, minlength=3),
                # "T": travel_time_per_route,
                "sum_of_belief_updates": sum_of_belief_updates,
                # "alignment": alignment,
                # "recommendation_alignment": recommendation_alignment,
                # "action_alignment": action_alignment,
                }
    return M, elist


def prisoners_dilemma(a1, a2, r, s):
    if a1 == 0 and a2 == 0:
        r1 = r
        r2 = r
    elif a1 == 0 and a2 == 1:
        r1 = -s
        r2 = 1
    elif a1 == 1 and a2 == 0:
        r1 = 1
        r2 = -s
    elif a1 == 1 and a2 == 1:
        r1 = 0
        r2 = 0

    state = a1 + a2

    R = np.array([r1, r2])
    S = np.array([state, state])

    return R, S


def run_prisoners(EPSILON=0):
    # Q = initialize_q_table(QINIT, N_AGENTS, N_STATES, N_ACTIONS)
    Q = np.random.random((N_AGENTS, N_STATES, N_ACTIONS)) - 0.5

    ALPHA = np.random.random_sample(size=N_AGENTS)

    if EPSILON == "UNIFORM":
        EPSILON = np.random.random_sample(size=N_AGENTS) * mask
    else:
        EPSILON = EPSILON * np.ones(N_AGENTS) * mask

    EPS_START = EPSILON
    EPS_END = EPSILON
    EPS_DECAY = N_ITER / 8

    M = {}
    ind = np.arange(N_AGENTS)
    S = np.random.randint(N_STATES, size=N_AGENTS)
    R = np.ones(N_AGENTS) * -2
    A = np.random.randint(N_STATES, size=N_AGENTS)
    action1 = A[0]
    action2 = A[1]

    elist = []

    for t in range(N_ITER):
        EPSILON = (EPS_END + (EPS_START - EPS_END) * math.exp(-1. * t / EPS_DECAY))  # if t < N_ITER/10 else 0
        elist.append(EPSILON)

        A = e_greedy_select_action(Q, S, EPSILON)

        action1 = A[0]
        action2 = A[1]

        R, S = prisoners_dilemma(a1=action1, a2=action2, r=0.5, s=0.5)

        Q, sum_of_belief_updates = bellman_update_q_table(Q, S, A, R, ALPHA, GAMMA)

        ### SAVE PROGRESS DATA
        M[t] = {"nA": np.bincount(A, minlength=3),
                "R": R,
                "Qmean": Q.mean(axis=1).mean(axis=0),
                # "groups": count_groups(Q[ind, S, :], 0.1),
                "Qvar": Q[ind, S, :].var(axis=0),
                "nA": np.bincount(A, minlength=3),
                # "T": travel_time_per_route,
                "sum_of_belief_updates": sum_of_belief_updates,
                # "alignment": alignment,
                # "recommendation_alignment": recommendation_alignment,
                # "action_alignment": action_alignment,
                }
    return M, elist


def two_route_game(A, balance_parameter):
    n_players = len(A)
    fraction_a = (A == 0).sum() / n_players
    fraction_b = 1 - fraction_a

    travel_time_a = fraction_a + balance_parameter
    travel_time_b = fraction_b + (1 - balance_parameter)

    T = [-travel_time_a, -travel_time_b]
    R = np.array([T[a] for a in A])
    return R, T


def run_two_route_game(EPSILON=0, BALANCE=1):
    Q = initialize_q_table(QINIT, N_AGENTS, N_STATES, N_ACTIONS)
    # Q = np.random.random((N_AGENTS, N_STATES, N_ACTIONS)) * 2

    ALPHA = np.random.random_sample(size=N_AGENTS)

    if EPSILON == "UNIFORM":
        EPSILON = np.random.random_sample(size=N_AGENTS) * mask
    else:
        EPSILON = EPSILON * np.ones(N_AGENTS) * mask

    EPS_START = EPSILON
    EPS_END = EPSILON
    EPS_DECAY = N_ITER / 8

    M = {}
    ind = np.arange(N_AGENTS)
    S = np.random.randint(N_STATES, size=N_AGENTS)
    R = np.ones(N_AGENTS) * -2
    A = np.random.randint(N_STATES, size=N_AGENTS)

    elist = []

    for t in range(N_ITER):
        EPSILON = (EPS_END + (EPS_START - EPS_END) * math.exp(-1. * t / EPS_DECAY))  # if t < N_ITER/10 else 0
        elist.append(EPSILON)

        A = e_greedy_select_action(Q, S, EPSILON)

        R, T = two_route_game(A, balance_parameter=BALANCE)

        Q, sum_of_belief_updates = bellman_update_q_table(Q, S, A, R, ALPHA, GAMMA)

        ### SAVE PROGRESS DATA
        M[t] = {"nA": np.bincount(A, minlength=3),
                "R": R,
                "Qmean": Q.mean(axis=1).mean(axis=0),
                # "groups": count_groups(Q[ind, S, :], 0.1),
                "Qvar": Q[ind, S, :].var(axis=0),
                # "T": travel_time_per_route,
                "sum_of_belief_updates": sum_of_belief_updates,
                # "alignment": alignment,
                # "recommendation_alignment": recommendation_alignment,
                # "action_alignment": action_alignment,
                }
    return M, elist


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


def run_pigou_game(EPSILON=0, BALANCE=1):
    Q = initialize_q_table(QINIT, N_AGENTS, N_STATES, N_ACTIONS)
    # Q = np.random.random((N_AGENTS, N_STATES, N_ACTIONS)) * 2

    ALPHA = np.random.random_sample(size=N_AGENTS)

    if EPSILON == "DECAYED":
        EPS_START = 1
        EPS_END = 0
        EPS_DECAY = N_ITER / 8
    else:
        EPS_START = EPSILON
        EPS_END = EPSILON
        EPS_DECAY = N_ITER / 8

    M = {}
    ind = np.arange(N_AGENTS)
    S = np.random.randint(N_STATES, size=N_AGENTS)
    R = np.ones(N_AGENTS) * -2
    A = np.random.randint(N_STATES, size=N_AGENTS)

    elist = []

    for t in range(N_ITER):
        EPSILON = (EPS_END + (EPS_START - EPS_END) * math.exp(-1. * t / EPS_DECAY))  # if t < N_ITER/10 else 0
        elist.append(EPSILON)

        A = e_greedy_select_action(Q, S, EPSILON)

        R, T = pigou(A)

        Q, sum_of_belief_updates = bellman_update_q_table(Q, S, A, R, ALPHA, GAMMA)

        ### SAVE PROGRESS DATA
        M[t] = {"nA": np.bincount(A, minlength=3),
                "R": R,
                "Qmean": Q.mean(axis=1).mean(axis=0),
                # "groups": count_groups(Q[ind, S, :], 0.1),
                "Qvar": Q[ind, S, :].var(axis=0),
                # "T": travel_time_per_route,
                "sum_of_belief_updates": sum_of_belief_updates,
                # "alignment": alignment,
                # "recommendation_alignment": recommendation_alignment,
                # "action_alignment": action_alignment,
                }
    return M, elist


