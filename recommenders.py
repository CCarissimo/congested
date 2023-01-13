import numpy as np
from scipy.optimize import minimize


def heuristic_recommender(Q, n_agents):
    S = np.zeros(n_agents)
    flexible = []
    force_up = []
    force_down = []
    force_cross = []
    arg_max_Q = np.argmax(Q, axis=2)

    for i, argmax_q_table in enumerate(arg_max_Q):
        if 0 in argmax_q_table:  # could the agent go up?
            if 1 in argmax_q_table:  # could the agent go down?
                flexible.append(i)  # if both, store for later assignment
            else:
                force_up.append(i)  # if only up, assign agent to go up
        elif 1 in argmax_q_table:
            force_down.append(i)  # if only down, assign agent to go down
        else:
            force_cross.append(i)  # add logic for sure crossers

    n_flexible = len(flexible)
    n_up = len(force_up)
    n_down = len(force_down)
    n_cross = len(force_cross)
    diff_up_down = n_up - n_down

    if abs(diff_up_down) >= n_flexible:
        if diff_up_down > 0:
            while len(flexible) > 0:
                force_down.append(flexible.pop())  # assign all flexible to down
        else:
            while len(flexible) > 0:
                force_up.append(flexible.pop())  # assign all flexible to up

    elif abs(diff_up_down) < n_flexible:
        if diff_up_down > 0:
            for x in range(abs(diff_up_down)):
                force_down.append(flexible.pop())  # assign #diff_up_down flexible to down
        else:
            for x in range(abs(diff_up_down)):
                force_up.append(flexible.pop())  # assign #diff_up_down flexible to up

        counter = 0
        while len(flexible) > 0:  # split remaining flexible up and down equally
            if counter % 2 == 0:
                force_down.append(flexible.pop())
            else:
                force_up.append(flexible.pop())

    travel_time_estimate = [  # estimate travel times given, with optimistic assumption
        1 + (1 - n_cross / n_agents) / 2 + n_cross / n_agents,  # up
        1 + (1 - n_cross / n_agents) / 2 + n_cross / n_agents,  # down
        1 + n_cross / n_agents  # cross
    ]

    # pick the final states recommended to agents
    # using belief criteria
    #   - improve belief for up and down: argmax belief difference
    #   - worsen belief for cross: argmin belief difference
    # probably possible to optimize with numpy functions
    for i in force_up:
        recommendations_that_force = np.argwhere(arg_max_Q[i] == 0).flatten()
        belief_differences = - travel_time_estimate[0] - Q[i, recommendations_that_force, 0]
        best_recommendation = np.argmax(belief_differences)
        S[i] = recommendations_that_force[best_recommendation]

    for i in force_down:
        recommendations_that_force = np.argwhere(arg_max_Q[i] == 1).flatten()
        belief_differences = - travel_time_estimate[1] - Q[i, recommendations_that_force, 1]
        best_recommendation = np.argmax(belief_differences)
        S[i] = recommendations_that_force[best_recommendation]

    for i in force_cross:
        recommendations_that_force = np.argwhere(arg_max_Q[i] == 2).flatten()
        belief_differences = - travel_time_estimate[2] - Q[i, recommendations_that_force, 2]
        best_recommendation = np.argmin(belief_differences)
        S[i] = recommendations_that_force[best_recommendation]

    return S.astype(int)


def naive_recommender(Q, n_actions):
    initial_guess = np.random.randint(Q.shape[1], size=Q.shape[0])
    objective = total_welfare  # hard-coded to total welfare
    maximize = False
    coefficient = -1 if maximize else 0
    fun = lambda x: coefficient * objective(Q, x)
    recommendation = minimize(fun, x0=initial_guess, bounds=[(0, n_actions - 1) for i in range(len(initial_guess))],
                              method=None)
    S = np.rint(recommendation.x).astype(int)
    return S


def random_recommender(Q, n_actions):  # dummy function for random recommenders
    return np.random.randint(Q.shape[1], size=Q.shape[0])


def total_updates(Q, S):
    n_agents = len(S)
    S = np.rint(S).astype(int)
    indices = np.arange(n_agents)
    A = np.argmax(Q[indices, S, :], axis=1)

    n_up = (A == 0).sum()
    n_down = (A == 1).sum()
    n_cross = (A == 2).sum()

    r_0 = 1 + (n_up + n_cross) / n_agents
    r_1 = 1 + (n_down + n_cross) / n_agents
    r_2 = (n_up + n_cross) / n_agents + (n_down + n_cross) / n_agents
    T = [-r_0, -r_1, -r_2]

    dict_map = {0: r_0, 1: r_1, 2: r_2}

    R = -1 * np.vectorize(dict_map.get)(A)

    return np.sum((R - Q[indices, S, A]))


def total_welfare(Q, S):
    n_agents = len(S)
    S = np.rint(S).astype(int)
    indices = np.arange(n_agents)
    A = np.argmax(Q[indices, S, :], axis=1)

    n_up = (A == 0).sum()
    n_down = (A == 1).sum()
    n_cross = (A == 2).sum()

    r_0 = 1 + (n_up + n_cross) / n_agents
    r_1 = 1 + (n_down + n_cross) / n_agents
    r_2 = (n_up + n_cross) / n_agents + (n_down + n_cross) / n_agents
    T = [-r_0, -r_1, -r_2]

    dict_map = {0: r_0, 1: r_1, 2: r_2}

    R = -1 * np.vectorize(dict_map.get)(A)

    return np.mean(R)


def recommender(Q, initial_guess, objective, n_actions, maximize=False):
    coefficient = -1 if maximize else 0
    fun = lambda x: coefficient * objective(Q, x)
    recommendation = minimize(fun, x0=initial_guess, bounds=[(0, n_actions - 1) for i in range(len(initial_guess))],
                              method=None)
    S = np.rint(recommendation.x).astype(int)
    return S
