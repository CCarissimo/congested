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


def split_up_down_cross_flexible(arg_max_Q):
    up = []
    down = []
    cross = []
    flexible = []

    for i, argmax_q_table in enumerate(arg_max_Q):
        if 0 in argmax_q_table:  # could the agent go up?
            if 1 in argmax_q_table:  # could the agent go down?
                flexible.append(i)  # if both, store for later assignment
            else:
                up.append(i)  # if only up, assign agent to go up
        elif 1 in argmax_q_table:
            down.append(i)  # if only down, assign agent to go down
        else:
            cross.append(i)  # add logic for sure crossers

    return up, down, cross, flexible


def sort_by_belief_difference(Q, arg_max_Q, agent_indices, action):
    ind, state = np.where(arg_max_Q[agent_indices] == action)

    values = Q[agent_indices][ind, state, action] - Q[agent_indices][ind, state, 2]

    structured_array = np.array(
        list(zip(values, ind, state)),
        dtype=[("belief_diff", float), ("index", int), ("state", int)]
    )
    structured_array.sort(order="belief_diff")

    sorted_agent_indices = []
    sorted_agent_recommendations = []

    for agent in reversed(structured_array):
        agent_index = agent_indices[agent[1]]
        recommendation = agent[2]
        if agent_index in sorted_agent_indices:
            continue
        else:
            sorted_agent_indices.append(agent_index)
            sorted_agent_recommendations.append(recommendation)

    return sorted_agent_indices, sorted_agent_recommendations


def sort_by_belief_improvement(Q, arg_max_Q, agent_indices, action, method, travel_time_estimate, minimize=False):
    coefficient = -1 if minimize else 1

    ind, state = np.where(arg_max_Q[agent_indices] == action)

    target = travel_time_estimate[action] if method == "estimate" else Q[agent_indices][ind, state, 2]

    values = coefficient * (- target - Q[agent_indices][ind, state, action])

    structured_array = np.array(
        list(zip(values, ind, state)),
        dtype=[("belief_diff", float), ("index", int), ("state", int)]
    )
    structured_array.sort(order="belief_diff")

    sorted_agent_indices = []
    sorted_agent_recommendations = []

    for agent in reversed(structured_array):
        agent_index = agent_indices[agent[1]]
        recommendation = agent[2]
        if agent_index in sorted_agent_indices:
            continue
        else:
            sorted_agent_indices.append(agent_index)
            sorted_agent_recommendations.append(recommendation)

    return sorted_agent_indices, sorted_agent_recommendations


def assign_flexible_to_up_down(S, force_up, force_down, flexible, Q, arg_max_Q, method, estimate, minimize=False):
    n_flexible = len(flexible)
    n_up = len(force_up)
    n_down = len(force_down)
    diff_up_down = n_up - n_down
    minority = 1 if diff_up_down >= 0 else 0

    sorted_flexible, flexible_recommendation = sort_by_belief_improvement(
        Q, arg_max_Q, flexible, minority, method, estimate, minimize=minimize)

    if abs(diff_up_down) >= n_flexible:

        while len(sorted_flexible) > 0:
            S[sorted_flexible.pop()] = flexible_recommendation.pop()

    elif abs(diff_up_down) < n_flexible:

        for x in range(abs(diff_up_down) + int((n_flexible - abs(diff_up_down)) / 2)):
            S[sorted_flexible.pop()] = flexible_recommendation.pop()

        sorted_majority, majority_recommendation = sort_by_belief_improvement(
            Q, arg_max_Q, sorted_flexible, abs(1 - minority), method, estimate, minimize=minimize)

        while len(sorted_majority) > 0:
            S[sorted_majority.pop()] = majority_recommendation.pop()

    return S


def assign(S, Q, arg_max_Q, agent_indices, action, method, estimate, minimize=False):

    sorted_agent_indices, sorted_agent_recommendations = sort_by_belief_improvement(
        Q, arg_max_Q, agent_indices, action, method, estimate, minimize=minimize)

    while len(sorted_agent_indices) > 0:
        S[sorted_agent_indices.pop()] = sorted_agent_recommendations.pop()
    return S


def calculate_travel_time_estimate(n_cross, n_agents):
    return [
            1+(1-n_cross/n_agents)/2+n_cross/n_agents,  # up
            1+(1-n_cross/n_agents)/2+n_cross/n_agents,  # down
            1+n_cross/n_agents                          # cross
            ]


def optimized_heuristic_recommender(Q, n_agents, method="estimate", minimize=False):
    S = np.zeros(n_agents)
    arg_max_Q = np.argmax(Q, axis=2)

    up, down, cross, flexible = split_up_down_cross_flexible(arg_max_Q)

    estimate = calculate_travel_time_estimate(len(cross), n_agents)

    assign_flexible_to_up_down(S, up, down, flexible, Q, arg_max_Q, method, estimate, minimize)

    S = assign(S, Q, arg_max_Q, up, 0, method, estimate, minimize)
    S = assign(S, Q, arg_max_Q, down, 1, method, estimate, minimize)
    S = assign(S, Q, arg_max_Q, cross, 2, method, estimate, (not minimize))

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


def constant_recommender(Q, n_actions):
    S = np.zeros(Q.shape[0])
    S[int(Q.shape[0]/2):] = 1  # half up and half down  
    return S.astype(int)


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


def aligned_heuristic_recommender(Q, n_agents):
    S = np.zeros(n_agents)
    arg_max_Q = np.argmax(Q, axis=2)
    alignment_table = arg_max_Q[:, :3] == np.broadcast_to(np.arange(3), (n_agents, 3))  # alignment of up and down, only

    aligned_both = []
    aligned_up = []
    aligned_down = []
    aligned_cross = []
    misaligned = []

    up = 0
    down = 1
    cross = 2

    for i, alignings in enumerate(alignment_table):
        if alignings[up]:  # could the agent go up?
            if alignings[down]:  # could the agent also go down?
                aligned_both.append(i)  # if both, store for later assignment
            else:
                aligned_up.append(i)  # if only up, assign agent to go up
        elif alignings[down]:
            aligned_down.append(i)  # if only down, assign agent to go down
        elif alignings[cross]:
            aligned_cross.append(i)
        else:
            misaligned.append(i)

    n_tuple = (len(aligned_up), len(aligned_down), len(aligned_cross), len(aligned_both), len(misaligned))

    n_cross = len(aligned_cross) + int(len(misaligned))  # optimistic estimate of crossers

    travel_time_estimate = calculate_travel_time_estimate(n_cross, n_agents)

    # assign all aligned recommendations possible
    S[aligned_up] = up
    S[aligned_down] = down
    S[aligned_cross] = cross

    # ASSIGN FLEXIBLE AGENTS
    diff_up_down = len(aligned_up) - len(aligned_down)
    n_flexible = len(aligned_both)
    if abs(diff_up_down) >= n_flexible > 0:
        if diff_up_down > 0:
            # assign all flexible to down
            while len(aligned_both) > 0: S[aligned_both.pop()] = down
        else:
            # assign all flexible to up
            while len(aligned_both) > 0: S[aligned_both.pop()] = up
    elif abs(diff_up_down) < n_flexible and n_flexible > 0:
        if diff_up_down > 0:
            # assign #diff_up_down flexible to down
            for x in range(abs(diff_up_down)): S[aligned_both.pop()] = down
        else:
            # assign #diff_up_down flexible to up
            for x in range(abs(diff_up_down)): S[aligned_both.pop()] = up
        # split remaining flexible up and down equally
        counter = np.random.randint(0, 1)
        while len(aligned_both) > 0:
            if counter % 2 == 0:
                S[aligned_both.pop()] = down
            else:
                S[aligned_both.pop()] = up
            counter += 1

    action = (-(- np.array(travel_time_estimate) - Q.max(axis=2))).mean(axis=0).argmax()
    S[misaligned] = action

    return S.astype(int)
