import numpy as np
from run_functions import *
from agent_functions import *


def single_run(routing_network, n_agents, n_states, n_actions, n_iter, epsilon, gamma, alpha, q_initial, recommender):
    Q = initialize_q_table(q_initial, n_agents, n_states, n_actions)
    alpha = initialize_learning_rates(alpha, n_agents)
    epsilon = initialize_exploration_rates(epsilon, n_agents)
    data = {}
    ind = np.arange(n_agents)

    for t in range(n_iter):

        S = recommender(Q, n_agents)
        A = e_greedy_select_action(Q, S, epsilon)
        R, travel_time_per_route = routing_network(A)
        Q, sum_of_belief_updates = bellman_update_q_table(Q, S, A, R, alpha, gamma)

        ## SAVE PROGRESS DATA
        data[t] = {"nA": np.bincount(A, minlength=3),
                   "R": R,
                   "Qmean": Q.mean(axis=1).mean(axis=0),
                   "groups": count_groups(Q[ind, S, :], 0.1),
                   "Qvar": Q[ind, S, :].var(axis=0),
                   "T": travel_time_per_route,
                   "sum_of_belief_updates": sum_of_belief_updates,
                   "alignment": calculate_alignment(Q, S, A),
                   "S": S,
                   "A": A,
                   "Q": Q,
                   }
    return data


if __name__ == "__main__":
    from routing_networks import braess_augmented_network, braess_initial_network
    from recommenders import heuristic_recommender, constant_recommender
    from plot_functions import plot_run

    N_AGENTS = 100
    N_STATES = 3
    N_ACTIONS = 3
    N_ITER = 10000

    EPSILON = 0.01
    mask = np.zeros(N_AGENTS)
    mask[:] = 1
    GAMMA = 0
    ALPHA = 0.01

    QINIT = "UNIFORM"  # np.array([-2, -2, -2])

    M = single_run(braess_initial_network, N_AGENTS, N_STATES, N_ACTIONS, N_ITER, EPSILON, GAMMA, ALPHA, QINIT,
                   constant_recommender)

    NAME = f"run_N{N_AGENTS}_S{N_STATES}_A{N_ACTIONS}_I{N_ITER}_e{EPSILON}_g{GAMMA}_a{ALPHA}_q{QINIT}"

    plot_run(M, NAME, N_AGENTS, N_ACTIONS, N_ITER)
