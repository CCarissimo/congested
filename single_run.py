from learning_in_games.run_functions import *
from learning_in_games.agent_functions import *
import math

def single_run(game, n_agents, n_states, n_actions, n_iter, epsilon, gamma, alpha, q_initial, qmin, qmax, kwargs):
    Q = initialize_q_table(q_initial, n_agents, n_states, n_actions, qmin, qmax)
    alpha = initialize_learning_rates(alpha, n_agents)
    eps_decay = N_ITER / 8
    if epsilon == "DECAYED":
        eps_start = 1
        eps_end = 0
    else:
        eps_start = epsilon
        eps_end = epsilon

    ind = np.arange(n_agents)
    S = np.random.randint(n_states, size=n_agents)

    data = {}
    for t in range(n_iter):
        epsilon = (eps_end + (eps_start - eps_end) * math.exp(-1. * t / eps_decay))  # if t < N_ITER/10 else 0
        A = e_greedy_select_action(Q, S, epsilon)
        R, S = game(A)
        Q, sum_of_belief_updates = bellman_update_q_table(Q, S, A, R, alpha, gamma)

        ## SAVE PROGRESS DATA
        data[t] = {"nA": np.bincount(A, minlength=3),
                   "R": R,
                   "Qmean": Q.mean(axis=1).mean(axis=0),
                   "groups": count_groups(Q[ind, S, :], 0.1),
                   "Qvar": Q[ind, S, :].var(axis=0),
                   "A": A,
                   "Q": Q,
                   }
    return data


if __name__ == "__main__":
    from learning_in_games.games import braess_initial_network
    from recommenders import constant_recommender
    from learning_in_games.plot_functions import plot_run

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

    game = lambda x,y: braess_initial_network(x,y)

    M = single_run(braess_initial_network, N_AGENTS, N_STATES, N_ACTIONS, N_ITER, EPSILON, GAMMA, ALPHA, QINIT,
                   constant_recommender)

    NAME = f"run_N{N_AGENTS}_S{N_STATES}_A{N_ACTIONS}_I{N_ITER}_e{EPSILON}_g{GAMMA}_a{ALPHA}_q{QINIT}"

    plot_run(M, NAME, N_AGENTS, N_ACTIONS, N_ITER)
