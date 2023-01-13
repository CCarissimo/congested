import numpy as np
import scipy.cluster


def initialize_q_table(q_initial, n_agents, n_states, n_actions, trusting=False):
    if trusting:
        q_table = -1 * np.array([np.identity(n_actions) for i in range(n_agents)])

    if type(q_initial) == np.ndarray:
        if q_initial.shape == (n_agents, n_states, n_actions):
            q_table = q_initial
        else:
            q_table = q_initial.T * np.ones((n_agents, n_states, n_actions))
    elif q_initial == "UNIFORM":
        q_table = - np.random.random_sample(size=(n_agents, n_states, n_actions)) - 1

    return q_table


def initialize_learning_rates(alpha, n_agents):
    if alpha == "UNIFORM":
        alpha = np.random.random_sample(size=n_agents)
    return alpha


def initialize_exploration_rates(epsilon, n_agents, mask=1):  # default mask 1 leads to no change
    if epsilon == "UNIFORM":
        epsilon = np.random.random_sample(size=n_agents) * mask
    else:
        epsilon = epsilon * np.ones(n_agents) * mask
    return epsilon


def welfare(R, N_AGENTS, welfareType="AVERAGE"):
    if welfareType == "AVERAGE":
        return R.sum() / N_AGENTS
    elif welfareType == "MIN":
        return R.min()
    elif welfareType == "MAX":
        return R.max()
    else:
        raise "SPECIFY WELFARE TYPE"


def count_groups(Q_values, dist):
    y = scipy.cluster.hierarchy.average(Q_values)
    z = scipy.cluster.hierarchy.fcluster(y, dist, criterion='distance')
    groups = np.bincount(z)
    return len(groups)

