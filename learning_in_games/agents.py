import numpy as np


def bellman_update_q_table(Q, S, A, R, S_, alpha, gamma):
    """
    Performs a one-step update using the bellman update equation for Q-learning.
    :param Q: np.ndarray Q-table indexed by (agents, states, actions)
    :param S: np.ndarray States indexed by (agents)
    :param A: np.ndarray Actions indexed by (agents)
    :param R: np.ndarray Rewards indexed by (agents)
    :param S_: np.ndarray Next States indexed by (agents)
    :param alpha: float learning rate for Q-learning
    :param gamma: float discount parameter for Q-learning
    :return: np.ndarray Q-table indexed by (agents, states, actions)
    """
    ind = np.arange(len(S))
    all_belief_updates = alpha * (R + gamma * Q[ind, S_].max(axis=1) - Q[ind, S, A])
    Q[ind, S, A] = Q[ind, S, A] + all_belief_updates
    return Q, np.abs(all_belief_updates).sum()


def e_greedy_select_action(Q, S, epsilon):
    """
    Select actions based on an epsilon greedy policy. Epsilon determines the probability with which
    an action is selected at random. Otherwise, the action is selected as the argmax of the state.
    During the argmax operation and given a tie the argmax operator selects the first occurrence.
    :param Q: np.ndarray Q-table indexed by (agents, states, actions)
    :param S: np.ndarray States indexed by (agents)
    :param epsilon: float learning rate for Q-learning
    :return: np.ndarray Actions indexed by (agents)
    """
    indices = np.arange(len(S))
    rand = np.random.random_sample(size=len(S))
    randA = np.random.randint(len(Q[0, 0, :]), size=len(S))
    A = np.where(rand >= epsilon,
                 np.argmax(Q[indices, S, :], axis=1),
                 randA)
    return A


def e_greedy_select_action_randomized_argmax(Q, S, epsilon):
    """
    Select actions based on an epsilon greedy policy. Epsilon determines the probability with which
    an action is selected at random. Otherwise, the action is selected as the argmax of the state.
    Additionally, this function takes care of ties during the argmax operation, and selects at random
    in the event of a tie. Standard numpy argmax selects the first element of a tie.
    :param Q: np.ndarray Q-table indexed by (agents, states, actions)
    :param S: np.ndarray States indexed by (agents)
    :param epsilon: float learning rate for Q-learning
    :return: np.ndarray Actions indexed by (agents)
    """
    indices = np.arange(len(S))
    rand = np.random.random_sample(size=len(S))

    randA = np.random.randint(len(Q[0, 0, :]), size=len(S))

    rng = np.random.default_rng()
    argmax_actions = rng.choice(np.where(
        [np.isclose(vals.max(axis=0), vals) for vals in Q[indices, S, :]]
    ))
    A = np.where(rand >= epsilon, argmax_actions, randA)
    return A


def boltzman_select_action(Q, S, temperature):
    """
    Selects actions by drawing from a distribution determined by a softmax operator on the q-values.
    The temperature parameter approaching 0 leads to a distribution which approaches an argmax, while
    approaching infinity leads to the uniform random distribution.
    WARNING: low temperature values can easily lead to NaNs or infs.
    :param Q: np.ndarray Q-table indexed by (agents, states, actions)
    :param S: np.ndarray States indexed by (agents)
    :param temperature: float (0,+inf] temperature for the Q-learning action selection
    :return: np.ndarray Actions indexed by (agents)
    """
    indices = np.arange(len(S)).astype(int)
    rng = np.random.default_rng()
    values_exponential = np.exp(Q[indices, S]/temperature)
    denominator = np.sum(values_exponential, axis=1)
    probabilities = np.divide(values_exponential.T, denominator).T
    actions = np.array(
        [rng.choice(
            a=Q.shape[-1], #n_actions
            p=probabilities[i]
        ) for i in range(Q.shape[0])] #n_agents
    )
    return actions


def follow_the_regularized_leader_select_action(Q, S, temperature):
    """
    Selects actions based on the follow the regularized leader algorithms which apply an argmax operation
    to q-values which have been regularized with a softmax operation.
    :param Q: np.ndarray Q-table indexed by (agents, states, actions)
    :param S: np.ndarray States indexed by (agents)
    :param temperature: float (0,+inf] temperature for the Q-learning action selection
    :return: np.ndarray Actions indexed by (agents)
    """
    indices = np.arange(len(S))
    values_exponential = np.exp(Q[indices, S] / temperature)
    denominator = np.sum(values_exponential, axis=1)
    regularization = np.divide(values_exponential.T, denominator).T
    x = Q[indices, S] - regularization
    A = np.argmax(x, axis=1)
    return A


def average_rolled_q_tables(Q, neighborhood):
    """
    Creates an array where successive rows are "rolled" versions of Q, along the axis of agents.
    This is created and used for experiments where agents average their q-values with each others for
    different sizes of neighborhoods.
    :param Q: np.ndarray Q-table indexed by (agents, states, actions)
    :param neighborhood:
    :return: np.ndarray Q-table with averaged entries indexed by (agents, states, actions)
    """
    tmp_Q = np.array([np.roll(Q, shift=i, axis=0) for i in range(neighborhood)])
    Q = np.mean(tmp_Q, axis=0)
    return Q
