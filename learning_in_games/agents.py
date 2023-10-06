import numpy as np


def bellman_update_q_table(Q, S, A, R, alpha, gamma):
    ind = np.arange(len(S))
    all_belief_updates = alpha * (R + gamma * Q[ind, S].max(axis=1) - Q[ind, S, A])
    Q[ind, S, A] = Q[ind, S, A] + all_belief_updates
    return Q, np.abs(all_belief_updates).sum()


def e_greedy_select_action(Q, S, epsilon):
    ## DETERMINE ACTIONS
    indices = np.arange(len(S))
    rand = np.random.random_sample(size=len(S))
    randA = np.random.randint(len(Q[0, 0, :]), size=len(S))
    A = np.where(rand >= epsilon,
                 np.argmax(Q[indices, S, :], axis=1),
                 randA)
    return A


def e_greedy_select_action_randomized_argmax(Q, S, epsilon):
    ## DETERMINE ACTIONS
    indices = np.arange(len(S))
    rand = np.random.random_sample(size=len(S))

    randA = np.random.randint(len(Q[0, 0, :]), size=len(S))

    rng = np.random.default_rng()
    argmax_actions = rng.choice(np.where(
        [np.isclose(vals.max(axis=0), vals) for vals in Q[indices, S, :]]
    ))
    A = np.where(rand >= epsilon, argmax_actions, randA)
    return A


def follow_the_regularized_leader_select_action(Q, S):
    ## DETERMINE ACTIONS
    indices = np.arange(len(S))
    regularization = np.exp(Q[indices, S])/np.exp(Q[indices, S]).sum()
    x = Q[indices, S] - regularization
    A = np.argmax(x, axis=1)
    return A


def average_rolled_q_tables(Q, neighborhood):
    # creates an array where sucessive rows are "rolled" versions of Q, along the axis of agents
    tmp_Q = np.array([np.roll(Q, shift=i, axis=0) for i in range(neighborhood)])
    Q = np.mean(tmp_Q, axis=0)
    return Q
