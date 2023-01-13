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
    # print(rand)
    randA = np.random.randint(len(Q[0, 0, :]), size=len(S))
    # print(randA)
    A = np.where(rand >= epsilon,
                 np.argmax(Q[indices, S, :], axis=1),
                 randA)
    return A
