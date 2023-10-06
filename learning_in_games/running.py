import numpy as np
import scipy.cluster
from tqdm.auto import tqdm
import multiprocessing as mp


def initialize_q_table(q_initial, n_agents, n_states, n_actions, qmin=0, qmax=1):
    if type(q_initial) == np.ndarray:
        if q_initial.shape == (n_agents, n_states, n_actions):
            q_table = q_initial
        else:
            q_table = q_initial.T * np.ones((n_agents, n_states, n_actions))
    elif q_initial == "UNIFORM":
        q_table = (qmax-qmin)*np.random.random_sample(size=(n_agents, n_states, n_actions)) + qmin
    elif q_initial == "ALIGNED":
        if n_actions == 3:
            q_table = np.array([[-1, -2, -2], [-2, -1, -2], [-2, -2, -1]]).T * np.ones((n_agents, n_states, n_actions))
        elif n_actions == 2:
            q_table = np.array([[-1, -2], [-2, -1]]).T * np.ones((n_agents, n_states, n_actions))
    elif q_initial == "MISALIGNED":
        if n_actions == 3:
            q_table = np.array([[-2, -1, -2], [-2, -2, -1], [-1, -2, -2]]).T * np.ones((n_agents, n_states, n_actions))
        elif n_actions == 2:
            q_table = np.array([[-2, -1], [-1, -2]]).T * np.ones((n_agents, n_states, n_actions))
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


def count_groups(q_values, dist):
    y = scipy.cluster.hierarchy.average(q_values)
    z = scipy.cluster.hierarchy.fcluster(y, dist, criterion='distance')
    groups = np.bincount(z)
    return len(groups)


# def calculate_alignment(q_table):
#     argmax_q_table = np.argmax(q_table, axis=2)
#     return (argmax_q_table == np.broadcast_to(np.arange(q_table.shape[2]), (q_table.shape[0], q_table.shape[1]))).mean(axis=0)


def calculate_alignment(q_table, recommendation, actions):
    argmax_q_table = np.argmax(q_table, axis=2)
    belief_alignment = (argmax_q_table == np.broadcast_to(np.arange(q_table.shape[2]), (q_table.shape[0], q_table.shape[1]))).mean(axis=0)
    recommendation_alignment = (recommendation == argmax_q_table[np.arange(q_table.shape[0]), recommendation]).mean()
    action_alignment = (recommendation==actions).mean()
    return belief_alignment, recommendation_alignment, action_alignment


def run_apply_async_multiprocessing(func, argument_list, num_processes=None):
    if num_processes:
        pool = mp.Pool(processes=num_processes)
    else:
        pool = mp.Pool(processes=mp.cpu_count())

    jobs = [
        pool.apply_async(func=func, args=(*argument,)) if isinstance(argument, tuple) else pool.apply_async(func=func,
                                                                                                            args=(
                                                                                                            argument,))
        for argument in argument_list]
    pool.close()
    result_list_tqdm = []
    for job in tqdm(jobs):
        result_list_tqdm.append(job.get())

    return result_list_tqdm
