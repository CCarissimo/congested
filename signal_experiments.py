import pandas as pd
from learning_in_games.games import braess_augmented_network
from learning_in_games.running import *
from learning_in_games.agents import *
from pathlib import Path
from learning_in_games import utilities
import math
from dataclasses import dataclass
from typing import Union
from large_braess_experiments import *


@dataclass
class SignalExperimentConfig:
    signal_type: str
    signal_param: Union[float, None]
    n_agents: int
    n_states: int
    n_actions: int
    n_iter: int
    epsilon: Union[float, str]
    alpha: Union[np.ndarray, float]
    gamma: float
    q_initial: Union[np.ndarray, str]
    qmin: float
    qmax: float
    path: str
    name: str
    repetitions: int


def imitate(Q, S, A, reward_per_action, S_, n_agents, n_actions, alpha, gamma):
    for i in range(n_actions):
        ind = np.where(A != i)[0]  # [0] for the array of indices
        if type(alpha) is np.ndarray:
            sub_alpha = alpha[ind]
        else:
            sub_alpha = alpha
        other_actions = (np.ones(n_agents) * i).astype(int)
        other_rewards = np.ones(n_agents) * reward_per_action[i]
        Q, _ = bellman_update_q_table(ind, Q, S, other_actions, other_rewards, S_, sub_alpha, gamma)
    return Q


def run_signal_game(config: SignalExperimentConfig):
    Q = initialize_q_table(config.q_initial, config.n_agents, config.n_states, config.n_actions, config.qmin, config.qmax)

    eps_decay = config.n_iter / 8
    if config.epsilon == "DECAYED":
        eps_start = 1
        eps_end = 0
    else:
        eps_start = config.epsilon
        eps_end = config.epsilon

    ind = np.arange(config.n_agents)
    S = np.random.randint(config.n_states, size=config.n_agents)
    S_ = S

    data = {}
    for t in range(config.n_iter):
        epsilon = (eps_end + (eps_start - eps_end) * math.exp(-1. * t / eps_decay))  # if t < N_ITER/10 else 0
        A = e_greedy_select_action(Q, S, epsilon)
        R, _, reward_per_action = large_braess_network(A, paths, adj, config.n_agents)

        if config.signal_type == "mean_threshold":
            S_ = np.ones(n_agents).astype(int) if R.mean() < -config.signal_param else np.zeros(n_agents).astype(int)
        elif config.signal_type == "argmax_cross":
            S_ = np.where(Q[ind, S].argmax(axis=1) == 2, 1, 0)
        elif config.signal_type == "argmax":
            S_ = Q[ind, S].argmax(axis=1).astype(int)
        elif config.signal_type == "argmax_switch":
            S_ = np.where(Q[ind, S].argmax(axis=1) == 2, np.logical_not(S), S)
        else:
            print(f"{config.signal_type} signal type not found")
            break

        Q, sum_of_belief_updates = bellman_update_q_table(ind, Q, S, A, R, S_, config.alpha, config.gamma)

        S = S_
        ## SAVE PROGRESS DATA
        data[t] = {
                   "R": R,
                   "Qmean": Q.mean(axis=1).mean(axis=0),
                   # "groups": count_groups(Q[ind, S, :], 0.1),
                   "Qvar": Q[ind, S, :].var(axis=0),
                   # "A": A,
                   "Q": Q,
                   }
    return data


def increase_decrease_size(W):
    differences = np.diff(W)
    increase_indices = np.where(differences >= 0)
    decrease_indices = np.where(differences < 0)

    return differences[increase_indices].mean(), differences[decrease_indices].mean()


def main(config: SignalExperimentConfig):
    all_repetitions = []
    for i in range(config.repetitions):
        # run
        M = run_signal_game(config)
        full_path = f"{config.path}/{config.name}"
        Path(full_path).mkdir(parents=True, exist_ok=True)

        # save
        timeseries = np.stack([M[t]["R"].mean() for t in M.keys()])
        utilities.save_numpy_array_with_unique_filename(timeseries, f"{full_path}/timeseries_{i}.npy")

        # calculate features
        exclusion_threshold = 0.8
        W = np.array([M[t]["R"].mean() for t in range(0, config.n_iter)])
        T = np.mean(W[int(exclusion_threshold * config.n_iter):config.n_iter])
        T_all = np.mean(W)
        median = np.median(W[int(exclusion_threshold * config.n_iter):config.n_iter])
        T_std = np.std(W[int(exclusion_threshold * config.n_iter):config.n_iter])
        increase, decrease = increase_decrease_size(W)
        Qvar = [M[t]["Qvar"] for t in range(0, config.n_iter)]
        Qvar_mean = np.mean(Qvar)

        # store features
        row = {
            "signal_type": config.signal_type,
            "signal_param": config.signal_param,
            "repetition": i,
            "n_agents": config.n_agents,
            "alpha": config.alpha,
            "epsilon": config.epsilon,
            "increase": increase,
            "decrease": decrease,
            "T_mean": T,
            "T_mean_all": T_all,
            "T_std": T_std,
            "median": median,
            "Qvar_mean": Qvar_mean,
        }
        all_repetitions.append(row)

    return all_repetitions


def run_apply_async_multiprocessing(func, argument_list, num_processes):
    pool = mp.Pool(processes=num_processes)

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


if __name__ == '__main__':
    import argparse
    import multiprocessing as mp
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    Path(args.path).mkdir(parents=True, exist_ok=True)

    path = args.path
    n_agents = 100
    n_states = 2
    n_actions = 20
    n_iter = 100000
    epsilon = "DECAYED"
    alpha = 0.01
    gamma = 0.95
    q_initial = "UNIFORM"
    qmin = -4
    qmax = -2
    repetitions = 40

    num_cpus = int(os.environ.get("SLURM_NTASKS", os.cpu_count()))  # specific for euler cluster
    precision = 3
    print(f"Found {num_cpus} processors to use")
    argument_list = []
    for signal_type in ["mean_threshold", "argmax"]:
        for signal_param in np.linspace(2.5, 3, 100):
            if signal_type == "argmax":
                n_states = 20
            experiment_name = f"signal_type{signal_type}_param{signal_param:.{precision}f}"
            experiment_config = SignalExperimentConfig(
                signal_type=signal_type,
                signal_param=signal_param,
                n_agents=n_agents,
                n_states=n_states,
                n_actions=n_actions,
                n_iter=n_iter,
                epsilon=epsilon,
                alpha=alpha,
                gamma=gamma,
                q_initial=q_initial,
                qmin=qmin,
                qmax=qmax,
                path=path,
                name=experiment_name,
                repetitions=repetitions
            )

            argument_list.append(experiment_config)
            if signal_type != "mean_threshold":
                break  # should skip all other params

    results = run_apply_async_multiprocessing(main, argument_list=argument_list, num_processes=num_cpus)

    utilities.save_pickle_with_unique_filename(results, f"{path}/results.pkl")
    name = f"results.csv"
    unique_name = utilities.get_unique_filename(base_filename=name)
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{path}/{unique_name}", index=False)
    print(f"saving to {path}/{unique_name}")
