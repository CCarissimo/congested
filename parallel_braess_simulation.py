#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import pandas as pd
import itertools
import multiprocessing as mp
import re
import os
from tqdm import tqdm
import pickle
from dataclasses import asdict
from braess_deviation_run import *
from learning_in_games.games import braess_augmented_network


def player1params_dirname(alpha, epsilon, gamma):
    return f"/params_a({alpha})_e({epsilon})_g({gamma})/"


def run_and_store_one_setting(args):

    base_addr, repeat_count, params = args
    records = {}
    file_name = str(params)

    # extract player 1 parameters to create directory for player 1 settings
    sub_directory_name = player1params_dirname(params.alpha, params.epsilon, params.gamma)
    save_path = base_addr + sub_directory_name
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)

    # create tmp directory
    # os.makedirs(save_path + "/" + file_name)
    extracted_records = []
    for i in range(repeat_count):
        run_results = run_deviation_braess(
            params.n_iter,
            params.n_agents,
            params.q_initial,
            params.alpha,
            params.alpha_deviator,
            params.epsilon,
            params.epsilon_deviator,
            params.gamma,
            params.gamma_deviator,
            params.number_of_deviators
        )
        # records[i] = run_results
        extracted_records.append(record2df(run_results, params, i))

        q_table = run_results[params.n_iter]["Q"]
        np.savez_compressed(f"{save_path}{file_name}_run{i}", a=q_table)

    # records["params"] = params

    # with open(f"{save_path}{file_name}.pkl", "wb") as file:
    #     pickle.dump(records, file)

    return extracted_records


def record2df(record, params, repeat_no):
    frame = asdict(params)

    exclusion_threshold = 0.2
    n_dev = params.number_of_deviators

    W = np.array([record[t]["R"].mean() for t in range(0, params.n_iter)])
    welfare = np.mean(W)
    median = np.median(W)
    std = np.std(W)

    deviators = np.array([record[t]["R"][0:n_dev].mean() for t in record.keys()])
    deviator_average = np.mean(deviators)
    deviator_average_half = np.mean(deviators[int(params.n_iter / 2):])
    deviator_average_quarter = np.mean(deviators[int(params.n_iter / 4):])

    non_deviators = np.array([record[t]["R"][n_dev:].mean() for t in record.keys()])
    non_deviator_average = np.mean(non_deviators)
    non_deviator_average_half = np.mean(non_deviators[int(params.n_iter / 2):])
    non_deviator_average_quarter = np.mean(non_deviators[int(params.n_iter / 4):])

    increase, decrease = increase_decrease_size(W)
    up_len, down_len = increase_decrease_run_length(W)
    counts = drop_count_measure(W[int(exclusion_threshold * params.n_iter):-1])
    probability = simple_probability_measure(W[int(exclusion_threshold * params.n_iter):-1])

    # final one-shot evaluation
    q_tables = record[params.n_iter]["Q"]
    indices = np.arange(params.n_agents)
    S = np.zeros(params.n_agents)
    A = q_tables[indices, S, :].argmax(axis=1)
    R = braess_augmented_network(A, params.n_agents, cost=0)
    one_shot_welfare = np.mean(R)
    one_shot_welfare_deviator = np.mean(R[0:n_dev])
    one_shot_welfare_non_deviator = np.mean(R[n_dev:])

    row = {
        "repetition": repeat_no,
        "welfare": welfare,
        "std": std,
        "median": median,
        "deviator_average": deviator_average,
        "deviator_average_half": deviator_average_half,
        "deviator_average_quarter": deviator_average_quarter,
        "non_deviator_average": non_deviator_average,
        "non_deviator_average_half": non_deviator_average_half,
        "non_deviator_average_quarter": non_deviator_average_quarter,
        "one_shot": one_shot_welfare,
        "one_shot_deviator": one_shot_welfare_deviator,
        "one_shot_non_deviator": one_shot_welfare_non_deviator,
        "increase": increase,
        "decrease": decrease,
        "counts": counts,
        "probability": probability,
        "up_len": up_len,
        "down_len": down_len,
    }

    frame.update(row)

    return frame


def multi_file_simulation(settings, base_addr, repeat_count, num_processes):
    args_list = [[base_addr, repeat_count, params] for params in settings]
    return run_apply_async_multiprocessing(run_and_store_one_setting, args_list, num_processes)


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


def flatten(xss):
    return [x for xs in xss for x in xs]


if __name__ == '__main__':
    n_iter = [10 ** 2]  # I suggest to reduce it to 10**4
    n_agents = [100]
    q_init = ["UNIFORM"]
    repeat_count = 40
    alpha = [0.1]
    alpha_deviators = [0.1]
    epsilon = [0.01]
    epsilon_deviators = [0]
    gamma = [0.1]
    gamma_deviators = [0.22]
    population_threshold = [1]  # [2, 3, 6, 12, 25, 50]

    settings = [
        DeviationBraessExperimentConfig(I, N, Q, a, a_, e, e_, g, g_)
        for I, N, Q, a, a_, e, e_, g, g_ in itertools.product(
            n_iter,
            n_agents,
            q_init,
            alpha,
            alpha_deviators,
            epsilon,
            epsilon_deviators,
            gamma,
            gamma_deviators,
            population_threshold
        )
    ]

    num_cpus = int(os.cpu_count())  # specific for euler cluster
    print("identified cpus", num_cpus)
    worker_count = num_cpus

    addr = "/Users/ccarissimo/data/test_braess/data"

    results = multi_file_simulation(settings, addr, repeat_count, num_processes=num_cpus)

    results = flatten(results)
    df = pd.DataFrame(results)
    destination = "/Users/ccarissimo/data/test_braess/test_outfile.csv"
    df.to_csv(destination)

    # convert_directory(addr, "/cluster/home/ccarissimo/Bachelors_Project_Simulations/Utils/Simulations/duopoly_5_full_sweep_10e6_2.csv", RecordUtils.RecordMode.PICKLE)
