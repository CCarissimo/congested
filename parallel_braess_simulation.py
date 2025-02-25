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

from braess_deviation_run import run_deviation_braess, DeviationBraessExperimentConfig


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
            params.gamma_deviator
        )
        records[i] = run_results
        extracted_records.append(record2df(run_results, params, i))

    records["params"] = params

    with open(f"{save_path}{file_name}.pkl", "wb") as file:
        pickle.dump(records, file)

    return extracted_records


def record2df(record, params, repeat_no):
    frame = asdict(params)

    W = np.array([record[t]["R"].mean() for t in range(0, params.n_iter)])
    welfare = np.mean(W)
    median = np.median(W)
    std = np.std(W)

    user0 = np.array([record[t]["R"][0].mean() for t in record.keys()])
    deviator_average = np.mean(user0)
    non_deviators = np.array([record[t]["R"][1:].mean() for t in record.keys()])
    non_deviator_average = np.mean(non_deviators)

    row = {
        "repetition": repeat_no,
        "welfare": welfare,
        "std": std,
        "median": median,
        "deviator_average": deviator_average,
        "non_deviator_average": non_deviator_average,
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


def file2df(file_addr, mode):
    file_addr = file_addr.strip("'")
    # file = RecordUtils.read_record(file_addr, mode)
    with open(file_addr, "rb") as file:
        file = pickle.load(file)
    file = file['records']
    dfs = []
    for repeat_no in file.keys():
        rec_df = record2df(file[repeat_no], repeat_no)
        dfs.append(rec_df)
    return pd.concat(dfs)


def all2df(dir_addr, mode):
    dfs = []
    for file_name in tqdm(os.listdir(dir_addr)):
        if re.match('.+\.pkl', file_name):
            path = dir_addr + str(file_name)
            file_df = file2df(file_addr=path, mode=mode)
            dfs.append(file_df)

    return pd.concat(dfs)


def df2csv(df: pd.DataFrame, addr):
    df.to_csv(
        addr
    )


def convert_directory(dir_addr, dest, mode):
    df = all2df(dir_addr, mode)
    df2csv(df, dest)


# convert_directory('./TestData/', './test.csv', RecordUtils.RecordMode.PICKLE)


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
            gamma_deviators
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
    df2csv(df, destination)

    # convert_directory(addr, "/cluster/home/ccarissimo/Bachelors_Project_Simulations/Utils/Simulations/duopoly_5_full_sweep_10e6_2.csv", RecordUtils.RecordMode.PICKLE)
