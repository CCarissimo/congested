import argparse
import os
from parallel_braess_simulation import multi_file_simulation, flatten, df2csv
from braess_deviation_run import DeviationBraessExperimentConfig
import itertools
import pandas as pd
import numpy as np


if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description="input simulation parameters")

    # Add arguments
    parser.add_argument('alpha', type=float, help="learning rate")
    parser.add_argument('epsilon', type=float, help="exploration rate")
    parser.add_argument('gamma', type=float, help="discount factor")

    # Parse the arguments
    args = parser.parse_args()

    main_dir = "/cluster/work/coss/ccarissimo/braess_symmetric_meta_game/"
    # main_dir = "test_multiprocessing/"
    data_addr = f"{main_dir}data/"
    if not os.path.isdir(data_addr):
        os.mkdir(data_addr)
    dataframes_addr = f"{main_dir}dataframes/"
    if not os.path.isdir(dataframes_addr):
        os.mkdir(dataframes_addr)

    num_cpus = int(os.environ.get("SLURM_NTASKS", os.cpu_count()))  # specific for euler cluster
    print("identified cpus", num_cpus)

    n_iter = [4*(10**4)]  # I suggest to reduce it to 10**4
    n_agents = [100]
    q_init = ["UNIFORM"]
    repeat_count = 20
    alpha_deviators = np.linspace(0.01, 1, 10)
    epsilon_deviators = np.linspace(0, 0.5, 10)
    gamma_deviators = np.linspace(0, 0.99, 10)

    settings = [
        DeviationBraessExperimentConfig(I, N, Q, a, a_, e, e_, g, g_)
        for I, N, Q, a, a_, e, e_, g, g_ in itertools.product(
            n_iter,
            n_agents,
            q_init,
            [args.alpha],
            alpha_deviators,
            [args.epsilon],
            epsilon_deviators,
            [args.gamma],
            gamma_deviators
        )
    ]

    results = multi_file_simulation(settings, data_addr, repeat_count, num_processes=num_cpus)

    results = flatten(results)
    df = pd.DataFrame(results)
    filename = f"player1params_a({args.alpha})_e({args.epsilon})_g({args.gamma}).csv"
    destination = dataframes_addr + filename
    df2csv(df, destination)
