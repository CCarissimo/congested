import os
import numpy as np
import glob


def all_parameter_combinations():
    filenames = []

    grid_size = 10
    xs = np.logspace(-3, -0.7, grid_size)

    for alpha in np.linspace(0.01, 1, grid_size):
        for epsilon in np.round(xs, 3):
            for gamma in np.linspace(0, 0.99, grid_size):
                filename = f"player1params_a({alpha})_e({epsilon})_g({gamma}).csv"
                filenames.append(filename)
    return filenames


if __name__ == "__main__":

    all_filenames = set(all_parameter_combinations())

    directory = "/cluster/work/coss/ccarissimo/braess_symmetric_meta_game_core/dataframes/"

    files_in_directory = set(glob.glob(directory + "*.csv"))

    missing_files = all_filenames - files_in_directory

    # os.system(f"sbatch --time=24:00:00 --ntasks=64 --mem-per-cpu=1G --wrap='python3 ./chunking_parameter_simulations.py {alpha} {epsilon} {gamma}'")


