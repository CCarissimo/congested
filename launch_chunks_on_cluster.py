import os
import numpy as np

if __name__ == "__main__":

    grid_size = 10
    xs = np.logspace(-3, -0.7, grid_size)

    for alpha in np.linspace(0.01, 1, grid_size):
        for epsilon in np.round(xs, 3):
            for gamma in np.linspace(0, 0.99, grid_size):
                os.system(f"sbatch --time=24:00:00 --ntasks=64 --mem-per-cpu=1G --wrap='python3 ./chunking_parameter_simulations.py {alpha} {epsilon} {gamma}'")
