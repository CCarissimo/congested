import os
import numpy as np

if __name__ == "__main__":

    grid_size = 10
    for alpha in np.linspace(0.01, 1, grid_size):
        for epsilon in np.linspace(0, 0.5, grid_size):
            for gamma in np.linspace(0, 0.99, grid_size):
                os.system(f"sbatch --time=12:00:00 --ntasks=32 --mem-per-cpu=4G --wrap='python3 ./chunking_parameter_simulations.py {alpha} {epsilon} {gamma}'")
