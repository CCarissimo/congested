import os
import pickle
import numpy as np
import multiprocessing as mp
from tqdm.auto import tqdm


def get_unique_filename(base_filename):
    if not os.path.exists(base_filename):
        return base_filename

    filename, ext = os.path.splitext(base_filename)
    index = 1
    while True:
        new_filename = f"{filename}_{index}{ext}"
        if not os.path.exists(new_filename):
            return new_filename
        index += 1


def save_pickle_with_unique_filename(data, filename):
    unique_filename = get_unique_filename(filename)
    with open(unique_filename, 'wb') as file:
        pickle.dump(data, file)


def save_numpy_array_with_unique_filename(data, filename):
    unique_filename = get_unique_filename(filename)
    np.save(unique_filename, data)


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
