# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def main():
    from functions import vecSOrun, plot_run, vecSOrun_states, vecSOrun_recommender
    import numpy as np
    import tqdm
    import pickle
    import nolds
    import pandas as pd

    # Base Settings Which Will Not Change
    N_AGENTS = 100
    N_STATES = 3
    N_ACTIONS = 3
    NEIGHBOURS = 0
    N_ITER = 100  #10000
    N_REPEATS = 1  #10
    mask = np.zeros(N_AGENTS)
    mask[:] = 1
    GAMMA = 0
    ALPHA = 0.01
    PAYOFF_TYPE = "SELFISH"  ## "SELFISH" or "SOCIAL"
    SELECT_TYPE = "EPSILON"  ## "EPSILON" or "GNET"
    WELFARE_TYPE = "AVERAGE"  ## "AVERAGE" or "MIN" or "MAX"

    # Parameters which will be Varied
    EPSILON = "Variable"
    sizeEpsilon = 1  #11
    epsilons = np.linspace(0, 0.15, sizeEpsilon)
    
    # recommender parameters
    recommender_threshold = -1.7
    
    QINIT = "Variable"
    sizeQinit = 2
    qinits = {
        "uniform": "UNIFORM",
        "nash": np.array([-2, -2, -2]),
        # "cdu": np.array([-2, -1.5, -1]),
        # "cud": np.array([-1.5, -2, -1]),
        # "ucd": np.array([-1, -2, -1.5]),
        # "udc": np.array([-1, -1.5, -2]),
        # "dcu": np.array([-2, -1, -1.5]),
        # "duc": np.array([-1.5, -1, -2])
    }

    NAME = f"sweep_e{sizeEpsilon}_q{sizeQinit}_{PAYOFF_TYPE}_{SELECT_TYPE}_{WELFARE_TYPE}_N{N_AGENTS}_S{N_STATES}_A{N_ACTIONS}_n{NEIGHBOURS}_I{N_ITER}_e{EPSILON}_g{GAMMA}_a{ALPHA}_q{QINIT} "

    results = []

    for i, e in enumerate(tqdm.tqdm(epsilons)):
        for norm, initTable in qinits.items():
            for random_recommender in [False, True, None]:
                for t in range(0, N_REPEATS):
                    M, Q = vecSOrun_recommender(N_AGENTS, N_STATES, N_ACTIONS, N_ITER, e, GAMMA, ALPHA, initTable,
                                                PAYOFF_TYPE, SELECT_TYPE, recommender_threshold, random_recommender)
                    W = [M[t]["R"].mean() for t in range(0, N_ITER)]
                    L = nolds.lyap_r(W)
                    T = np.mean(W[int(0.8 * N_ITER):N_ITER])
                    T_std = np.std(W[int(0.8 * N_ITER):N_ITER])

                    groups = [M[t]["groups"] for t in range(0, N_ITER)]
                    groups_mean = np.mean(groups)
                    groups_var = np.var(groups)
                    Qvar = [M[t]["Qvar"] for t in range(0, N_ITER)]
                    Qvar_mean = np.mean(Qvar)

                    # ignore transient phase (approximately 20% of first timesteps)
                    above_threshold = np.where(np.array(W[int(0.2 * N_ITER):N_ITER]) >= recommender_threshold, 1, 0)
                    stabilization_score = above_threshold.mean()

                    # M, Q = vecSOrun_states(N_AGENTS, N_STATES, N_ACTIONS, NEIGHBOURS, 1, 0, mask, GAMMA, ALPHA, Q,
                    #                        PAYOFF_TYPE, SELECT_TYPE)
                    # oneShot = np.mean(M[0]["R"])

                    row = {
                        "epsilon": e,
                        "norm": norm,
                        "T_mean": T,
                        "T_std": T_std,
                        "Lyapunov": L,
                        "repetition": t,
                        # "oneShot": oneShot,
                        "groups_mean": groups_mean,
                        "groups_var": groups_var,
                        "Qvar_mean": Qvar_mean,
                        "random_recommender": random_recommender,
                        "stabilization_score": stabilization_score
                    }

                    results.append(row)

    df = pd.DataFrame(results)

    df.to_csv(NAME + ".csv")


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
