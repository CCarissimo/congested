# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def main():
    from functions import vecSOrun, plot_run, vecSOrun_recommender, total_welfare, total_updates, vecSOrun_heuristic_recommender
    import numpy as np
    # from recommenders import vecSOrun_heuristic_recommender

    N_AGENTS = 100
    N_STATES = 3
    N_ACTIONS = 3
    N_ITER = 5000

    EPSILON = 0.15
    mask = np.zeros(N_AGENTS)
    mask[:] = 1
    GAMMA = 0
    ALPHA = 0.25

    QINIT = "UNIFORM"  # np.array([-2, -2, -2])

    PAYOFF_TYPE = "SELFISH"  ## "SELFISH" or "SOCIAL"
    SELECT_TYPE = "EPSILON"  ## "EPSILON" or "GNET"
    WELFARE_TYPE = "AVERAGE"  ## "AVERAGE" or "MIN" or "MAX"

    NAME = f"run_{PAYOFF_TYPE}_{SELECT_TYPE}_{WELFARE_TYPE}_N{N_AGENTS}_S{N_STATES}_A{N_ACTIONS}_I{N_ITER}_e{EPSILON}_g{GAMMA}_a{ALPHA}_q{QINIT}"

    random_recommender = False
    recommender_objective = None

    M, Q = vecSOrun_heuristic_recommender(N_AGENTS, N_STATES, N_ACTIONS, N_ITER, EPSILON, GAMMA, ALPHA, QINIT,
                                                PAYOFF_TYPE, SELECT_TYPE, random_recommender, recommender_objective)
    T0 = np.mean([M[t]["R"] for t in range(int(0.8 * N_ITER), N_ITER)])
    T1 = np.mean([np.mean(M[t]["R"]) for t in range(int(0.8 * N_ITER), N_ITER)])

    print("T0", T0, "T1", T1)

    plot_run(M, NAME, N_AGENTS, N_ACTIONS, N_ITER)


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
