import matplotlib.pyplot as plt
import numpy as np


def welfare(R, N_AGENTS, welfareType="AVERAGE"):
    if welfareType == "AVERAGE":
        return R.sum() / N_AGENTS
    elif welfareType == "MIN":
        return R.min()
    elif welfareType == "MAX":
        return R.max()
    else:
        raise "SPECIFY WELFARE TYPE"


def plot_run(M, NAME, n_agents, n_actions, n_iter):
    ## PLOTTING EVOLUTION
    print(NAME)

    cmap = plt.get_cmap('plasma')
    a_labels = ["up", "down", "cross"] if n_actions == 3 else ["up", "down"]

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    W = [welfare(M[t]["R"], n_agents, "AVERAGE") for t in range(n_iter)]

    ax[0, 0].plot(W, color=cmap(0.5))
    # ax[0, 0].set_ylim((-2, -1))
    ax[0, 0].set_xlabel('t')
    ax[0, 0].set_ylabel('welfare')
    ax[0, 0].set_title("Average Travel Time")
    # ax[0, 0].plot(np.arange(0, n_iter, ))

    x_vals = np.arange(0, n_iter)
    T = {}
    for a in range(n_actions):
        T[a] = [M[t]["T"][a] for t in range(n_iter)]

    ax[0, 1].set_prop_cycle(color=[cmap(c) for c in np.linspace(0.1, 0.9, n_actions)])

    for a in range(n_actions):
        ax[0, 1].scatter(x_vals, T[a], label=a_labels[a], alpha=0.4)
    # ax[0, 1].set_ylim((-2, -1))
    ax[0, 1].set_xlabel('t')
    ax[0, 1].set_ylabel('travel time')
    ax[0, 1].set_title("Min/Max Travel Time")
    ax[0, 1].legend()

    x_vals = np.arange(0, n_iter)
    nA = {}
    for a in range(n_actions):
        nA[a] = [M[t]["nA"][a] for t in range(n_iter)]

    ax[1, 0].set_prop_cycle(color=[cmap(c) for c in np.linspace(0.1, 0.9, n_actions)])

    for a in range(n_actions):
        ax[1, 0].scatter(x_vals, nA[a], label=a_labels[a], alpha=0.4)
    ax[1, 0].set_ylim((0, n_agents))
    ax[1, 0].set_xlabel('t')
    ax[1, 0].set_ylabel('number of actions')
    ax[1, 0].set_title("Action Profile")
    ax[1, 0].legend()

    Qmean = [M[t]["Qmean"] for t in range(n_iter)]

    ax[1, 1].set_prop_cycle(color=[cmap(c) for c in np.linspace(0.1, 0.9, n_actions)])

    ax[1, 1].plot(Qmean, label=a_labels)
    # ax[1, 1].set_ylim((-2, -1))
    ax[1, 1].set_xlabel('t')
    ax[1, 1].set_ylabel(r'$\hat{Q}(a)$')
    ax[1, 1].set_title(r"$\hat{Q}(a)$ Averaged over Drivers")
    ax[1, 1].legend()

    plt.savefig(NAME + ".png")
    plt.show()