import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns

######### DATA IMPORTS AND PREPROCESSING
path = "experiments/"
path2 = "aligned_experiments/"

frames = []
for file in glob.glob(path + "*.csv"):
    frames.append(pd.read_csv(file))

df = pd.concat(frames)
df = df[df.recommender_type != "aligned_heuristic"]

frames = []
for file in glob.glob(path2 + "*.csv"):
    frames.append(pd.read_csv(file))

df = pd.concat([df] + frames)

df["alignment"] = df["alignment"].replace(to_replace='[None, None, None]', value=None)
df["alignment"] = df["alignment"].astype("float64")

means = df.groupby(["recommender_type", "norm", "epsilon"]).mean()
std = df.groupby(["recommender_type", "norm", "epsilon"]).std()
######### END OF DATA IMPORTS AND PREPROCESSING


def welfare(epsilon):
    return - (2 - 2 / 3 * epsilon + 2 / 9 * epsilon ** 2)


def plot_case(axis, recommender, initialization, quantity, linestyle):
    l, = means.loc[recommender, initialization][quantity].plot(legend=True, label=recommender, ax=axis, linestyle=linestyle)
    l, = axis.fill_between(np.linspace(0, 0.2, 11),
                     means.loc[recommender, initialization][quantity] + std.loc[recommender, initialization][quantity],
                     means.loc[recommender, initialization][quantity] - std.loc[recommender, initialization][quantity],
                     alpha=0.2)


norms = df.norm.unique().tolist()
opt = np.ones(21) * -1.5
x_vals = np.linspace(0, 0.2, 21)
y_vals = welfare(x_vals)

import matplotlib.pylab as pylab

sns.set_context("paper")
plt.style.use('paper.mplstyle')
# cmap = plt.get_cmap('plasma')
sns.set_palette("magma")
# colors = [cmap(c) for c in np.linspace(0.1, 0.9, n_actions)]

recommenders = ["none", "random", "optimized_estimate_maximize", "aligned_heuristic"]

fig, [ax, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(6.75, 3.2))

plot_case(ax, "none", "uniform", "T_mean_all", linestyle="-")
# plot_case(ax, "naive", "uniform", "T_mean_all", linestyle="-")
plot_case(ax, "random", "uniform", "T_mean_all", linestyle="--")
# plot_case(ax, "heuristic", "uniform", "T_mean_all", linestyle="-")
# plot_case(ax, "optimized_action_maximize", "uniform", "T_mean_all", linestyle="--")
plot_case(ax, "optimized_estimate_maximize", "uniform", "T_mean_all", linestyle=":")
plot_case(ax, "aligned_heuristic", "uniform", "T_mean_all", linestyle=":")

ax.set_ylim((-1.4, -2.1))
ax.set_yticks(ticks=np.linspace(-1.4, -2.1, 5), labels=np.linspace(1.4, 2.1, 5))

ax.set_xticks(ticks=np.linspace(0, 0.2, 5))

ax.plot(x_vals, opt, label="social optimum", linestyle="--", color="gray")
ax.plot(x_vals, y_vals, label="irrational agents", linestyle=":", color="black")
ax.set_xlabel(r"irrationality / exploration rate ($\epsilon$)", **{"fontname": "Times New Roman", "fontsize": "x-large"})
ax.set_ylabel(r"average travel time ($\bar{T}$)", **{"fontname": "Times New Roman", "fontsize": "x-large"})
# ax.legend(prop={"family": "Times New Roman", "size": "x-large"}, title="recommender type", bbox_to_anchor=(1.05, 1))
# plt.savefig("plots/recommenders_welfare_comparison.pdf")
# plt.show()

plot_case(ax2, "none", "uniform", "alignment", linestyle="-")
# plot_case(ax2, "naive", "uniform", "alignment", linestyle="-")
plot_case(ax2, "random", "uniform", "alignment", linestyle="--")
# plot_case(ax2, "heuristic", "uniform", "alignment", linestyle="-")
# plot_case(ax2, "optimized_action_maximize", "uniform", "alignment", linestyle="--")
plot_case(ax2, "optimized_estimate_maximize", "uniform", "alignment", linestyle=":")
plot_case(ax2, "aligned_heuristic", "uniform", "alignment", linestyle=":")

ax2.set_ylim((0, 1))
ax2.set_xticks(ticks=np.linspace(0, 0.2, 5))

ax2.set_xlabel(r"irrationality / exploration rate ($\epsilon$)", **{"fontname": "Times New Roman", "fontsize": "x-large"})
ax2.set_ylabel(r"recommendation to argmax alignment", **{"fontname": "Times New Roman", "fontsize": "x-large"})
# ax2.legend(prop={"family": "Times New Roman", "size": "x-large"}, title="recommender type", bbox_to_anchor=(1.05, 1))

fig.legend(bbox_to_anchor=(0.5,0.95), loc='lower center', ncol = 4, handles = handles)
# fig.tight_layout()
plt.savefig("plots/recommenders_alignment_comparison.pdf")
plt.show()
