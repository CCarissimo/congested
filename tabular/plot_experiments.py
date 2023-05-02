import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval

######### DATA IMPORTS AND PREPROCESSING
path = "experiments/"
path2 = "aligned_experiments/"

frames = []
for file in glob.glob(path + "sweep_aligned_*.csv"):
    frames.append(pd.read_csv(file))

df = pd.concat(frames)
print(df.dtypes)
type(df["alignment_all"].iloc[0])
print(df["alignment_all"].iloc[0])
# df = df[df.recommender_type != "aligned_heuristic"]
#
# frames = []
# for file in glob.glob(path2 + "*.csv"):
#     frames.append(pd.read_csv(file))
#
# df = pd.concat([df] + frames)

# df["alignment"] = df["alignment"].replace(to_replace='[None, None, None]', value=None)
df["alignment"] = df["alignment"].astype("float64")
# df["alignment_all"] = df["alignment_all"].apply(lambda x: str(x).split(" "))
# df["alignment_all"] = df["alignment_all"].apply(lambda x: np.array(x))
# df["alignment_all"] = df["alignment_all"].apply(lambda x: np.mean(x))

means = df.groupby(["recommender_type", "norm", "epsilon"]).mean()
std = df.groupby(["recommender_type", "norm", "epsilon"]).std()
######### END OF DATA IMPORTS AND PREPROCESSING


def welfare(epsilon):
    return - (2 - 2 / 3 * epsilon + 2 / 9 * epsilon ** 2)


def plot_case(axis, recommender, initialization, quantity, linestyle):
    means.loc[recommender, initialization][quantity].plot(label=recommender, ax=axis, linestyle=linestyle, legend=False)
    axis.fill_between(np.linspace(0, 0.2, 11),
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

initialization = "uniform"
metric_mean = "T_mean"
metric_alignment = "alignment"

fig, [ax, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(6.75, 3.2))
sns.despine()

plot_case(ax, "none", initialization, metric_mean, linestyle="-")
# plot_case(ax, "naive", initialization, metric_mean, linestyle="-")
plot_case(ax, "random", initialization, metric_mean, linestyle="--")
# plot_case(ax, "heuristic", initialization, metric_mean, linestyle="-")
# plot_case(ax, "optimized_action_maximize", initialization, metric_mean, linestyle="--")
plot_case(ax, "optimized_estimate_maximize", initialization, metric_mean, linestyle=":")
plot_case(ax, "aligned_heuristic", initialization, metric_mean, linestyle=":")

ax.set_ylim((-1.45, -2.05))
ax.set_yticks(ticks=np.linspace(-1.5, -2, 5), labels=np.linspace(-1.5, -2, 5))

ax.set_xticks(ticks=np.linspace(0, 0.2, 5))

# ax.plot(x_vals, opt, linestyle="--", color="gray", linewidth=)
ax.axhline(-1.5, linestyle="--", color="gray", linewidth=0.5)
ax.annotate('social optimum', xy=(0.55, 0.09), xycoords='axes fraction')
# ax.plot(x_vals, opt-0.5, linestyle="--", color="gray")
ax.axhline(-2, linestyle="--", color="gray", linewidth=0.5)
ax.annotate('nash equilibrium', xy=(0.505, 0.925), xycoords='axes fraction')
# ax.plot(x_vals, y_vals, label="irrational agents", linestyle=":", color="black")
ax.set_xlabel(r"exploration rate ($\epsilon$)", **{"fontname": "Times New Roman", "fontsize": "x-large"})
ax.set_ylabel(r"average latency", **{"fontname": "Times New Roman", "fontsize": "x-large"})
# ax.legend(prop={"family": "Times New Roman", "size": "x-large"}, title="recommender type", bbox_to_anchor=(1.05, 1))
# plt.savefig("plots/recommenders_welfare_comparison.pdf")
# plt.show()

plot_case(ax2, "none", initialization, metric_alignment, linestyle="-")
# plot_case(ax2, "naive", initialization, metric_alignment, linestyle="-")
plot_case(ax2, "random", initialization, metric_alignment, linestyle="--")
# plot_case(ax2, "heuristic", initialization, metric_alignment, linestyle="-")
# plot_case(ax2, "optimized_action_maximize", initialization, metric_alignment, linestyle="--")
plot_case(ax2, "optimized_estimate_maximize", initialization, metric_alignment, linestyle=":")
plot_case(ax2, "aligned_heuristic", initialization, metric_alignment, linestyle=":")

ax2.set_ylim((0, 1.05))
ax2.set_xticks(ticks=np.linspace(0, 0.2, 5))

ax2.set_xlabel(r"exploration rate ($\epsilon$)", **{"fontname": "Times New Roman", "fontsize": "x-large"})
ax2.set_ylabel(r"recommendation alignment", **{"fontname": "Times New Roman", "fontsize": "x-large"})
# ax2.legend(prop={"family": "Times New Roman", "size": "x-large"}, title="recommender type", bbox_to_anchor=(1.05, 1))

handles, labels = ax.get_legend_handles_labels()
print(handles)
print(labels)
labels[2] = "heuristic"
labels[3] = "aligned heuristic"

fig.legend(bbox_to_anchor=(0.5, 0.95), loc='lower center', ncol=4, handles=handles, labels=labels)
fig.tight_layout()
fig.savefig(f"plots/recommenders_{initialization}_{metric_mean}_{metric_alignment}.pdf", bbox_inches="tight")
plt.show()
