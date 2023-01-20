import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# from run_functions import welfare

path = "experiments/"

frames = []
for file in glob.glob(path + "*.csv"):
    frames.append(pd.read_csv(file))

df = pd.concat(frames)

df["alignment"] = df["alignment"].replace(to_replace='[None, None, None]', value=None)
df["alignment"] = df["alignment"].astype("float64")

means = df.groupby(["recommender_type", "norm", "epsilon"]).mean()
std = df.groupby(["recommender_type", "norm", "epsilon"]).std()


def welfare(epsilon):
    return - (2 - 2 / 3 * epsilon + 2 / 9 * epsilon ** 2)


def plot_case(recommender, initialization, quantity, linestyle):
    means.loc[recommender, initialization][quantity].plot(legend=True, label=recommender, linestyle=linestyle)
    plt.fill_between(np.linspace(0, 0.2, 11),
                     means.loc[recommender, initialization][quantity] + std.loc[recommender, initialization][quantity],
                     means.loc[recommender, initialization][quantity] - std.loc[recommender, initialization][quantity],
                     alpha=0.2)


norms = df.norm.unique().tolist()
opt = np.ones(21) * -1.5
x_vals = np.linspace(0, 0.2, 21)
y_vals = welfare(x_vals)

cmap = plt.get_cmap('plasma')
sns.set_palette("magma")
# colors = [cmap(c) for c in np.linspace(0.1, 0.9, n_actions)]

plt.figure(figsize=(12, 9))

plot_case("none", "uniform", "T_mean_all", linestyle="-")
plot_case("naive", "uniform", "T_mean_all", linestyle="-")
plot_case("random", "uniform", "T_mean_all", linestyle="--")
plot_case("heuristic", "uniform", "T_mean_all", linestyle="-")
plot_case("optimized_action_maximize", "uniform", "T_mean_all", linestyle="--")
plot_case("optimized_estimate_maximize", "uniform", "T_mean_all", linestyle=":")
plot_case("aligned_heuristic", "uniform", "T_mean_all", linestyle=":")

plt.ylim((-1.4, -2.1))
plt.yticks(ticks=np.linspace(-1.4, -2.1, 5), labels=np.linspace(1.4, 2.1, 5))

plt.xticks(ticks=np.linspace(0, 0.2, 5))

plt.plot(x_vals, opt, label="social optimum", linestyle="--", color="gray")
plt.plot(x_vals, y_vals, label="irrational agents", linestyle=":", color="black")
plt.xlabel(r"irrationality / exploration rate ($\epsilon$)", **{"fontname": "Times New Roman", "fontsize": "x-large"})
plt.ylabel(r"average travel time ($\bar{T}$)", **{"fontname": "Times New Roman", "fontsize": "x-large"})
plt.legend(prop={"family": "Times New Roman", "size": "x-large"}, title="recommender type", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig("plots/recommenders_welfare_comparison.pdf")
plt.show()

plt.figure(figsize=(12, 9))

plot_case("none", "uniform", "alignment", linestyle="-")
plot_case("naive", "uniform", "alignment", linestyle="-")
plot_case("random", "uniform", "alignment", linestyle="--")
plot_case("heuristic", "uniform", "alignment", linestyle="-")
plot_case("optimized_action_maximize", "uniform", "alignment", linestyle="--")
plot_case("optimized_estimate_maximize", "uniform", "alignment", linestyle=":")
plot_case("aligned_heuristic", "uniform", "alignment", linestyle=":")

plt.ylim((0, 1))
plt.xticks(ticks=np.linspace(0, 0.2, 5))

plt.xlabel(r"irrationality / exploration rate ($\epsilon$)", **{"fontname": "Times New Roman", "fontsize": "x-large"})
plt.ylabel(r"recommendation to argmax alignment", **{"fontname": "Times New Roman", "fontsize": "x-large"})
plt.legend(prop={"family": "Times New Roman", "size": "x-large"}, title="recommender type", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig("plots/recommenders_alignment_comparison.pdf")
plt.show()
