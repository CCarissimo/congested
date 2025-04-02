import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import seaborn as sns
import matplotlib.cm as cm
# from matplotlib.animation import FuncAnimation
import re
import os


def get_symmetric_strategies(df):
    return (df[(df['alpha'] == df['alpha_deviator_x']) &\
           (df['gamma'] == df['gamma_deviator_x']) &\
           (df['epsilon'] == df['epsilon_deviator_x'])])


def extract_deviators_n(filename):
    match = re.search(r'_deviators(\d+)_', filename)
    return int(match.group(1)) if match else None


def main(directory, filename):
    df = pd.read_csv(directory+filename)
    n_deviators = extract_deviators_n(filename)

    os.mkdir(directory+"plots/"+str(n_deviators))

    symbr = get_symmetric_strategies(df)

    df["e_diff"] = df["epsilon_deviator_y"] - df["epsilon"]
    df["a_diff"] = df["alpha_deviator_y"] - df["alpha"]
    df["g_diff"] = df["gamma_deviator_y"] - df["gamma"]

    symbr["e_diff"] = symbr["epsilon_deviator_y"] - symbr["epsilon"]
    symbr["a_diff"] = symbr["alpha_deviator_y"] - symbr["alpha"]
    symbr["g_diff"] = symbr["gamma_deviator_y"] - symbr["gamma"]

    br_points = symbr.groupby(['alpha_deviator_y', 'epsilon_deviator_y', 'gamma_deviator_y']).size().reset_index(name='count')
    br_points_full = df.groupby(['alpha_deviator_y', 'epsilon_deviator_y', 'gamma_deviator_y']).size().reset_index(name='count')

    # Create a new figure for the 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Extract the vectors from the DataFrame
    x = symbr['epsilon']  # x-coordinates
    y = symbr['alpha']  # y-coordinates
    z = symbr['gamma']  # z-coordinates

    u = symbr['e_diff']  # Vector components along x-axis
    v = symbr['a_diff']  # Vector components along y-axis
    w = symbr['g_diff']  # Vector components along z-axis

    # Color map based on the 'color_var' column
    colors = symbr['deviator_gain']  # You can change this to any column you'd like

    # Normalize the color variable to the range [0, 1]
    norm = plt.Normalize(colors.min(), colors.max())
    cmap = cm.plasma_r  # You can choose any colormap here (like 'plasma', 'inferno', etc.)
    mapped_colors = cmap(norm(colors))

    # Plot the vectors in 3D
    ax.quiver(x, y, z, u, v, w, length=1, normalize=False, alpha=0.5, arrow_length_ratio=0.2, colors=mapped_colors)
    ax.view_init(elev=28, azim=45)  # Adjust the view for perspective

    # Set labels for the axes
    ax.set_xlabel('epsilon')
    ax.set_ylabel('alpha')
    ax.set_zlabel('gamma')

    ax.set_xlim([0, 0.2])  # X-axis limits
    ax.set_ylim([0, 1])  # Y-axis limits
    ax.set_zlim([0, 1])  # Z-axis limits

    # ax.set_xscale("log")

    # ax.set_frame_on(False)  # Remove 3D box frame
    # ax.set_xticks([])  # Remove x-axis ticks
    # ax.set_yticks([])  # Remove y-axis ticks
    # ax.set_zticks([])  # Remove z-axis ticks
    # ax.set_xlabel('')  # Remove x-axis label
    # ax.set_ylabel('')  # Remove y-axis label
    # ax.set_zlabel('')  # Remove z-axis label

    # Extract positions and size from DataFrame
    x = br_points_full['epsilon_deviator_y']
    y = br_points_full['alpha_deviator_y']
    z = br_points_full['gamma_deviator_y']
    sizes = br_points_full['count'] / 200  # Size of the spheres
    # Plot spheres as scatter points in 3D
    scatter = ax.scatter(x, y, z, s=sizes, alpha=0.6, edgecolor='w')

    # Add a color bar to indicate what the colors represent
    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(colors)
    # fig.colorbar(mappable, ax=ax, label='profit gain')

    # Set title
    # ax.set_title('Best Response Plot from SYMMETRIC Equilibria')

    # Save
    plt.savefig(f"{directory}/plots/{n_deviators}/best_response_full-length-arrows_SYMMETRIC_EQUILIBRIA.png", dpi=180)

    intersection = pd.merge(symbr, br_points_full,
                            left_on=["alpha_deviator_x", "epsilon_deviator_x", "gamma_deviator_x"],
                            right_on=["alpha_deviator_y", "epsilon_deviator_y", "gamma_deviator_y"], how="inner")

    # Create a new figure for the 3D plot
    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # Extract positions and size from DataFrame
    x = br_points_full['epsilon_deviator_y']
    y = br_points_full['alpha_deviator_y']
    z = br_points_full['gamma_deviator_y']
    sizes = np.sqrt(br_points_full['count'] * 10e1)  # Size of the spheres

    # Plot spheres as scatter points in 3D
    scatter = ax.scatter(x, y, z, s=sizes, alpha=0.6, edgecolor='w')

    # Extract the vectors from the DataFrame
    x = intersection['epsilon']  # x-coordinates
    y = intersection['alpha']  # y-coordinates
    z = intersection['gamma']  # z-coordinates

    u = intersection['e_diff']  # Vector components along x-axis
    v = intersection['a_diff']  # Vector components along y-axis
    w = intersection['g_diff']  # Vector components along z-axis

    # Color map based on the 'color_var' column
    metric_tag = "deviator_gain"
    colors = intersection[metric_tag]  # You can change this to any column you'd like

    # Normalize the color variable to the range [0, 1]
    norm = plt.Normalize(colors.min(), colors.max())
    cmap = cm.plasma_r  # You can choose any colormap here (like 'plasma', 'inferno', etc.)
    mapped_colors = cmap(norm(colors))

    # Plot the vectors in 3D
    ax.quiver(x, y, z, u, v, w,
              length=0.85,
              linewidth=2.5,
              normalize=False,
              alpha=0.9,
              arrow_length_ratio=0.15,
              colors=mapped_colors)
    ax.view_init(elev=28, azim=45)  # Adjust the view for perspective

    # ax.set_axis_off()

    # Add color bar to show the size mapping
    # fig.colorbar(scatter, ax=ax, label='symmetric profiles for which point is best response')
    # Add a color bar to indicate what the colors represent
    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(colors)
    # fig.colorbar(mappable, ax=ax, label=metric_tag)

    # Set the title
    # ax.set_title('The best reponses from the best reponse parameters')

    ax.set_xlim([0, 0.5])  # X-axis limits
    ax.set_ylim([0, 1])  # Y-axis limits
    ax.set_zlim([0, 1])  # Z-axis limits

    ax.set_frame_on(False)  # Remove 3D box frame
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_zticks([])  # Remove z-axis ticks
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_ylabel('')  # Remove y-axis label
    ax.set_zlabel('')  # Remove z-axis label

    # # Add labels
    # ax.set_xlabel('epsilon')
    # ax.set_ylabel('alpha')
    # ax.set_zlabel('gamma')

    # plt.tight_layout()
    # save
    plt.savefig(f"{directory}/plots/{n_deviators}/3D_plot_middle.png", dpi=300)

    def create_labels(row):
        line1 = fr"$\alpha$: {row['alpha_deviator_x']}"
        line2 = fr"$\epsilon$: {row['epsilon_deviator_x']}"
        line3 = fr"$\gamma$: {row['gamma_deviator_x']}"
        return line1 + " \n" + line2 + " \n" + line3

    intersection["node_label"] = intersection.apply(create_labels, axis=1)

    # Function to find the index of the matching row for a given row B
    def find_matching_row_index(row):
        # Create a boolean mask to find matches in the DataFrame based on the current row B values
        mask = (intersection['alpha'] == row['alpha_deviator_y_x']) & \
               (intersection['epsilon'] == row['epsilon_deviator_y_x']) & \
               (intersection['gamma'] == row['gamma_deviator_y_x'])

        # Get the matching indices
        matching_indices = intersection.index[mask].tolist()

        # Return the first matching index if found, otherwise return None
        return matching_indices[0] if matching_indices else None

    # Apply the function to each row in the DataFrame and create a new column 'MatchingIndex'
    intersection['MatchingIndex'] = intersection.apply(find_matching_row_index, axis=1)

    # %matplotlib notebook

    fig = plt.figure(figsize=(10, 10))

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes with labels
    for idx, row in intersection.iterrows():
        G.add_node(idx, label=row['node_label'], size=row['count'])

    # Add directed edges based on the MatchingIndex condition
    for idx, row in intersection.iterrows():
        if pd.notna(row['MatchingIndex']):  # Only consider non-null MatchingIndices
            G.add_edge(idx, int(row['MatchingIndex']))

    # Set up the position for the nodes in the plot
    pos = nx.kamada_kawai_layout(G)
    # pos[0] = np.array([0.33484752, 0.84425323])
    # pos[5] = np.array([0.5965479, 0.63980577])
    # pos[10] = np.array([0.46249581, 0.28107043])

    # Extract sizes from the DataFrame for each node
    node_sizes = [G.nodes[idx]['size'] / 100 for idx in G.nodes()]

    # Draw the nodes and edges
    nx.draw(G, pos, with_labels=False, arrows=True, node_size=node_sizes, node_color='skyblue')

    # Label the nodes with custom labels from the DataFrame
    labels = nx.get_node_attributes(G, 'label')

    # Ensure all labels are strings and handle any None values
    labels = {idx: row["node_label"] for idx, row in intersection.iterrows()}

    # Draw the node labels
    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_color='black')

    plt.title("Directed Network from DataFrame")

    # save
    plt.savefig(f"{directory}/plots/{n_deviators}/cycle_between_best_responses_SYMMETRIC.png", dpi=180)

    # Create a new figure for the 3D plot
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Extract positions and size from DataFrame
    x = symbr['epsilon']
    y = symbr['alpha']
    z = symbr['gamma']
    # sizes = 10*(symbr['profit 1 avg'] + symbr['profit 1 avg'])  # Size of the spheres
    metric_tag = "welfare"
    sizes = symbr[f"{metric_tag}"]  # Size of the spheres

    # Plot spheres as scatter points in 3D
    scatter = ax.scatter(x, y, z, c=sizes, s=100, cmap='viridis', alpha=0.3, edgecolor='w')

    ax.view_init(elev=28, azim=45)  # Adjust the view for perspective

    # Add labels
    ax.set_xlabel('epsilon')
    ax.set_ylabel('alpha')
    ax.set_zlabel('gamma')

    # Add color bar to show the size mapping
    fig.colorbar(scatter, ax=ax, label='welfare')

    # Set the title
    ax.set_title(f'Collusion as {metric_tag}')

    ax.set_xlim([0, 0.5])  # X-axis limits
    ax.set_ylim([0, 1])  # Y-axis limits
    ax.set_zlim([0, 1])  # Z-axis limits

    # save
    plt.savefig(f'{directory}/plots/{n_deviators}/symmetric_player_{metric_tag}.png')


if __name__ == "__main__":
    import argparse

    # Initialize the parser
    parser = argparse.ArgumentParser(description="input data locations")

    # Add arguments
    parser.add_argument('directory', type=str, help="main directory which contains the dataframes directory")
    parser.add_argument('-n', '--name', type=str, help="name for file save", default="results_v0")
    # parser.add_argument('-d', '--deviators', type=int, help="what is the size of the deviator population", default=None)
    # parser.add_argument('-j', '--joined', action='store_true', help="have the dataframes already been joined")

    # Parse the arguments
    args = parser.parse_args()

    main(args.directory, args.name)