import pandas as pd
import re
import os
from tqdm import tqdm


def all2df(dir_addr):
    dfs = []
    for file_name in tqdm(os.listdir(dir_addr)):
        if re.match('.+\.csv', file_name):
            path = dir_addr + str(file_name)
            file_df = pd.read_csv(path)
            if file_df is not None:
                dfs.append(file_df)

    return pd.concat(dfs)


def aggregate_dfs(input_df):
    print("aggregating...")

    df = input_df.drop(columns=['#repeat', 'q_initial'])

    df = df.groupby(['alpha', 'alpha_deviator', 'epsilon', 'epsilon_deviator', 'gamma', 'gamma_deviator']).mean()

    df.reset_index(inplace=True)

    print("length of merge", len(df))

    # add any other things to be calculated on the merged dataframe

    return df


if __name__ == "__main__":
    directory = "/cluster/work/coss/ccarissimo/braess_symmetric_meta_game/dataframes/"
    save_directory = "/cluster/work/coss/ccarissimo/braess_symmetric_meta_game/"

    final_df = all2df(directory)
    final_df = aggregate_dfs(final_df)
    final_df.to_csv(save_directory + "braess_symmetric_meta_game_results_v0.csv")
