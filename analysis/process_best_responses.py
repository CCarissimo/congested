import pandas as pd
from tqdm.auto import tqdm
import pickle


def calculate_metric_changes_after_best_responses(main_df, best_responses, sub_df=None):

    if sub_df is None:
        print("reminder: no sub_df provided")
        sub_df = main_df

    results = {}
    for index, row in tqdm(sub_df.iterrows(), total=len(sub_df)):
        br_ego = best_responses[index]["deviator"]
        br_alter = best_responses[index]["alter"]
        br_par_ego = main_df.loc[br_ego][["alpha_deviator", "epsilon_deviator", "gamma_deviator"]].to_dict()
        br_par_alter = main_df.loc[br_alter][["alpha", "epsilon", "gamma"]].to_dict()

        # metrics of interest
        results[index] = {
            "br": br_ego,
            "isbest": br_ego == index,
            "deviator_gain": main_df["deviator_average"].loc[br_ego] - main_df["deviator_average"].loc[index],
            "br_alter": br_ego,
            "isbest_alter": br_alter == index,
            "alter_gain": main_df["non_deviator_average"].loc[br_alter] - main_df["non_deviator_average"].loc[index],
        }
        results[index].update(br_par_ego)
        results[index].update(br_par_alter)

    results_df = pd.DataFrame.from_dict(results, orient='index')
    final_df = pd.merge(sub_df, results_df, left_index=True, right_index=True)

    return final_df


def process_and_save_main_df(path_to_df, path_to_best_responses, save_path=None, path_to_sub_df=None):
    main_df = pd.read_csv(path_to_df)
    if path_to_sub_df is not None:
        sub_df = pd.read_csv(path_to_sub_df)
    else:
        sub_df = None

    with open(path_to_best_responses, "rb") as file:
        best_responses = pickle.load(file)

    final_df = calculate_metric_changes_after_best_responses(main_df, best_responses, sub_df)
    if save_path is not None:
        final_df.to_csv(save_path)

    return final_df


def main(dir_path, df):

    path_to_multi_br = f"{dir_path}multi_parameter_best_responses.pkl"
    # path_to_alpha_br = f"{dir_path}alpha_best_responses.pkl"
    # path_to_epsilon_br = f"{dir_path}epsilon_best_responses.pkl"
    # path_to_gamma_br = f"{dir_path}gamma_best_responses.pkl"
    # path_to_multi_welfare = f"{dir_path}multi_parameter_pareto_responses.pkl"

    save_path_for_multi = f"{dir_path}multi_parameter_profits_metric_after_best_responses.csv"
    # save_path_for_alpha = f"{dir_path}alpha_profits_metric_after_best_responses.csv"
    # save_path_for_epsilon = f"{dir_path}epsilon_profits_metric_after_best_responses.csv"
    # save_path_for_gamma = f"{dir_path}gamma_profits_metric_after_best_responses.csv"
    # save_path_for_multi_welfare = f"{dir_path}multi_parameter_profits_metric_after_welfare_responses.csv"

    print("(1/1) processing multi_parameter best responses:")
    final_df = process_and_save_main_df(path_to_df, path_to_multi_br, save_path_for_multi)


    # print("(1/4) processing multi_parameter welfare responses:")
    # process_and_save_main_df(path_to_df, path_to_multi_welfare, save_path_for_multi_welfare)

    # print("(2/4) processing alpha best responses:")
    # process_and_save_main_df(path_to_df, path_to_alpha_br, save_path_for_alpha)
    #
    # print("(3/4) processing epsilon best responses:")
    # process_and_save_main_df(path_to_df, path_to_epsilon_br, save_path_for_epsilon)
    #
    # print("(4/4) processing gamma best responses:")
    # process_and_save_main_df(path_to_df, path_to_gamma_br, save_path_for_gamma)


if __name__ == "__main__":
    dir_path = "/cluster/work/coss/ccarissimo/braess_symmetric_meta_game/"
    path_to_df = f"{dir_path}braess_symmetric_meta_game_results_v0.csv"
    main(dir_path, path_to_df)
