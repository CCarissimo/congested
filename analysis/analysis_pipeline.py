import join_dataframes
import compute_best_responses
import process_best_responses
import pickle


def main(path_to_data, name="results_v0"):

    print("(1/3) joining dataframes")
    df = join_dataframes.all2df(path_to_data+"dataframes/")
    df = join_dataframes.aggregate_dfs(df)
    df.to_csv(path_to_data + f"{name}.csv")

    print("(2/3) computing best responses")
    best_responses = compute_best_responses.compute_deviator_best_response(df)
    with open(path_to_data + f"{name}_mp_br_indices.pkl", "wb") as file:
        pickle.dump(best_responses, file)

    print("(3/3) processing multi_parameter best responses")
    final_df = process_best_responses.calculate_metric_changes_after_best_responses(df, best_responses, sub_df=None)
    final_df.to_csv(f"{path_to_data}{name}_mp_br_metrics.csv")


if __name__ == "__main__":
    directory = "/cluster/work/coss/ccarissimo/braess_symmetric_meta_game_2/"
    name = "braess_symmetric_results_v0"
    main(directory, name)
