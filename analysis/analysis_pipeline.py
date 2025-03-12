import join_dataframes
import compute_best_responses
import process_best_responses


def main(path_to_data):

    print("(1/3) joining dataframes")
    df = join_dataframes.main(path_to_data+"dataframes/", path_to_data)

    print("(2/3) computing best responses")
    best_responses = compute_best_responses.main(df)

    print("(3/3) processing multi_parameter best responses")
    final_df = process_best_responses.calculate_metric_changes_after_best_responses(df, best_responses, sub_df=None)

    save_path_for_multi = f"{path_to_data}multi_parameter_profits_metric_after_best_responses.csv"
    final_df.to_csv(save_path_for_multi)


if __name__ == "__main__":
    directory = "/cluster/work/coss/ccarissimo/braess_symmetric_meta_game_2/"
    main(directory)
