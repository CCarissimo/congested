import join_dataframes
import compute_best_responses
import process_best_responses
import pickle
import argparse


def main(path_to_data, name="results_v0", deviators=None):

    print("(1/3) joining dataframes")
    df = join_dataframes.all2df(path_to_data+"dataframes/")
    df = join_dataframes.aggregate_dfs(df)
    df.to_csv(path_to_data + f"{name}.csv")

    if deviators is None:
        for n_deviators in [2, 3, 6, 12, 25, 50]:

            sub_df = df[df["number_of_deviators"]==n_deviators]

            print(f"N:{n_deviators} computing best responses")
            best_responses = compute_best_responses.compute_deviator_best_response(sub_df)
            with open(path_to_data + f"{name}_deviators{n_deviators}_mp_br_indices.pkl", "wb") as file:
                pickle.dump(best_responses, file)

            print(f"N:{n_deviators} processing multi_parameter best responses")
            final_df = process_best_responses.calculate_metric_changes_after_best_responses(sub_df, best_responses, sub_df=None)
            final_df.to_csv(f"{path_to_data}{name}_deviators{n_deviators}_mp_br_metrics.csv")
    else:
        sub_df = df[df["number_of_deviators"] == deviators]

        print(f"N:{deviators} computing best responses")
        best_responses = compute_best_responses.compute_deviator_best_response(sub_df)
        with open(path_to_data + f"{name}_deviators{deviators}_mp_br_indices.pkl", "wb") as file:
            pickle.dump(best_responses, file)

        print(f"N:{deviators} processing multi_parameter best responses")
        final_df = process_best_responses.calculate_metric_changes_after_best_responses(sub_df, best_responses,
                                                                                        sub_df=None)
        final_df.to_csv(f"{path_to_data}{name}_deviators{deviators}_mp_br_metrics.csv")


if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description="input data locations")

    # Add arguments
    parser.add_argument('directory', type=str, help="main directory which contains the dataframes directory")
    parser.add_argument('-n', '--name', type=str, help="name for file save", default="results_v0")
    parser.add_argument('-d', '--deviators', type=str, help="what is the size of the deviator population", default=None)

    # Parse the arguments
    args = parser.parse_args()

    # Run the analysis
    main(args.directory, args.name, args.deviators)
