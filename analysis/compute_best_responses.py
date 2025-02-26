import pandas as pd
from tqdm import tqdm
import pickle


def get_symmetric_strategies(df):
    return (df[(df['alpha'] == df['alpha_deviator']) &
               (df['gamma'] == df['gamma_deviator']) &
               (df['epsilon'] == df['epsilon_deviator'])])


# def get_single_parameter_deviations(df, index, parameter, player=2):
#     """
#     the index determines the starting strategy
#     this function returns a df with all of the strategies which have all but one parameter difference
#     """
#
#     alpha1 = df["alpha"].loc[index]
#     alpha2 = df["alpha_deviator"].loc[index]
#     gamma1 = df["gamma"].loc[index]
#     gamma2 = df["gamma_deviator"].loc[index]
#     epsilon1 = df["epsilon"].loc[index]
#     epsilon2 = df["epsilon_deviator"].loc[index]
#
#     if player == 1:
#         if parameter == "alpha":
#             # get deviations from alpha
#             dev_alpha = (df[(df['alpha'] != alpha1) &
#                             (df['alpha_deviator'] == alpha2) &
#                             (df['gamma'] == gamma1) &
#                             (df['gamma_deviator'] == gamma2) &
#                             (df['epsilon'] == epsilon1) &
#                             (df['epsilon_deviator'] == epsilon2)])
#             return dev_alpha
#
#         elif parameter == "gamma":
#             # get deviations from gamma
#             dev_gamma = (df[(df['alpha'] == alpha1) &
#                             (df['alpha_deviator'] == alpha2) &
#                             (df['gamma'] != gamma1) &
#                             (df['gamma_deviator'] == gamma2) &
#                             (df['epsilon'] == epsilon1) &
#                             (df['epsilon_deviator'] == epsilon2)])
#             return dev_gamma
#
#         elif parameter == "epsilon":
#             # get deviations from esilon
#             dev_epsilon = (df[(df['alpha'] == alpha1) &
#                               (df['alpha_deviator'] == alpha2) &
#                               (df['gamma'] == gamma1) &
#                               (df['gamma_deviator'] == gamma2) &
#                               (df['epsilon'] != epsilon1) &
#                               (df['epsilon_deviator'] == epsilon2)])
#             return dev_epsilon
#
#     elif player == 2:
#         if parameter == "alpha":
#             # get deviations from alpha
#             dev_alpha = (df[(df['alpha'] == alpha1) &
#                             (df['alpha_deviator'] != alpha2) &
#                             (df['gamma'] == gamma1) &
#                             (df['gamma_deviator'] == gamma2) &
#                             (df['epsilon'] == epsilon1) &
#                             (df['epsilon_deviator'] == epsilon2)])
#             return dev_alpha
#
#         elif parameter == "gamma":
#             # get deviations from gamma
#             dev_gamma = (df[(df['alpha'] == alpha1) &
#                             (df['alpha_deviator'] == alpha2) &
#                             (df['gamma'] == gamma1) &
#                             (df['gamma_deviator'] != gamma2) &
#                             (df['epsilon'] == epsilon1) &
#                             (df['epsilon_deviator'] == epsilon2)])
#             return dev_gamma
#
#         elif parameter == "epsilon":
#             # get deviations from esilon
#             dev_epsilon = (df[(df['alpha'] == alpha1) &
#                               (df['alpha_deviator'] == alpha2) &
#                               (df['gamma'] == gamma1) &
#                               (df['gamma_deviator'] == gamma2) &
#                               (df['epsilon'] == epsilon1) &
#                               (df['epsilon_deviator'] != epsilon2)])
#             return dev_epsilon
#
#
# def get_double_parameter_deviations(df, index, parameters, player=2):
#     """
#     the index determines the starting strategy
#     this function returns a df with all of the strategies which have all but one parameter difference
#     """
#
#     alpha1 = df["alpha"].loc[index]
#     alpha2 = df["alpha_deviator"].loc[index]
#     gamma1 = df["gamma"].loc[index]
#     gamma2 = df["gamma_deviator"].loc[index]
#     epsilon1 = df["epsilon"].loc[index]
#     epsilon2 = df["epsilon_deviator"].loc[index]
#
#     if player == 1:
#         if ("alpha" in parameters) and ("epsilon" in parameters):
#             # get deviations from alpha, epsilon
#             deviations = (df[(df['alpha'] != alpha1) &
#                              (df['alpha_deviator'] == alpha2) &
#                              (df['gamma'] == gamma1) &
#                              (df['gamma_deviator'] == gamma2) &
#                              (df['epsilon'] != epsilon1) &
#                              (df['epsilon_deviator'] == epsilon2)])
#             return deviations
#
#         elif ("alpha" in parameters) and ("gamma" in parameters):
#             # get deviations from alpha, gamma
#             deviations = (df[(df['alpha'] != alpha1) &
#                              (df['alpha_deviator'] == alpha2) &
#                              (df['gamma'] != gamma1) &
#                              (df['gamma_deviator'] == gamma2) &
#                              (df['epsilon'] == epsilon1) &
#                              (df['epsilon_deviator'] == epsilon2)])
#             return deviations
#
#         elif ("epsilon" in parameters) and ("gamma" in parameters):
#             # get deviations from esilon
#             deviations = (df[(df['alpha'] == alpha1) &
#                              (df['alpha_deviator'] == alpha2) &
#                              (df['gamma'] != gamma1) &
#                              (df['gamma_deviator'] == gamma2) &
#                              (df['epsilon'] != epsilon1) &
#                              (df['epsilon_deviator'] == epsilon2)])
#             return deviations
#
#     elif player == 2:
#         if ("alpha" in parameters) and ("epsilon" in parameters):
#             # get deviations from alpha, epsilon
#             deviations = (df[(df['alpha'] == alpha1) &
#                              (df['alpha_deviator'] != alpha2) &
#                              (df['gamma'] == gamma1) &
#                              (df['gamma_deviator'] == gamma2) &
#                              (df['epsilon'] == epsilon1) &
#                              (df['epsilon_deviator'] != epsilon2)])
#             return deviations
#
#         elif ("alpha" in parameters) and ("gamma" in parameters):
#             # get deviations from alpha, gamma
#             deviations = (df[(df['alpha'] == alpha1) &
#                              (df['alpha_deviator'] != alpha2) &
#                              (df['gamma'] == gamma1) &
#                              (df['gamma_deviator'] != gamma2) &
#                              (df['epsilon'] == epsilon1) &
#                              (df['epsilon_deviator'] == epsilon2)])
#             return deviations
#
#         elif ("epsilon" in parameters) and ("gamma" in parameters):
#             # get deviations from esilon
#             deviations = (df[(df['alpha'] == alpha1) &
#                              (df['alpha_deviator'] == alpha2) &
#                              (df['gamma'] == gamma1) &
#                              (df['gamma_deviator'] != gamma2) &
#                              (df['epsilon'] == epsilon1) &
#                              (df['epsilon_deviator'] != epsilon2)])
#             return deviations


def get_multi_parameter_deviations(df, index, player=2):
    """
    the index determines the starting strategy
    this function returns a df with all the strategies which deviate from the index strategy
    """

    alpha1 = df["alpha"].loc[index]
    alpha2 = df["alpha_deviator"].loc[index]
    gamma1 = df["gamma"].loc[index]
    gamma2 = df["gamma_deviator"].loc[index]
    epsilon1 = df["epsilon"].loc[index]
    epsilon2 = df["epsilon_deviator"].loc[index]

    if player == 2:
        deviations = (df[((df['alpha'] == alpha1) &
                          (df['alpha_deviator'] != alpha2)) |
                         ((df['gamma'] == gamma1) &
                          (df['gamma_deviator'] != gamma2)) |
                         ((df['epsilon'] == epsilon1) &
                          (df['epsilon_deviator'] != epsilon2))])
    else:
        deviations = (df[((df['alpha'] != alpha1) &
                          (df['alpha_deviator'] == alpha2)) |
                         ((df['gamma'] != gamma1) &
                          (df['gamma_deviator'] == gamma2)) |
                         ((df['epsilon'] != epsilon1) &
                          (df['epsilon_deviator'] == epsilon2))])

    return deviations


# def check_best_response(df, deviations, index, player):
#     profit = df[f"profit {player} avg"].loc[index]
#
#     best_response = deviations[deviations[f"profit {player} avg"] > profit]
#
#     if len(best_response) == 0:
#         return index
#     else:
#         best = best_response[best_response[f"profit {player} avg"] == best_response[f"profit {player} avg"].max()]
#         response_index = best.index
#         return response_index[-1]


def check_deviator_response(df, deviations, index):
    profit = df[f"deviator_average"].loc[index]

    best_response = deviations[deviations[f"deviator_average"] > profit]

    if len(best_response) == 0:
        return index
    else:
        best = best_response[best_response[f"deviator_average"] == best_response[f"deviator_average"].max()]
        response_index = best.index
        return response_index[-1]


def check_welfare_optimal_response(df, deviations, index):
    welfare = df[f"welfare"].loc[index]

    welfare_optimal_responses = deviations[deviations[f"welfare"] > welfare]

    if len(welfare_optimal_responses) == 0:
        return index
    else:
        best = welfare_optimal_responses[
            welfare_optimal_responses[f"welfare"] == welfare_optimal_responses[f"welfare"].max()]
        response_index = best.index
        return response_index[-1]


# def process_player(player, dataframe, index, parameter, welfare_optimal=False):
#     responses = {}
#
#     if type(parameter) == list() and len(parameter) == 2:  # a list with two parameters
#         deviations = get_double_parameter_deviations(dataframe, index, parameter, player)
#
#     elif type(parameter) == list() and len(parameter) == 1:  # parameter is alpha, epsilon or gamma
#         deviations = get_single_parameter_deviations(dataframe, index, parameter, player)
#
#     else:
#         deviations = get_multi_parameter_deviations(dataframe, index, player)
#
#     if welfare_optimal:
#         response_index = check_welfare_optimal_response(dataframe, deviations, index)
#     else:  # unilateral best response
#         response_index = check_best_response(dataframe, deviations, index, player)
#
#     responses[index] = response_index
#
#     for ind, _ in deviations.iterrows():
#         responses[ind] = response_index
#
#     return responses


def process_deviator(dataframe, index, welfare_optimal=False):
    responses = {}

    deviations = get_multi_parameter_deviations(dataframe, index)

    if welfare_optimal:
        response_index = check_welfare_optimal_response(dataframe, deviations, index)
    else:  # unilateral best response
        response_index = check_deviator_response(dataframe, deviations, index)

    responses[index] = response_index

    for ind, _ in deviations.iterrows():
        responses[ind] = response_index

    return responses


# def compute_best_responses(df, parameter, welfare_optimal=False):
#     if parameter is None:
#         print("parameter is None")
#         return None
#
#     best_responses_1 = {}
#     best_responses_2 = {}
#
#     for index, strat in tqdm(df.iterrows(), total=len(df)):
#         if index not in best_responses_1.keys():
#             responses = process_player(1, df, index, parameter, welfare_optimal=welfare_optimal)
#             best_responses_1.update(responses)
#
#         if index not in best_responses_2.keys():
#             responses = process_player(2, df, index, parameter, welfare_optimal=welfare_optimal)
#             best_responses_2.update(responses)
#
#     best_responses = {
#         ind: {
#             'p1': best_responses_1[ind],
#             'p2': best_responses_2[ind]
#         } for ind in best_responses_1.keys()}
#
#     return best_responses


def compute_deviator_best_response(df, welfare_optimal=False):

    best_responses = {}

    for index, strat in tqdm(df.iterrows(), total=len(df)):
        if index not in best_responses.keys():
            responses = process_deviator(df, index, welfare_optimal=welfare_optimal)
            best_responses.update(responses)

    best_responses = {
        ind: {
            'deviator': best_responses[ind],
        } for ind in best_responses.keys()}

    return best_responses


def main(df):
    print("(1/1) computing multi_parameter best responses:")
    best_responses = compute_deviator_best_response(df)
    with open(dir_path + "multi_parameter_best_responses.pkl", "wb") as file:
        pickle.dump(best_responses, file)

    # print("(2/2) computing multi_parameter welfare responses:")
    # best_responses = compute_best_responses(df, parameter="multi_parameter", welfare_optimal=True)
    # with open(dir_path + "multi_parameter_pareto_responses.pkl", "wb") as file:
    #     pickle.dump(best_responses, file)

    # print("(2/4) computing alpha best responses:")
    # best_responses = compute_best_responses(df, parameter=["alpha"])
    # with open(dir_path + "alpha_best_responses.pkl", "wb") as file:
    #     pickle.dump(best_responses, file)
    #
    # print("(3/4) computing epsilon best responses:")
    # best_responses = compute_best_responses(df, parameter=["epsilon"])
    # with open(dir_path + "epsilon_best_responses.pkl", "wb") as file:
    #     pickle.dump(best_responses, file)
    #
    # print("(4/4) computing gamma best responses:")
    # best_responses = compute_best_responses(df, parameter=["gamma"])
    # with open(dir_path + "gamma_best_responses.pkl", "wb") as file:
    #     pickle.dump(best_responses, file)

    return None


if __name__ == "__main__":
    dir_path = "/cluster/work/coss/ccarissimo/braess_symmetric_meta_game/"
    filename = "braess_symmetric_meta_game_results_v0.csv"
    df = pd.read_csv(dir_path + filename)

    main(df)
