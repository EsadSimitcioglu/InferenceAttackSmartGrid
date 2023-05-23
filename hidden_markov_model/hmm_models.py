import xxhash
from hmmlearn import hmm
import numpy as np


def create_model_start_transmission(k, train_type, user_value_list):
    model = hmm.MultinomialHMM(n_components=k, algorithm='viterbi')

    if train_type == 'plain':
        model.startprob_ = np.array([1 / k] * k)
        matrix_list = []
        for i in range(k):
            sub_list = []
            for j in range(k):
                sub_list.append(1 / k)
            matrix_list.append(sub_list)
        model.transmat_ = np.array(matrix_list)
    elif train_type == 'advance':
        dict_of_path = analyze_taken_path(user_value_list)

        start_grid_dict = {}
        for user_true_values in user_value_list:
            init_grid = user_true_values.item(0)
            start_grid_dict[init_grid] = start_grid_dict.get(init_grid, 0) + 1

        start_grid_dict = dict(sorted(start_grid_dict.items()))
        sum_grid_counts = sum(start_grid_dict.values())
        start_prob_list = list()
        for grid_count in start_grid_dict.values():
            start_prob_list.append(grid_count / sum_grid_counts)

        for _ in range(k - len(start_prob_list)):
            start_prob_list.append(0)

        model.startprob_ = np.array(start_prob_list)

        matrix_list = []
        for i in range(1, k + 1):

            sub_list = []
            sum_of_path = 0

            if i in dict_of_path:
                sum_of_path = sum(dict_of_path[i].values())
                if len(dict_of_path[i].values()) != k:
                    sum_of_path += k - len(dict_of_path[i].values())
            else:
                sum_of_path = k

            for j in range(1, k + 1):
                if i in dict_of_path and j in dict_of_path[i]:
                    sub_list.append(dict_of_path[i][j] / sum_of_path)
                else:
                    sub_list.append(1 / sum_of_path)
            matrix_list.append(sub_list)

        model.transmat_ = np.array(matrix_list)
    return model


def analyze_taken_path(users_grid_value_list):
    path_dict = {}
    for user_value in users_grid_value_list:
        for value_index, value in enumerate(user_value):
            if value_index == 0:
                continue

            prev_grid = user_value[value_index - 1]

            if prev_grid in path_dict:
                value_dict = path_dict.get(prev_grid)

                if value in value_dict:
                    value_dict[value] += 1
                else:
                    value_dict[value] = 1

            else:
                value_dict = {value: 1}
                path_dict[prev_grid] = value_dict

    return path_dict


def isValidPos(i, j, n, m):
    if i < 0 or j < 0 or i > n - 1 or j > m - 1:
        return 0
    return 1


# Function that returns all adjacent elements
def getAdjacent(arr, number):
    if number % 4 == 0:
        i = int(number / 4) - 1
        j = 3
    else:
        i = int(number / 4)
        j = (number % 4) - 1

    # Size of given 2d array
    n = len(arr)
    m = len(arr[0])

    # Initialising a vector array
    # where adjacent element will be stored
    v = []

    # Checking for all the possible adjacent positions
    if isValidPos(i - 1, j - 1, n, m):
        v.append(arr[i - 1][j - 1])
    if isValidPos(i - 1, j, n, m):
        v.append(arr[i - 1][j])
    if isValidPos(i - 1, j + 1, n, m):
        v.append(arr[i - 1][j + 1])
    if isValidPos(i, j - 1, n, m):
        v.append(arr[i][j - 1])
    if isValidPos(i, j + 1, n, m):
        v.append(arr[i][j + 1])
    if isValidPos(i + 1, j - 1, n, m):
        v.append(arr[i + 1][j - 1])
    if isValidPos(i + 1, j, n, m):
        v.append(arr[i + 1][j])
    if isValidPos(i + 1, j + 1, n, m):
        v.append(arr[i + 1][j + 1])

    # Returning the vector
    return [x + 1 for x in v]


def guess(model, user_perturbed_report):
    obs_sequence_list = []
    for perturbed_report in user_perturbed_report:
        obs_sequence_list.append(perturbed_report)
    obs_sequence = np.array([obs_sequence_list]).T

    _, state_sequence = model.decode(obs_sequence)

    for i in range(len(state_sequence)):
        state_sequence[i] = state_sequence[i] + 1

    return state_sequence

def hmm_model_GRR(epsilon, k, train_type, user_value_list=None):
    p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
    q = (1 - p) / (k - 1)

    model = create_model_start_transmission(k, train_type, user_value_list)

    matrix_list = []
    for i in range(k):
        row_list = []
        for j in range(k):
            if i == j:
                row_list.append(p)
            else:
                row_list.append(q)
        matrix_list.append(row_list)

    model.emissionprob_ = np.array(matrix_list)

    return model


def hmm_model_RAPPOR(epsilon, k, train_type, user_value_list=None):
    p = np.exp(epsilon / 2) / (np.exp(epsilon / 2) + 1)
    q = 1 / (np.exp(epsilon / 2) + 1)

    model = create_model_start_transmission(k, train_type, user_value_list)

    rappor_report_list = list()
    for x in range(2 ** k):
        rappor_report_list.append((bin(x)[2:].zfill(k)))

    user_value_list = list()
    for i in range(k):
        bit_vector = ''
        for j in range(k):
            if i == j:
                bit_vector += '1'
            else:
                bit_vector += '0'
        user_value_list.append(bit_vector)

    emission_prob_list = list()
    for row_index in range(len(user_value_list)):
        row = user_value_list[row_index]
        row_prob_list = list()
        for column_index in range(len(rappor_report_list)):
            column = rappor_report_list[column_index]
            prob = 1
            for char_index in range(len(row)):
                if row[char_index] == column[char_index]:
                    prob *= p
                else:
                    prob *= q
            row_prob_list.append(prob)
        emission_prob_list.append(row_prob_list)

    model.emissionprob_ = np.array(emission_prob_list)

    return model


def hmm_model_OUE(epsilon, k, train_type, user_value_list=None):
    p = 1 / 2
    q = 1 / (np.exp(epsilon) + 1)

    model = create_model_start_transmission(k, train_type, user_value_list)

    oue_report_list = list()
    for x in range(2 ** k):
        oue_report_list.append((bin(x)[2:].zfill(k)))

    user_value_list = list()
    for i in range(k):
        bit_vector = ''
        for j in range(k):
            if i == j:
                bit_vector += '1'
            else:
                bit_vector += '0'
        user_value_list.append(bit_vector)

    emission_prob_list = list()
    for row_index in range(len(user_value_list)):
        row = user_value_list[row_index]
        row_prob_list = list()
        for column_index in range(len(oue_report_list)):
            column = oue_report_list[column_index]
            prob = 1
            for char_index in range(len(row)):
                if row[char_index] == column[char_index]:
                    prob *= p
                else:
                    prob *= q
            row_prob_list.append(prob)
        emission_prob_list.append(row_prob_list)

    model.emissionprob_ = np.array(emission_prob_list)

    return model


def hmm_model_OLH(epsilon, k, seed_counter, train_type, user_value_list=None):
    p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
    g = int(round(np.exp(epsilon))) + 1
    q = (1 - p) / (g - 1)

    model = create_model_start_transmission(k, train_type, user_value_list)

    matrix_list = []
    for obs_state in range(k):
        row_list = []
        hash_value_of_obs_state = (xxhash.xxh32(str(obs_state), seed=seed_counter).intdigest() % g)
        for hidden_state in range(g):
            if hash_value_of_obs_state == hidden_state:
                row_list.append(p)
            else:
                row_list.append(q)
        matrix_list.append(row_list)

    model.emissionprob_ = np.array(matrix_list)

    return model
