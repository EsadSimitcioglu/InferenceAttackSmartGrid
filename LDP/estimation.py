import numpy as np

from hidden_markov_model.hmm_models import hmm_model_GRR, hmm_model_RAPPOR, hmm_model_OUE, hmm_model_OLH
from LDP.protocols import GRR_Client, SIMPLE_RAPPOR_Client, OUE_Client, OLH_Client2


def binary_to_decimal(binary_number):
    decimal_number = 0
    index_counter = len(binary_number) - 1

    for number in binary_number:
        decimal_number += (int(number) * (2 ** index_counter))
        index_counter -= 1
    return decimal_number


def perturb(protocol_type, epsilon, k, user_true_value_list):
    perturbed_reports = list()
    if protocol_type == 'GRR':
        for user_true_values in user_true_value_list:
            perturbed_reports.append([GRR_Client(user_true_value, k, epsilon) for user_true_value in user_true_values])
    elif protocol_type == 'RAPPOR':
        for user_true_values in user_true_value_list:
            report_binary_list = list()
            rappor_reports = [SIMPLE_RAPPOR_Client(user_true_value, k, epsilon) for user_true_value in user_true_values]
            for rappor_report in rappor_reports:
                report_string = ""
                for index in rappor_report:
                    report_string += str(int(index))
                report_binary_list.append(binary_to_decimal(report_string))
            perturbed_reports.append(report_binary_list)
    elif protocol_type == 'OUE':
        for user_true_values in user_true_value_list:
            report_binary_list = list()
            oue_reports = [OUE_Client(user_true_value, k, epsilon) for user_true_value in user_true_values]
            for report in oue_reports:
                report_string = ""
                for index in report:
                    report_string += str(int(index))
                report_binary_list.append(binary_to_decimal(report_string))
            perturbed_reports.append(report_binary_list)
    elif protocol_type == 'OLH':
        seed_value = 1
        for user_true_values in user_true_value_list:
            perturbed_reports.append(OLH_Client2(user_true_values, k, epsilon, seed_value))
            seed_value += 1

    return perturbed_reports


def guess(epsilon, k, user_true_value_list, protocol_type, test_type, model=None, user_guess_value_list=None):
    guess_prob_list = list()
    guess_list = list()

    perturbed_reports = perturb(protocol_type, epsilon, k, user_true_value_list)

    for index, user_perturbed_report in enumerate(perturbed_reports):

        if protocol_type == "OLH":
            if user_guess_value_list is None:
                model = hmm_model_OLH(epsilon, k, index + 1, 'plain')
            else:
                model = hmm_model_OLH(epsilon, k, index + 1, 'advance', user_guess_value_list)

        obs_sequence_list = []
        for perturbed_report in user_perturbed_report:
            if protocol_type == "GRR":
                obs_sequence_list.append(perturbed_report - 1)
            else:
                obs_sequence_list.append(perturbed_report)
        obs_sequence = np.array([obs_sequence_list]).T

        _, state_sequence = model.decode(obs_sequence)
        prob_sum = 0
        index_counter = 0

        if test_type == 'guess':
            for o, s in zip(obs_sequence.T[0], state_sequence):
                true_value = user_true_value_list[index][index_counter]
                guess_value = s if protocol_type == "OLH" else (s + 1)

                if guess_value == 0 and true_value == 0:
                    prob_sum += 1
                index_counter += 1
            guess_prob_list.append(prob_sum / index_counter)
        elif test_type == 'advance':
            user_guess_list = list()
            for o, s in zip(obs_sequence.T[0], state_sequence):
                guess_value = s if protocol_type == "OLH" else (s + 1)
                user_guess_list.append(guess_value)
            guess_list.append(np.array(user_guess_list))

    if test_type == 'guess':
        return np.average(guess_prob_list)
    elif test_type == 'advance':
        return guess_list


def GRR_estimated_guess_plain(user_values_list, k, epsilon):
    model = hmm_model_GRR(epsilon, k, 'plain')
    return guess(epsilon, k, user_values_list, "GRR", 'guess', model)


def GRR_estimated_guess_advance(user_values_list, k, epsilon):
    model = hmm_model_GRR(epsilon, k, 'plain')
    guess_list = guess(epsilon, k, user_values_list, "GRR", "advance", model)
    model_advance = hmm_model_GRR(epsilon, k, "advance", guess_list)
    return guess(epsilon, k, user_values_list, "GRR", 'guess', model_advance)


def RAPPOR_estimated_guess(user_values_list, k, epsilon):
    model = hmm_model_RAPPOR(epsilon, k, 'plain')
    return guess(epsilon, k, user_values_list, "RAPPOR", 'guess', model)


def RAPPOR_estimated_guess_advance(user_values_list, k, epsilon):
    model = hmm_model_RAPPOR(epsilon, k, 'plain')
    guess_list = guess(epsilon, k, user_values_list, "RAPPOR", "advance", model)
    model_advance = hmm_model_RAPPOR(epsilon, k, "advance", guess_list)
    return guess(epsilon, k, user_values_list, "RAPPOR", 'guess', model_advance)


def OUE_estimated_guess(user_values_list, k, epsilon):
    model = hmm_model_OUE(epsilon, k, 'plain')
    return guess(epsilon, k, user_values_list, "OUE", 'guess', model)


def OUE_estimated_guess_advance(user_values_list, k, epsilon):
    model = hmm_model_OUE(epsilon, k, 'plain')
    guess_list = guess(epsilon, k, user_values_list, "OUE", "advance", model)
    model_advance = hmm_model_OUE(epsilon, k, "advance", guess_list)
    return guess(epsilon, k, user_values_list, "OUE", 'guess', model_advance)


def OLH_estimated_guess(user_values_list, k, epsilon):
    return guess(epsilon, k, user_values_list, "OLH", 'guess')

def OLH_estimated_guess_advance(user_values_list, k, epsilon):
    hmm_values_list = guess(epsilon, k, user_values_list, "OLH", 'advance')
    return guess(epsilon, k, user_values_list, "OLH", "guess", user_guess_value_list=hmm_values_list)
