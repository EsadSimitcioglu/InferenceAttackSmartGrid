import numpy as np

from LDP.hidden_markov_model.hmm_models import hmm_model_GRR, hmm_model_RAPPOR, hmm_model_OUE, hmm_model_OLH, guess
from LDP.LDP_Protocols.protocols import GRR_Client, SIMPLE_RAPPOR_Client, OUE_Client, OLH_Client2
from LDP.metric.metric import ratio_of_guess



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


def experiment(epsilon, k, user_values_list, protocol_type, model=None, user_guess_value_list=None,
               test_count=None):
    guess_metric = list()

    perturbed_reports = perturb(protocol_type, epsilon, k, user_values_list)

    for index, user_perturbed_report in enumerate(perturbed_reports):

        if protocol_type == "OLH":
            if user_guess_value_list is None:
                model = hmm_model_OLH(epsilon, k, index + 1, 'plain')
            else:
                model = hmm_model_OLH(epsilon, k, index + 1, 'plain')
                for _ in range(test_count):
                    guess_list = list()
                    perturbed_value_list = perturb(protocol_type='OLH', epsilon=epsilon, k=k,
                                                   user_true_value_list=user_values_list)

                    for perturbed_value in perturbed_value_list:
                        guess_list.append(guess(model, perturbed_value))
                    model = hmm_model_OLH(epsilon, k, index + 1, 'advance', guess_list)

        guessed_users_value = guess(model, user_perturbed_report)
        guess_metric.append(ratio_of_guess(user_values_list[index], guessed_users_value))

    return np.average(guess_metric)


def GRR_estimated_guess(user_values_list, k, epsilon):
    model = hmm_model_GRR(epsilon, k, 'plain')
    return experiment(epsilon, k, user_values_list, "GRR", model)


def GRR_FK_estimated_guess(user_values_list, k, epsilon):
    model = hmm_model_GRR(epsilon, k, 'advance', user_values_list)
    return experiment(epsilon, k, user_values_list, "GRR", model)


def GRR_advance_estimated_guess(user_values_list, k, epsilon, test_count):
    model = hmm_model_GRR(epsilon, k, 'plain')

    for _ in range(test_count):
        guess_from_hmm = list()
        perturbed_value_list = perturb(protocol_type='GRR', epsilon=epsilon, k=k, user_true_value_list=user_values_list)

        for perturbed_value in perturbed_value_list:
            guess_from_hmm.append(guess(model, perturbed_value))
        model = hmm_model_GRR(epsilon, k, 'advance', guess_from_hmm)

    return experiment(epsilon, k, user_values_list, "GRR", model)


def RAPPOR_estimated_guess(user_values_list, k, epsilon):
    model = hmm_model_RAPPOR(epsilon, k, 'plain')
    return experiment(epsilon, k, user_values_list, "RAPPOR", model)


def RAPPOR_FK_estimated_guess(user_values_list, k, epsilon):
    model = hmm_model_RAPPOR(epsilon, k, 'advance', user_values_list)
    return experiment(epsilon, k, user_values_list, "RAPPOR", model)


def RAPPOR_advance_estimated_guess(user_values_list, k, epsilon, test_count):
    model = hmm_model_RAPPOR(epsilon, k, 'plain')

    for _ in range(test_count):
        guess_from_hmm = list()
        perturbed_value_list = perturb(protocol_type='RAPPOR', epsilon=epsilon, k=k,
                                       user_true_value_list=user_values_list)

        for perturbed_value in perturbed_value_list:
            guess_from_hmm.append(guess(model, perturbed_value))
        model = hmm_model_RAPPOR(epsilon, k, 'advance', guess_from_hmm)

    return experiment(epsilon, k, user_values_list, "RAPPOR", model)


def OUE_estimated_guess(user_values_list, k, epsilon):
    model = hmm_model_OUE(epsilon, k, 'plain')
    return experiment(epsilon, k, user_values_list, "OUE", model)


def OUE_advance_estimated_guess(user_values_list, k, epsilon, test_count):
    model = hmm_model_OUE(epsilon, k, 'plain')

    for _ in range(test_count):
        guess_from_hmm = list()
        perturbed_value_list = perturb(protocol_type='OUE', epsilon=epsilon, k=k,
                                       user_true_value_list=user_values_list)

        for perturbed_value in perturbed_value_list:
            guess_from_hmm.append(guess(model, perturbed_value))
        model = hmm_model_OUE(epsilon, k, 'advance', guess_from_hmm)

    return experiment(epsilon, k, user_values_list, "OUE", model)


def OUE_FK_estimated_guess(user_values_list, k, epsilon):
    model = hmm_model_OUE(epsilon, k, 'advance', user_values_list)
    return experiment(epsilon, k, user_values_list, "OUE", model)


def OLH_estimated_guess(user_values_list, k, epsilon):
    return experiment(epsilon, k, user_values_list, "OLH")


def OLH_advance_estimated_guess(user_values_list, k, epsilon, test_count):
    return experiment(epsilon, k, user_values_list, "OLH", user_guess_value_list=user_values_list,
                      test_count=test_count)


def OLH_FK_estimated_guess(user_values_list, k, epsilon):
    return experiment(epsilon, k, user_values_list, "OLH", user_values_list)
