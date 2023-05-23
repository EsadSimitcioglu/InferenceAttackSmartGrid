import numpy
import numpy as np
import xxhash
from sys import maxsize

# [1] Erlingsson, Pihur, and Korolova (2014) "RAPPOR: Randomized aggregatable privacy-preserving ordinal response" (ACM CCS).
# [2] Arcolezi et al (2022) "Improving the Utility of Locally Differentially Private Protocols for Longitudinal and Multidimensional Frequency Estimates" (Digital Communications and Networks).
# [3] Ding, Kulkarni, and Yekhanin (2017) "Collecting telemetry data privately." (NeurIPS).
from numpy import exp

seeds = np.arange(200)
seed_counter = 0


def setting_seed(seed):
    """ Function to set seed for reproducibility.
    Calling numpy.random.seed() from interpreted code will
    seed the NumPy random generator, not the Numba random generator.
    Check: https://numba.readthedocs.io/en/stable/reference/numpysupported.html"""

    np.random.seed(seed)


def GRR_Client(input_data, k, epsilon):
    """
    Generalized Randomized Response (GRR) protocol, a.k.a., direct encoding [1] or k-RR [2].
    :param input_data: user's true value;
    :param k: attribute's domain size;
    :param epsilon: privacy guarantee;
    :return: sanitized value.
    """

    if epsilon is not None or k is not None:

        # GRR parameters
        p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)

        # Mapping domain size k to the range [0, ..., k-1]
        domain = np.arange(1, k + 1)

        # GRR perturbation function
        rnd = np.random.random()
        if rnd <= p:
            return input_data - 1
        else:
            return np.random.choice(domain[domain != input_data] - 1)

    else:
        raise ValueError('k (int) and epsilon (float) need a numerical value.')


def GRR_Aggregator(reports, k, epsilon):
    """
    Statistical Estimator for Normalized Frequency (0 -- 1) with post-processing to ensure non-negativity.
    :param reports: list of all GRR-based sanitized values;
    :param k: attribute's domain size;
    :param epsilon: privacy guarantee;
    :return: normalized frequency (histogram) estimation.
    """

    if len(reports) == 0:

        raise ValueError('List of reports is empty.')

    else:

        if epsilon is not None or k is not None:

            # Number of reports
            n = len(reports)

            # GRR parameters
            p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
            q = (1 - p) / (k - 1)

            # Count how many times each value has been reported
            count_report = np.zeros(k)
            for rep in reports:
                count_report[rep] += 1

            # Ensure non-negativity of estimated frequency
            est_freq = np.array((count_report - n * q) / (p - q)).clip(0)

            # Re-normalized estimated frequency
            norm_est_freq = np.nan_to_num(est_freq / sum(est_freq))

            return norm_est_freq

        else:
            raise ValueError('k (int) and epsilon (float) need a numerical value.')


def UE_Client(input_ue_data, k, p, q):
    """
    Unary Encoding (UE) protocol
    """

    # Initializing a zero-vector
    sanitized_vec = np.zeros(k)

    # UE perturbation function
    for ind in range(k):
        if input_ue_data[ind] != 1:
            rnd = np.random.random()
            if rnd <= q:
                sanitized_vec[ind] = 1
        else:
            rnd = np.random.random()
            if rnd <= p:
                sanitized_vec[ind] = 1
    return sanitized_vec


# Ours: LOLOHA
def LOLOHA_Client(input_sequence, k, eps_perm, eps_1, alpha, optimal=True):
    # BiLOLOHA parameter
    g = 2

    if optimal:
        # Optimal LH (OLOLOHA) parameter
        g = int(max(np.rint((np.sqrt(
            np.exp(4 * eps_perm) - 14 * np.exp(2 * eps_perm) - 12 * np.exp(2 * eps_perm * (alpha + 1)) + 12 * np.exp(
                eps_perm * (alpha + 1)) + 12 * np.exp(eps_perm * (alpha + 3)) + 1) - np.exp(2 * eps_perm) + 6 * np.exp(
            eps_perm) - 6 * np.exp(eps_perm * alpha) + 1) / (6 * (np.exp(eps_perm) - np.exp(eps_perm * alpha)))), 2))

    # GRR parameters for round 1
    p1_llh = np.exp(eps_perm) / (np.exp(eps_perm) + g - 1)
    q1_llh = (1 - p1_llh) / (g - 1)

    # GRR parameters for round 2
    p2_llh = (q1_llh - np.exp(eps_1) * p1_llh) / (
            (-p1_llh * np.exp(eps_1)) + g * q1_llh * np.exp(eps_1) - q1_llh * np.exp(eps_1) - p1_llh * (
            g - 1) + q1_llh)
    q2_llh = (1 - p2_llh) / (g - 1)

    # Cache for memoized values
    lst_memoized = {val: None for val in range(g)}

    # Random 'hash function', i.e., a seed to use xxhash package
    rnd_seed = np.random.randint(0, maxsize, dtype=np.int64)

    # List of sanitized reports throughout \tau data collections
    sanitized_reports = []
    for input_data in input_sequence:

        # Hash the user's value
        hashed_input_data = (xxhash.xxh32(str(input_data), seed=rnd_seed).intdigest() % g)

        if lst_memoized[hashed_input_data] is None:  # If hashed value not memoized

            # Memoization
            first_sanitization = GRR_Client(hashed_input_data, g, p1_llh)
            lst_memoized[hashed_input_data] = first_sanitization

        else:  # Use already memoized hashed value
            first_sanitization = lst_memoized[hashed_input_data]

        sanitized_reports.append((GRR_Client(first_sanitization, g, p2_llh), rnd_seed))

    # Number of hash value changes, i.e, of privacy budget consumption
    final_budget = sum([val is not None for val in lst_memoized.values()])

    return sanitized_reports, final_budget


def LOLOHA_Aggregator(reports, k, eps_perm, eps_1, alpha, optimal=True):
    # Number of reports
    n = len(reports)

    # BiLOLOHA parameter
    g = 2

    if optimal:
        # Optimal LH (OLOLOHA) parameter
        g = max(np.rint((np.sqrt(
            np.exp(4 * eps_perm) - 14 * np.exp(2 * eps_perm) - 12 * np.exp(2 * eps_perm * (alpha + 1)) + 12 * np.exp(
                eps_perm * (alpha + 1)) + 12 * np.exp(eps_perm * (alpha + 3)) + 1) - np.exp(2 * eps_perm) + 6 * np.exp(
            eps_perm) - 6 * np.exp(eps_perm * alpha) + 1) / (6 * (np.exp(eps_perm) - np.exp(eps_perm * alpha)))), 2)

    # GRR parameters for round 1
    p1 = np.exp(eps_perm) / (np.exp(eps_perm) + g - 1)
    q1 = (1 - p1) / (g - 1)

    # GRR parameters for round 2
    p2 = (q1 - np.exp(eps_1) * p1) / (
            (-p1 * np.exp(eps_1)) + g * q1 * np.exp(eps_1) - q1 * np.exp(eps_1) - p1 * (g - 1) + q1)
    q2 = (1 - p2) / (g - 1)

    # Count how many times each value has been reported
    q1 = 1 / g  # updating q1 in the server
    count_report = np.zeros(k)
    for tuple_val_seed in reports:
        usr_val = tuple_val_seed[0]  # user's sanitized value
        usr_seed = tuple_val_seed[1]  # user's 'hash function'
        for v in range(k):
            if usr_val == (xxhash.xxh32(str(v), seed=usr_seed).intdigest() % g):
                count_report[v] += 1

    # Ensure non-negativity of estimated frequency
    est_freq = ((count_report - n * q1 * (p2 - q2) - n * q2) / (n * (p1 - q1) * (p2 - q2))).clip(0)

    # Re-normalized estimated frequency
    norm_est_freq = np.nan_to_num(est_freq / sum(est_freq))

    return norm_est_freq


# Competitor: RAPPOR [1]
def RAPPOR_Client(input_sequence, k, eps_perm, eps_1):
    # The analytical analysis of how to calculate parameters (p1, q2, p2, q2) is from [2]

    # Sue parameters for round 1
    p1 = np.exp(eps_perm / 2) / (np.exp(eps_perm / 2) + 1)
    q1 = 1 - p1

    # SUE parameters for round 2
    p2 = - (np.sqrt((4 * np.exp(7 * eps_perm / 2) - 4 * np.exp(5 * eps_perm / 2) - 4 * np.exp(
        3 * eps_perm / 2) + 4 * np.exp(eps_perm / 2) + np.exp(4 * eps_perm) + 4 * np.exp(3 * eps_perm) - 10 * np.exp(
        2 * eps_perm) + 4 * np.exp(eps_perm) + 1) * np.exp(eps_1)) * (np.exp(eps_1) - 1) * (
                    np.exp(eps_perm) - 1) ** 2 - (
                    np.exp(eps_1) - np.exp(2 * eps_perm) + 2 * np.exp(eps_perm) - 2 * np.exp(
                eps_1 + eps_perm) + np.exp(eps_1 + 2 * eps_perm) - 1) * (
                    np.exp(3 * eps_perm / 2) - np.exp(eps_perm / 2) + np.exp(eps_perm) - np.exp(
                eps_1 + eps_perm / 2) - np.exp(eps_1 + eps_perm) + np.exp(eps_1 + 3 * eps_perm / 2) + np.exp(
                eps_1 + 2 * eps_perm) - 1)) / ((np.exp(eps_1) - 1) * (np.exp(eps_perm) - 1) ** 2 * (
            np.exp(eps_1) - np.exp(2 * eps_perm) + 2 * np.exp(eps_perm) - 2 * np.exp(eps_1 + eps_perm) + np.exp(
        eps_1 + 2 * eps_perm) - 1))
    q2 = 1 - p2

    # Cache for memoized values
    lst_memoized = {val: None for val in range(k)}

    # List of sanitized reports throughout \tau data collections
    sanitized_reports = []
    for input_data in input_sequence:

        # Unary encoding
        input_ue_data = np.zeros(k)
        input_ue_data[input_data] = 1

        if lst_memoized[input_data] is None:  # If hashed value not memoized

            # Memoization
            first_sanitization = UE_Client(input_ue_data, k, p1, q1)
            lst_memoized[input_data] = first_sanitization

        else:  # Use already memoized hashed value
            first_sanitization = lst_memoized[input_data]

        sanitized_reports.append(UE_Client(first_sanitization, k, p2, q2))

    # Number of data value changes, i.e, of privacy budget consumption
    final_budget = sum([val is not None for val in lst_memoized.values()])

    return sanitized_reports, final_budget


def RAPPOR_Aggregator(ue_reports, eps_perm, eps_1):
    # Number of reports
    n = len(ue_reports)

    # SUE parameters for round 1
    p1 = np.exp(eps_perm / 2) / (np.exp(eps_perm / 2) + 1)
    q1 = 1 - p1

    # SUE parameters for round 2
    p2 = - (np.sqrt((4 * np.exp(7 * eps_perm / 2) - 4 * np.exp(5 * eps_perm / 2) - 4 * np.exp(
        3 * eps_perm / 2) + 4 * np.exp(eps_perm / 2) + np.exp(4 * eps_perm) + 4 * np.exp(3 * eps_perm) - 10 * np.exp(
        2 * eps_perm) + 4 * np.exp(eps_perm) + 1) * np.exp(eps_1)) * (np.exp(eps_1) - 1) * (
                    np.exp(eps_perm) - 1) ** 2 - (
                    np.exp(eps_1) - np.exp(2 * eps_perm) + 2 * np.exp(eps_perm) - 2 * np.exp(
                eps_1 + eps_perm) + np.exp(eps_1 + 2 * eps_perm) - 1) * (
                    np.exp(3 * eps_perm / 2) - np.exp(eps_perm / 2) + np.exp(eps_perm) - np.exp(
                eps_1 + eps_perm / 2) - np.exp(eps_1 + eps_perm) + np.exp(eps_1 + 3 * eps_perm / 2) + np.exp(
                eps_1 + 2 * eps_perm) - 1)) / ((np.exp(eps_1) - 1) * (np.exp(eps_perm) - 1) ** 2 * (
            np.exp(eps_1) - np.exp(2 * eps_perm) + 2 * np.exp(eps_perm) - 2 * np.exp(eps_1 + eps_perm) + np.exp(
        eps_1 + 2 * eps_perm) - 1))
    q2 = 1 - p2

    # Ensure non-negativity of estimated frequency
    est_freq = ((sum(ue_reports) - n * q1 * (p2 - q2) - n * q2) / (n * (p1 - q1) * (p2 - q2))).clip(0)

    # Re-normalized estimated frequencies
    norm_est_freq = np.nan_to_num(est_freq / sum(est_freq))

    return norm_est_freq


def SIMPLE_RAPPOR_Client(input_data, k, epsilon):
    bit_vector = np.zeros(k)
    bit_vector[input_data - 1] = 1

    p = (np.exp(epsilon / 2)) / (np.exp(epsilon / 2) + 1)

    perturbed_bit_vector = bit_vector.copy()
    for bit_index in range(len(bit_vector)):
        rnd = np.random.random()
        if rnd > p:
            if perturbed_bit_vector[bit_index] == 1:
                perturbed_bit_vector[bit_index] = 0
            else:
                perturbed_bit_vector[bit_index] = 1

    perturbed_bit_vector = perturbed_bit_vector.tolist()
    return perturbed_bit_vector


def SIMPLE_RAPPOR_Aggregator(perturbed_bit_vectors, epsilon):
    perturbed_bit_vectors = numpy.array(perturbed_bit_vectors)
    n = 2
    p = (np.exp(epsilon / 2)) / (np.exp(epsilon / 2) + 1)
    q = 1 / (np.exp(epsilon / 2) + 1)
    perturbed_sum_bit_vector = sum(perturbed_bit_vectors)
    est_freq_vector = list()

    for sum_bit in perturbed_sum_bit_vector:
        numerator = sum_bit - (n * q)
        denominator = p - q
        est_freq_vector.append(numerator / denominator)

    # Re-normalized estimated frequencies
    norm_est_freq = np.nan_to_num(est_freq_vector / sum(est_freq_vector))

    return norm_est_freq


# Competitor: L-OSUE [2]
def L_OSUE_Client(input_sequence, k, eps_perm, eps_1):
    # The analytical analysis of how to calculate parameters (p1, q2, p2, q2) is from [2]

    # OUE parameters for round 1
    p1 = 1 / 2
    q1 = 1 / (np.exp(eps_perm) + 1)

    # SUE parameters for round 2
    p2 = (1 - np.exp(eps_1 + eps_perm)) / (np.exp(eps_1) - np.exp(eps_perm) - np.exp(eps_1 + eps_perm) + 1)
    q2 = 1 - p2

    # Cache for memoized values
    lst_memoized = {val: None for val in range(k)}

    # List of sanitized reports throughout \tau data collections
    sanitized_reports = []
    for input_data in input_sequence:

        # Unary encoding
        input_ue_data = np.zeros(k)
        input_ue_data[input_data] = 1

        if lst_memoized[input_data] is None:  # If hashed value not memoized

            # Memoization
            first_sanitization = UE_Client(input_ue_data, k, p1, q1)
            lst_memoized[input_data] = first_sanitization

        else:  # Use already memoized hashed value
            first_sanitization = lst_memoized[input_data]

        sanitized_reports.append(UE_Client(first_sanitization, k, p2, q2))

    # Number of data value changes, i.e, of privacy budget consumption
    final_budget = sum([val is not None for val in lst_memoized.values()])

    return sanitized_reports, final_budget


def L_OSUE_Aggregator(ue_reports, eps_perm, eps_1):
    # Number of reports
    n = len(ue_reports)

    # OUE parameters for round 1
    p1 = 1 / 2
    q1 = 1 / (np.exp(eps_perm) + 1)

    # SUE parameters for round 2
    p2 = (1 - np.exp(eps_1 + eps_perm)) / (np.exp(eps_1) - np.exp(eps_perm) - np.exp(eps_1 + eps_perm) + 1)
    q2 = 1 - p2

    # Ensure non-negativity of estimated frequency
    est_freq = ((sum(ue_reports) - n * q1 * (p2 - q2) - n * q2) / (n * (p1 - q1) * (p2 - q2))).clip(0)

    # Re-normalized estimated frequency
    norm_est_freq = np.nan_to_num(est_freq / sum(est_freq))

    return norm_est_freq


# Competitor: L-GRR [2]
def L_GRR_Client(input_sequence, k, eps_perm, eps_1):
    # The analytical analysis of how to calculate parameters (p1, q2, p2, q2) is from [2]

    # GRR parameters for round 1
    p1 = np.exp(eps_perm) / (np.exp(eps_perm) + k - 1)
    q1 = (1 - p1) / (k - 1)

    # GRR parameters for round 2
    p2 = (q1 - np.exp(eps_1) * p1) / (
            (-p1 * np.exp(eps_1)) + k * q1 * np.exp(eps_1) - q1 * np.exp(eps_1) - p1 * (k - 1) + q1)
    q2 = (1 - p2) / (k - 1)

    # Cache for memoized values
    lst_memoized = {val: None for val in range(k)}

    # List of sanitized reports throughout \tau data collections
    sanitized_reports = []
    for input_data in input_sequence:

        if lst_memoized[input_data] is None:  # If hashed value not memoized

            # Memoization
            first_sanitization = GRR_Client(input_data, k, p1)
            lst_memoized[input_data] = first_sanitization

        else:  # Use already memoized hashed value
            first_sanitization = lst_memoized[input_data]

        sanitized_reports.append(GRR_Client(first_sanitization, k, p2))

    # Number of data value changes, i.e, of privacy budget consumption
    final_budget = sum([val is not None for val in lst_memoized.values()])

    return sanitized_reports, final_budget


def L_GRR_Aggregator(reports, k, eps_perm, eps_1):
    # Number of reports
    n = len(reports)

    # GRR parameters for round 1
    p1 = np.exp(eps_perm) / (np.exp(eps_perm) + k - 1)
    q1 = (1 - p1) / (k - 1)

    # GRR parameters for round 2
    p2 = (q1 - np.exp(eps_1) * p1) / (
            (-p1 * np.exp(eps_1)) + k * q1 * np.exp(eps_1) - q1 * np.exp(eps_1) - p1 * (k - 1) + q1)
    q2 = (1 - p2) / (k - 1)

    # Count how many times each value has been reported
    count_report = np.zeros(k)
    for rep in reports:
        count_report[rep] += 1

    # Ensure non-negativity of estimated frequency
    est_freq = ((count_report - n * q1 * (p2 - q2) - n * q2) / (n * (p1 - q1) * (p2 - q2))).clip(0)

    # Re-normalized estimated frequency
    norm_est_freq = np.nan_to_num(est_freq / sum(est_freq))

    return norm_est_freq


# Competitor: dBitFlipPM [3]
def dBitFlipPM_Client(input_sequence, k, b, d, eps_perm):
    def dBit(bucketized_data, b, d, j, p1, q1):

        # b = 96
        # d = 1
        # k = 1

        # Unary encoding
        permanent_sanitization = np.ones(b) * - 1  # set to -1 non-sampled bits

        # Permanent Memoization
        idx_j = 0
        for i in range(b):
            if i in j:  # only the sampled bits
                rand = np.random.random()
                if bucketized_data == j[idx_j]:
                    permanent_sanitization[j[idx_j]] = int(rand <= p1)
                else:
                    permanent_sanitization[j[idx_j]] = int(rand <= q1)

                idx_j += 1
        return permanent_sanitization

    # SUE parameters
    p1 = np.exp(eps_perm / 2) / (np.exp(eps_perm / 2) + 1)
    q1 = 1 - p1

    # calculate bulk number of user's value
    bulk_size = k / b

    # Cache for memoized values
    lst_memoized = {val: None for val in range(b)}

    # Random bits
    j = np.random.choice(range(0, b), d, replace=False)

    # Use already memoized hashed value
    sanitized_reports = []
    for input_data in input_sequence:

        bucketized_data = int(input_data / bulk_size)

        # Unary encoding
        input_ue_data = np.zeros(b)
        input_ue_data[bucketized_data] = 1

        if lst_memoized[bucketized_data] is None:  # List of sanitized reports throughout \tau data collections

            # Memoization
            first_sanitization = dBit(bucketized_data, b, d, j, p1, q1)
            lst_memoized[bucketized_data] = first_sanitization

        else:  # If hashed value not memoized
            first_sanitization = lst_memoized[bucketized_data]

        sanitized_reports.append(first_sanitization)

    # Number of bucket value changes, i.e, of privacy budget consumption
    final_budget = sum([val is not None for val in lst_memoized.values()])

    # Number memoized responses
    nb_changes = []
    for key in lst_memoized.keys():
        if lst_memoized[key] is not None:
            nb_changes.append(list(lst_memoized[key]))

    # Boolean value to indicate if number of memoized responses equal number of bucket value changes
    detect_change = final_budget == len(np.unique(np.array(nb_changes), axis=0))

    return sanitized_reports, final_budget, detect_change


def dBitFlipPM_Aggregator(reports, b, d, eps_perm):
    # Estimated frequency of each bucket
    est_freq = []
    for v in range(b):
        h = 0
        for bi in reports:
            if bi[v] >= 0:  # only the sampled bits
                h += (bi[v] * (np.exp(eps_perm / 2) + 1) - 1) / (np.exp(eps_perm / 2) - 1)
        est_freq.append(h * b / (len(reports) * d))

    # Ensure non-negativity of estimated frequency
    est_freq = np.array(est_freq).clip(0)

    # Re-normalized estimated frequency
    norm_est = np.nan_to_num(est_freq / sum(est_freq))
    return norm_est


def OLH_Client(input_data_list, k, epsilon, seed_init):
    p = exp(epsilon) / (exp(epsilon) + k - 1)
    g = int(round(np.exp(epsilon))) + 1
    report_list = list()

    for input_data in input_data_list:
        report_value = (xxhash.xxh32(str(input_data), seed=seed_init).intdigest() % g)
        rnd = np.random.random()
        if rnd > p:
            report_value = np.random.randint(0, g)
        report_list.append(report_value)
        seed_init += 1
    return report_list


def OLH_Client2(input_data_list, k, epsilon, seed_init):
    p = exp(epsilon) / (exp(epsilon) + k - 1)
    g = int(round(np.exp(epsilon))) + 1
    report_list = list()

    for input_data in input_data_list:
        input_data -= 1
        report_value = (xxhash.xxh32(str(input_data), seed=seed_init).intdigest() % g)
        rnd = np.random.random()
        if rnd > p:
            report_value = np.random.randint(0, g)
        report_list.append(report_value)
    return report_list


def OLH_Aggregator(perturbed_datas, n, k, epsilon):
    g = int(round(exp(epsilon))) + 1
    p = exp(epsilon) / (exp(epsilon) + g - 1)

    global ESTIMATE_DIST
    ESTIMATE_DIST = np.zeros((k + 1))
    for i in range(n):
        for v in range((k + 1)):
            if perturbed_datas[i] == (xxhash.xxh32(str(v), seed=i).intdigest() % g):
                ESTIMATE_DIST[v] += 1
    a = 1.0 * g / (p * g - 1)
    b = 1.0 * n / (p * g - 1)
    ESTIMATE_DIST = a * ESTIMATE_DIST - b

    norm_est_freq = np.nan_to_num(ESTIMATE_DIST / sum(ESTIMATE_DIST))
    return norm_est_freq


def OLH_Aggregator2(reports, n, k, epsilon):
    # GRR parameters
    p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
    q = (1 - p) / (k - 1)

    # Count how many times each value has been reported
    count_report = np.zeros(k)

    for i in range(n):
        for v in range(k):
            if reports[i] == (xxhash.xxh32(str(v), seed=i).intdigest() % k):
                count_report[v] += 1

    # Ensure non-negativity of estimated frequency
    est_freq = np.array((count_report - n * q) / (p - q)).clip(0)

    # Re-normalized estimated frequency
    norm_est_freq = np.nan_to_num(est_freq / sum(est_freq))

    return norm_est_freq


def OUE_Client(input_data, k, epsilon):
    bit_vector = np.zeros(k)
    bit_vector[input_data - 1] = 1

    p = 1 / 2
    q = 1 / (np.exp(epsilon) + 1)

    perturbed_bit_vector = bit_vector.copy()
    for bit_index in range(k):
        if perturbed_bit_vector[bit_index] == 0:
            rnd = np.random.random()
            if rnd <= q:
                perturbed_bit_vector[bit_index] = 1
        else:
            rnd = np.random.random()
            if rnd > p:
                perturbed_bit_vector[bit_index] = 0
    return perturbed_bit_vector


def OUE_Aggregator(perturbed_bit_vectors, epsilon):
    n = 2
    perturbed_sum_bit_vector = sum(perturbed_bit_vectors)
    est_freq_vector = list()

    for sum_bit in perturbed_sum_bit_vector:
        numerator = 2 * ((np.exp(epsilon) + 1) * sum_bit - n)
        denominator = np.exp(epsilon) - 1
        est_freq_vector.append(numerator / denominator)

    # Re-normalized estimated frequencies
    norm_est_freq = np.nan_to_num(est_freq_vector / sum(est_freq_vector))
    return norm_est_freq