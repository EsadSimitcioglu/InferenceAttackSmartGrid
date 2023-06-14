import csv
import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from LDP_Protocols.estimation import GRR_estimated_guess, RAPPOR_estimated_guess, OUE_estimated_guess, OLH_estimated_guess, \
    GRR_advance_estimated_guess, RAPPOR_advance_estimated_guess, OUE_advance_estimated_guess, \
    OLH_advance_estimated_guess

# Parameters for simulation
k = 11  # attribute's domain size
maximum_value = 2.113  # maximum value of the dataset
epsilon_list = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5]  # number of epsilon for test cases
alpha_list = [1, 2, 3, 4, 5]  # number of alpha for test cases
users_consumption_value_list = list()
user_date_values_dict = dict()
probability_of_guess_grr = list()
probability_of_guess_rappor = list()
probability_of_guess_oue = list()
probability_of_guess_olh = list()

df = pd.read_csv('dataset/2012-2013 Solar home electricity data v2.csv')
customer_id = 0
main_list = list()
for row in df.values:
    if row[3] == 'GC':
        customer_id = row[0]
        date = row[4]
        consumption_list = list()
        for value in row[5: 53]:
            percentage = int(float(value) / maximum_value * 100)
            consumption_list.append((math.ceil(percentage / 10)))
        user_date_values_dict[date] = consumption_list
        users_consumption_value_list.append(np.array(consumption_list))

# Save dictionary to csv
with open('dataset/User_Consumption_Percentage_Dict.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    for date, true_values_list in user_date_values_dict.items():

        # Skip writing the row if all values are empty
        if any(true_values_list):
            writer.writerow([date] + true_values_list)

for alpha in alpha_list:

    print("Epsilon Value: " + str(alpha))
    temp_probability_of_guess_grr = list()
    temp_probability_of_guess_rappor = list()
    temp_probability_of_guess_oue = list()
    temp_probability_of_guess_olh = list()

    for _ in range(10):

        grr_est_freq = GRR_advance_estimated_guess(users_consumption_value_list, k, 3, alpha)
        temp_probability_of_guess_grr.append(grr_est_freq)

        rappor_est_freq = RAPPOR_advance_estimated_guess(users_consumption_value_list, k, 3, alpha)
        temp_probability_of_guess_rappor.append(rappor_est_freq)

        oue_est_freq = OUE_advance_estimated_guess(users_consumption_value_list, k, 3, alpha)
        temp_probability_of_guess_oue.append(oue_est_freq)

        olh_est_freq = OLH_advance_estimated_guess(users_consumption_value_list, k, 3, alpha)
        temp_probability_of_guess_olh.append(olh_est_freq)

    probability_of_guess_grr.append(sum(temp_probability_of_guess_grr) / len(temp_probability_of_guess_grr))
    probability_of_guess_rappor.append(sum(temp_probability_of_guess_rappor) / len(temp_probability_of_guess_rappor))
    probability_of_guess_oue.append(sum(temp_probability_of_guess_oue) / len(temp_probability_of_guess_oue))
    probability_of_guess_olh.append(sum(temp_probability_of_guess_olh) / len(temp_probability_of_guess_olh))

    print("GRR: " + str(sum(temp_probability_of_guess_grr) / len(temp_probability_of_guess_grr)))
    print("RAPPOR: " + str(sum(temp_probability_of_guess_rappor) / len(temp_probability_of_guess_rappor)))
    print("OUE: " + str(sum(temp_probability_of_guess_oue) / len(temp_probability_of_guess_oue)))
    print("OLH: " + str(sum(temp_probability_of_guess_olh) / len(temp_probability_of_guess_olh)))

plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(4 * 1.33, 4 * 1.33))
plt.plot(alpha_list, probability_of_guess_grr, linewidth=2, color='purple', marker='o', markersize=10, mew=1.5,
         fillstyle='none', clip_on=False, label="GRR")
plt.plot(alpha_list, probability_of_guess_rappor, linewidth=2, color='grey', marker='s', markersize=10, mew=1.5,
         fillstyle='none', clip_on=False, label="RAPPOR")
plt.plot(alpha_list, probability_of_guess_oue, linewidth=2, color='blue', marker='x', markersize=10, mew=1.5,
         fillstyle='none', clip_on=False, label="OUE")
plt.plot(alpha_list, probability_of_guess_olh, linewidth=2, color='green', marker='d', markersize=10, mew=1.5,
         fillstyle='none', clip_on=False, label="OLH")
plt.ylim(0, 1)
plt.xticks(fontsize=15)
plt.ylabel("Ratio Of Guess")
plt.xlabel('Alpha Values')
plt.grid(linestyle=':')
plt.legend(prop={'size': 12}, ncol=2, columnspacing=0.75)
plt.savefig('ChainAttack.png', format='png', dpi=300, bbox_inches='tight')
plt.show()