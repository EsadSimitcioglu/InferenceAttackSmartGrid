import csv
import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from LDP.estimation import GRR_estimated_guess_plain, GRR_estimated_guess_advance, RAPPOR_estimated_guess, \
    RAPPOR_estimated_guess_advance, OUE_estimated_guess, \
    OUE_estimated_guess_advance, OLH_estimated_guess, OLH_estimated_guess_advance

# Parameters for simulation
k = 11  # attribute's domain size
epsilon_list = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5]  # number of epsilon for test cases
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
            percentage = int(float(value) / float(row[1]) * 100)
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

for epsilon in epsilon_list:
    print("Epsilon Value: " + str(epsilon))
    probability_of_guess_grr.append(GRR_estimated_guess_plain(users_consumption_value_list, k, epsilon))
    print("GRR is Ready")
    probability_of_guess_rappor.append(RAPPOR_estimated_guess(users_consumption_value_list, k, epsilon))
    print("RAPPOR is Ready")
    probability_of_guess_oue.append(OUE_estimated_guess(users_consumption_value_list, k, epsilon))
    print("OUE is Ready")
    # probability_of_guess_olh.append(OLH_estimated_guess_advance(users_consumption_value_list, k, epsilon))
    print("OLH is Ready")

print(probability_of_guess_grr)
print(probability_of_guess_rappor)
print(probability_of_guess_oue)
plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(4 * 1.33, 4 * 1.33))
plt.plot(epsilon_list, probability_of_guess_grr, linewidth=2, color='purple', marker='o', markersize=10, mew=1.5, fillstyle='none', clip_on=False, label="GRR")
plt.plot(epsilon_list, probability_of_guess_rappor, linewidth=2, color='grey', marker='s', markersize=10, mew=1.5, fillstyle='none', clip_on=False, label="RAPPOR")
plt.plot(epsilon_list, probability_of_guess_oue, linewidth=2, color='blue', marker='x', markersize=10, mew=1.5, fillstyle='none', clip_on=False, label="OUE")
plt.ylim(0, 1)
plt.xticks(fontsize=15)
plt.ylabel("Ratio Of Guess")
plt.xlabel('Epsilon Values')
plt.grid(linestyle=':')
plt.legend(prop={'size': 12}, ncol=2, columnspacing=0.75)
plt.savefig('SmartGridLDP.png', format='png', dpi=300, bbox_inches='tight')
plt.show()
