import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from LDP.estimation import GRR_estimated_guess_plain, GRR_estimated_guess_advance, RAPPOR_estimated_guess,RAPPOR_estimated_guess_advance, OUE_estimated_guess, \
    OUE_estimated_guess_advance, OLH_estimated_guess, OLH_estimated_guess_advance

# Parameters for simulation
k = 11  # attribute's domain size
epsilon_list = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5]  # number of epsilon for test cases
users_consumption_value_list = list()
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
            consumption_list.append((math.ceil(percentage/10)))
        users_consumption_value_list.append(np.array(consumption_list))

for epsilon in epsilon_list:
    print("Epsilon Value: " + str(epsilon))
    probability_of_guess_grr.append(GRR_estimated_guess_advance(users_consumption_value_list, k, epsilon))
    print("GRR is Ready")
    probability_of_guess_rappor.append(RAPPOR_estimated_guess_advance(users_consumption_value_list, k, epsilon))
    print("RAPPOR is Ready")
    probability_of_guess_oue.append(OUE_estimated_guess_advance(users_consumption_value_list, k, epsilon))
    print("OUE is Ready")
    #probability_of_guess_olh.append(OLH_estimated_guess_advance(users_consumption_value_list, k, epsilon))
    print("OLH is Ready")

print(probability_of_guess_grr)
print(probability_of_guess_rappor)
print(probability_of_guess_oue)
#print(probability_of_guess_olh)
plt.ylim(0, 1)
plt.xlim(min(epsilon_list), max(epsilon_list))
plt.plot(epsilon_list, probability_of_guess_grr, label='GRR', color='red')
plt.plot(epsilon_list, probability_of_guess_rappor, label='RAPPOR', color='green')
plt.plot(epsilon_list, probability_of_guess_oue, label='OUE', color='yellow')
#plt.plot(epsilon_list, probability_of_guess_olh, label='OLH', color='purple')
plt.ylabel('Probability of Guess')
plt.xlabel('Epsilon values')
plt.legend(loc='upper right', bbox_to_anchor=(1.015, 1.15))
plt.show()