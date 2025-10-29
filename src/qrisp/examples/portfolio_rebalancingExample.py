# table below alternates between avg return and covar

# create data --
import numpy as np

data1 = [
    0.000401,
    0.009988,
    -0.000316,
    0.014433,
    0.000061,
    0.010024,
    0.001230,
    0.014854,
    0.000916,
    0.013465,
    -0.000176,
    0.010974,
    -0.000619,
    0.015910,
    0.000396,
    0.010007,
    0.000212,
    0.009201,
    -0.000881,
    0.013377,
    0.001477,
    0.013156,
    0.000184,
    0.009907,
    0.001047,
    0.011216,
    0.000492,
    0.008399,
    0.000794,
    0.010052,
    0.000291,
    0.013247,
    0.000204,
    0.009193,
    0.000674,
    0.008477,
    0.001500,
    0.014958,
    0.000491,
    0.010873,
]
mu = [data1[i] for i in range(len(data1)) if i % 2 == 0]
from sklearn import preprocessing

mu_array = np.array(mu)
norm_mu_full = preprocessing.normalize([mu_array])

# create covar_matrix -- from the original protfolio balancing paper
covar_string = "99.8 42.5 37.2 40.3 38.0 30.0 46.8 14.9 42.5 100.5 41.1 15.2 71.1 27.8 47.5 12.7 37.2 41.1 181.3 17.9 38.4 27.9 39.0 8.3 40.3 15.2 17.9 253.1 12.4 48.7 33.3 3.8 38.0 71.1 38.4 12.4 84.7 28.5 42.0 13.1 30.0 27.8 27.9 48.7 28.5 173.1 28.9 -12.7 46.8 47.5 39.0 33.3 42.0 28.9 125.8 14.6 14.9 12.7 8.3 3.8 13.1 -12.7 14.6 179.0"
li = list(covar_string.split(" "))
fin_list = [float(item) for item in li]
norm_sigma_array = preprocessing.normalize([fin_list])
norm_sigma_full = np.reshape(norm_sigma_array, (8, 8))


# actual own inputs:
n_assets = 6
lots = 3
old_pos = [1, 1, 0, 1, 0, 0]
risk_return = 0.9
T = 0

# further datamanipulation
norm_mu = list(norm_mu_full[0][:n_assets])
norm_sigma = norm_sigma_full[0:n_assets, 0:n_assets]
problem = [old_pos, risk_return, norm_sigma, norm_mu, T]

from qrisp import QuantumArray, QuantumVariable
from qrisp.qaoa import QAOAProblem
from qrisp.qaoa.mixers import portfolio_mixer
from qrisp.qaoa.problems.portfolio_rebalancing import *

# assign operators
cost_op = portfolio_cost_operator(problem=problem)
cl_cost = portfolio_cl_cost_function(problem=problem)
init_fun = portfolio_init(lots=lots)
mixer_op = portfolio_mixer()


def a_cost_fct(key):
    half = len(key[0])
    new_key = [int(key[0][i]) - int(key[1][i]) for i in range(half)]
    rr1 = sum(
        [
            risk_return * norm_sigma[i][j] * new_key[i] * new_key[j]
            for i in range(half)
            for j in range(half)
        ]
    )
    rr2 = sum([(1 - risk_return) * norm_mu[j] * new_key[j] for j in range(half)])
    c_tc = sum([T for i in range(half) if new_key[i] != old_pos[i]])
    energy = -(rr1 + rr2 + c_tc)
    return energy


def brute_force_sol():

    qv = QuantumVariable(n_assets)
    q_array = QuantumArray(qtype=qv, shape=(2))
    init_fun(q_array)
    mixer_op(q_array, np.pi)
    maxi = 0
    mini = -10000
    max_key = []
    min_key = []
    res = q_array.get_measurement()
    for item in list(res.keys()):
        if a_cost_fct(item) < maxi:
            maxi = a_cost_fct(item)
            max_key = item
        elif a_cost_fct(item) > mini:
            mini = a_cost_fct(item)
            min_key = item
    return max_key, maxi, min_key, mini


the_max, max_val, the_min, min_val = brute_force_sol()
print("brute_force")
print(the_max, max_val)
print(the_min, min_val)

approx_ratio_array = []

# for index_bm in range(50):
# assign QuantumArray to operate on
qv = QuantumVariable(n_assets)
q_array = QuantumArray(qtype=qv, shape=(2))

""" 
init_fun(q_array)
mixer_op(q_array,np.pi/2) """

# print(q_array)


# run the problem!
theproblem = QAOAProblem(
    cost_operator=cost_op, mixer=mixer_op, cl_cost_function=cl_cost
)
theproblem.set_init_function(init_fun)
theNiceQAOA = theproblem.run(q_array, depth=3)
# print the results
print(theNiceQAOA)
# print the ideal solutions
print("5 most likely Solutions")


maxfive = sorted(theNiceQAOA, key=theNiceQAOA.get, reverse=True)[:5]
for (
    key,
    val,
) in theNiceQAOA.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
    if key in maxfive:

        print((key, val))
        print(a_cost_fct(key))

for (
    key,
    val,
) in theNiceQAOA.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
    if a_cost_fct(key) <= max_val:
        print("best sol")
        print(key, val)


cost_val = cl_cost(theNiceQAOA)
approx_ratio = (max_val - cost_val) / (max_val - min_val)
print(approx_ratio)
approx_ratio_array.append(approx_ratio)


# import pickle

# with open('approx_ratios.pkl', 'wb') as f:
#    pickle.dump(approx_ratio_array, f)
