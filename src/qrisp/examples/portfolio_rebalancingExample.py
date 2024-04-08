import numpy as np
from qrisp import QuantumArray, QuantumVariable

# assign problem definitions
n_assets = 4
# lots
lots = 2
# covariance between assets
A = [45, 37, 42, 35, 39]
B = [38, 31, 26, 28, 33]
C = [10, 15, 17, 21, 12]
D = [11,14,14,20,33]
data = np.array([A, B, C,D])
covar_matrix = np.cov(data, bias=True)
# old positions
old_pos = [1, 1, 0, 0]
#risk-return factor
risk_return = 0.5
# normalized asset returns
asset_return = [1.2, 0.8, 1.3, 1.1]
# trading costs
tc = 0.1
# full problem instance
problem = [old_pos,risk_return,covar_matrix,asset_return, tc]

from qrisp.qaoa.mixers import portfolio_mixer 
from qrisp.qaoa.problems.portfolio_rebalancing import * 
from qrisp.qaoa import QAOAProblem

# assign operators
cost_op = portfolio_cost_operator(problem=problem)
cl_cost = portfolio_cl_cost_function(problem=problem)
init_fun = portfolio_init(lots=lots)
mixer_op = portfolio_mixer()

# assign QuantumArray to operate on
qv = QuantumVariable(n_assets)
q_array = QuantumArray(qtype=qv, shape=(2))

# run the problem!
theproblem = QAOAProblem(cost_operator=cost_op, mixer=mixer_op, cl_cost_function=cl_cost)
theproblem.set_init_function(init_fun)
theNiceQAOA = theproblem.run(q_array,depth = 3)
# print the results
print(theNiceQAOA)
#print the ideal solutions
print("5 most likely Solutions") 

def a_cost_fct(key):
    half = len(key[0])
    new_key = [int(key[0][i])-int(key[1][i]) for i in range(half)]
    rr1 = sum([risk_return*covar_matrix[i][j] *new_key[i]*new_key[j] for i in range(half) for j in range(half)])
    rr2 = sum([(1-risk_return)*asset_return[j] *new_key[j] for j in range(half)])
    c_tc= sum([tc  for i in range(half) if new_key[i] != old_pos[i]])
    energy = -(rr1+ rr2+ c_tc)
    return energy

maxfive = sorted(theNiceQAOA, key=theNiceQAOA.get, reverse=True)[:5]
for key, val in theNiceQAOA.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
    if key in maxfive:

        print((key, val))
        print(a_cost_fct(key))


#print(q_array.qs)
