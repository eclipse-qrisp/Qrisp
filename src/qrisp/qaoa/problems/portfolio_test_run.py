import numpy as np

lots = 2

A = [45, 37, 42, 35, 39]
B = [38, 31, 26, 28, 33]
C = [10, 15, 17, 21, 12]
D = [11,14,14,20,33]
data = np.array([A, B, C,D])

old_pos = [1, 1, 0, 0]
risk_return = 0.5
covar_matrix = np.cov(data, bias=True)
asset_return = [1.2, 0.8, 1.3, 1.1]
tc = 0.1
problem = [old_pos,risk_return,covar_matrix,asset_return, tc]

from qrisp.qaoa.build_dicke_mixer import * 
from qrisp.qaoa.buildportfolio_costop import * 
from qrisp.qaoa.buildportfolio_init import * 
from qrisp.qaoa import QAOAProblem
cost_op = portfolio_cost_operator(problem=problem)

cl_cost = portfolio_cl_cost_function(problem=problem)
init_fun = portfolio_init(lots=lots)
mixer_op = portfolio_mixer()

qv = QuantumVariable(4)
q_array = QuantumArray(qtype=qv, shape=(2))

        
theproblem = QAOAProblem(cost_operator=cost_op, mixer=mixer_op, cl_cost_function=cl_cost)
theproblem.set_init_function(init_fun)

theNiceQAOA = theproblem.run(q_array,depth = 3)

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
for name, age in theNiceQAOA.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
    if name in maxfive:

        print((name, age))
        print(a_cost_fct(name))


print(q_array.qs)

"""

init_fun(q_array)
#print(qv.qs)
inital_state = q_array.duplicate(init = True)
#print(q_array)
#print(inital_state)
half = int(len(q_array[0])/2)
# the code below does what its supposed to
#dicke_state(q_array[0], half)
#dicke_state(q_array[1], half)  

# this one doesnt
mixer_op(q_array,np.pi/4)
#print(q_array)

#initial_vs_mixed = multi_measurement([inital_state, q_array]) 
#print(initial_vs_mixed)
"""


""" 

print(len(initial_vs_mixed))
count = 0 
for k,v in initial_vs_mixed.items():
    if v > 0.00001:
        count +=1

print(count)
 """

""" cost_op(q_array,np.pi/4)
res = q_array.get_measurement()


old_dict = dict()
for k,v in res.items():
    key_old = str(k[0] + k[1])
    old_dict.setdefault(key_old, v)

old_cost= portfolio_cl_cost_function_for_qv(problem=problem)

res_old, key_list_old = old_cost(old_dict)
res, key_list = cl_cost(res)
print(key_list_old)
print("new")
print(key_list)
print(len(key_list_old))

print(len(key_list))
print(res)
print(res_old) """

