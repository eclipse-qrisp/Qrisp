# just write a replacement_routine and throw out not relevant sets

from qrisp import rz, rzz, x
import numpy as np
import copy
from qrisp.qiro.qiroproblems.qiro_utils import * 

#Done??
def create_maxSat_replacement_routine( res, oldclauses, numVars, solutions, exclusions):

    clauses = copy.deepcopy(oldclauses)

    # FOR SINGLE QUBIT CORRELATIONS
    orig_nodes = list(range(1,numVars + 1))
    combinations = []

    # add check for solution nodes      
    for index1 in range(1,numVars):
        for index2 in range(index1,numVars + 1):
            combinations.append((index1,index2))

    """ max = 0
    max_item = []
    #get the max_edge and eval the sum and sign
    for item2 in combinations:
        #if abs(item2[0]) == abs(item2[1]):
            #continue
        summe = 0

        # calc correlation expectation
        for key, val in res.items():
            summe += pow(val ,2) * pow(-1,int(key[int(abs(item2[0]))])) * pow(-1, int(key[int(abs(item2[1]))]))
            

        #find max
        if abs(summe) > abs(max):
            max, max_item = summe, item2
            sign = np.sign(summe)


    for node in orig_nodes:
        if node in solutions:
            continue

        for key, val in res.items():
            summe += val * pow(-1,int(key[int(node)])) 
            #maybe have to have this guy in here??
            #if index > 1:
                #summe += float(val * pow(-1,int(key[int(node)])) )
        #find max
        if abs(summe) > abs(max):
            max, max_item = summe, node
            sign = np.sign(summe) """


    max_item, sign = find_max(orig_nodes, combinations, res, solutions)

    # we just directly remove clauses from clauses
    if isinstance(max_item, int):
        if sign > 0:
            for item in clauses: 
                # if negation in item --> remove it
                if -1 * max_item in item:
                    clauses.remove(item)
            solutions.append(max_item)
            exclusions.append(max_item)

        elif sign < 0:
            for item in clauses: 
                if max_item in item:
                    clauses.remove(item)
            exclusions.append(max_item)

    else:
        if sign > 0:
            for item in clauses: 
                if max_item[1] in item:
                    temp = item.index[max_item[1]]
                    item[temp] = max_item[0]
                if -1* max_item[1] in item:
                    temp = item.index[-1* max_item[1]]
                    item[temp] = -1* max_item[0]
                exclusions.append(max_item[1])
                    
        # what here??
        elif sign < 0:
            for item in clauses: 
                if max_item[1] in item:
                    temp = item.index[max_item[1]]
                    item[temp] = -1 * max_item[0]
                if -1* max_item[1] in item:
                    temp = item.index[ -1  * max_item[1]]
                    item[temp] = max_item[0]
                exclusions.append(max_item[1])

    # create sign list somewhere?
    return clauses, solutions, sign, exclusions


#Done?
def create_maxSat_mixer_reduced(clauses, solutions, exclusions):

    def RX_mixer(qv, beta):

        from qrisp import rx
        for i in range(qv.size):
            if not i in exclusions:
                rx(2 * beta, qv[i])
        return qv
    
    return RX_mixer


# placeholder so i dont forget
# does not need changing right? since we already adjust the clauses directly 
def create_maxIndep_cost_operator_reduced(Graph, solutions):
    pass


def init_function_reduced(solutions, exclusions):

    def init_state(qv):
        from qrisp import h
        for i in range(qv.size):
            if not i in exclusions:
                h(qv[i])
        for i in solutions:
            x(qv[i])
    return init_state

