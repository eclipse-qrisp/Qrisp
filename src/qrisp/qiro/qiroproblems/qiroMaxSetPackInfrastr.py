# think about just using the dict and straight forward replacement routine

# just write a replacement_routine and throw out not relevant sets

from qrisp import rz, rzz, x
import numpy as np
import copy
from qrisp.qiro.qiroproblems.qiro_utils import * 

def create_maxSetPack_replacement_routine( res, old_n_dict, sets ,universe, solutions, exclusions):

    n_dict = copy.deepcopy(old_n_dict)

    # FOR SINGLE QUBIT CORRELATIONS
    orig_nodes = list(universe)
    combinations = []

    # add check for solution nodes      
    for index1 in range(len(universe)):
        for index2 in range(index1,len(universe)):
            combinations.append((index1,index2))

    # you are HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    max = 0
    max_item = []
    #get the max_edge and eval the sum and sign
    """ for item2 in combinations:
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

    # we just directly remove vertices from the graph 
    if isinstance(max_item, int):
        if sign > 0:
            the_set = sets[max_item]
            for node in the_set:
                rels = n_dict[node]
                for rel in rels:
                    sets.pop(rels) 
            # add set to sol
            # go through n_dict and remove all other sets that contain elements in this set
            # what does remove mean in this context
            solutions.append(max_item)
            exclusions.append(max_item)

        elif sign < 0:
            # remove set from sets?
            sets.pop(max_item) 
            exclusions.append(max_item)

    else:
        if sign > 0:
            # gotta think of something here

            pass

        elif sign < 0:
            # remove all sets that contain elements of the schnittmenge, because one of the two is going to be in there
            # how to fix the negative correlation??
            intersect = list(sets[max_item[0]] & sets[max_item[1]])
            for item in intersect:
                rels = n_dict[item]
                for rel in rels: 
                    if not rel in max_item:
                        sets.pop(rel)
            exclusions.append(rel)
            


    return sets, solutions, sign, exclusions

# what about other operatores??