# %%
import jax
import jax.numpy as jnp
import networkx as nx
from qrisp.qaoa_jasp.jax_cobyla import cobyla
from qrisp.jasp import jaspify
from qrisp import QuantumFloat, QuantumVariable, rz, rzz, h, rx, expectation_value
from itertools import combinations

Depth = 1

num_nodes_out = 8
G_outer= nx.erdos_renyi_graph(num_nodes_out, 0.7, seed=123)
edges_outer = jnp.array(list(G_outer.edges()))

@jax.jit
def extract_boolean_digit(integer, digit):
    return jnp.bool((integer>>digit & 1))

"""
TWO THINGS HERE:
--> jax.lax.dynamic_slice - SLICING DOESNT WORK with non concrete slicing borders  
-And otherwise, if I just return the full true_indices: with [3 000000] i somehow get a non fitting shape error?? 
- which is also the the case with the list comprehension forumlation that displayed in the end

FOR FURTHER DOCUMENTATION please refer to the MaxCut-instance
"""

 # %%

#@jax.jit
def true_index_combinations(arr,size):
    #jax.debug.print("{arr1sh} arr1.shape", arr1sh=arr.shape)
    #jax.debug.print("{arr1} arr1", arr1=arr)

    # fill_value = 99
    # atm fill_value = 0
    true_indices = jnp.where(arr,size=size)[0]
    return jnp.array(list(combinations(true_indices, 2)))




def all_subarrays_found(arr1, arr2):
    """ jax.debug.print("subarrix_combs")
    jax.debug.print("{arr1sh} arr1.shape", arr1sh=arr1.shape)
    jax.debug.print("{arr1} arr1", arr1=arr1)
    jax.debug.print("{arr2} arr2", arr2=arr2)
    jax.debug.print("{arr2sh} arr222.pe", arr2sh=arr2.shape) """
    
    return jnp.all(jnp.any(jnp.all(arr2[:, None] == arr1, axis=2), axis=0))



# %%

def max_clique_post_process(G):
    edges = jnp.array(list(G.edges()))
    n_nodes = len(G.nodes())
    def maxclique_post_process_inner(result):
        #create bool_arr with indexes of the ones in the solution
        bool_arr= jnp.array([extract_boolean_digit(jnp.uint8(result),i) for i in range(n_nodes)]) 
        sum_bool = jnp.sum(bool_arr) 
        sum_bool_int = sum_bool.astype(int)
        # subroutine to check if it is valid
        # i.e. check if all combinations of ones are found in the original edges
        ix_combs = true_index_combinations(bool_arr, size=8) # all combinations !!! atm also contains filler vals ( trailing zeros)
        jax.debug.print("{ix_combs} ix_combs", ix_combs=ix_combs)
        # check if the ix_combs has more than one entry 
        # (atm it always does have atleast one, atleast the fill value combination)
        pred_valid = jnp.bool(sum_bool_int  > 1 )
        def true_f(vals):
            ix_combs_in, edges_in = vals
            # check if all combs are found in orig edges
            return all_subarrays_found(ix_combs_in,edges_in)
        def false_f(vals):
            # if len-ix_combs is less than 1, it is a solution
            return True
        init_vals = (ix_combs, edges)
        valid_bool = jax.lax.cond(pred_valid, true_f,false_f, init_vals) 
        
        # if valid, return: sum of Trues (sum_bool_int)
        pred = valid_bool
        def tr_f(x_in):
            return x_in
        def fa_f(x_in): 
            # here we have to do the additional check:
            # if only one True, it is a solution, but has no edges, so the edge_check will fail
            # --> manually return 1
            pred_isone = jnp.bool(x_in == 1)
            def tf_one():
                return 1
            def ff_one():
                return 0
            return jax.lax.cond(pred_isone, tf_one,ff_one) 

        final_cost = jax.lax.cond(pred, tr_f, fa_f, sum_bool_int)

        return -final_cost
    
    return maxclique_post_process_inner


def create_maxclique_jax_cost_op(G, depth):
    G_compl = nx.complement(G)
    edge_list = G_compl.edges()
    num_nodes = len(G.nodes())

    def maxclique_jax_cost_op(gamma_beta):

        #gamma_beta = gamma_beta.reshape((depth,2)) # where to place this??
        qv = QuantumFloat(num_nodes)

        for pa in range(num_nodes):
            h(qv[pa])

        for pa in range(num_nodes):
            rx(2 * gamma_beta[0], qv[pa])

        for pair in edge_list:
            rzz(3*gamma_beta[1], qv[pair[0]], qv[pair[1]])
            rz(-gamma_beta[1], qv[pair[0]])
            rz(-gamma_beta[1], qv[pair[1]])


        return qv
    
    return maxclique_jax_cost_op



def maxclique_obj_JJ(G,depth):

    def maxclique_obj_inner(x_angles):
    
        res = expectation_value(
            #state_prep = apply_rx, 
            state_prep = create_maxclique_jax_cost_op(G,depth),
            #state_prep = mco,
            shots = 1000, 
            post_processor=max_clique_post_process(G))(x_angles)
        return res

    return maxclique_obj_inner


# %% 
@jaspify(terminal_sampling = True)
def main():

    key = jax.random.key(11)
    x0 = jax.random.uniform(key= key, shape=(2*Depth,))
    obj_func = maxclique_obj_JJ(G_outer, Depth)
    result, value = cobyla(obj_func, x0, cons=[],maxfun=100) 
    return result, value
    
result, value= main()

print("Optimal solution:", result)
print("Optimal value:", value)
 

# %% 


#Checks for correct functionality of cost function
""" import itertools


def cl_cost_function(int_state):
    state = bin(int_state)[2:].zfill(num_nodes_out)[::-1]
    
    cost = 0
    temp = True
    indices = [index for index, value in enumerate(state) if value == '1']
    combinations = list(itertools.combinations(indices, 2))
    for combination in combinations:
        if combination not in G_outer.edges():
            temp = False
            break
    if temp: 
        cost += -len(indices)
    return cost


@jaspify(terminal_sampling = True)
def main():
    tstfun = max_clique_post_process(G_outer)

    #cr = tstfun(x_here)
    #cw = tstfun(x_wrong)
    max_int = 2**num_nodes_out
    for index in range(max_int):
        #qf = QuantumFloat(num_nodes_out)
        #qf[:] = index
        jax_cost = tstfun(index)
        cl_cost = cl_cost_function(index)
        if not jax_cost == cl_cost:
            print("ERR")
            print(index)
            state1 = bin(index)[2:].zfill(num_nodes_out)[::-1]
            print(state1)
            print(jax_cost)
            print(cl_cost)
            print("    ")
main()
 """
#%%
# Check for correct functionality, i.e. compare to vanilla qaoa

"""

def QAOA_to_apply(G,qv_in, res_FINAL,depth ):
    num_nodes = len(G.nodes())
    edge_list = list(G.edges())
    gamma_beta = res_FINAL.reshape((depth,2))
    for pa in range(num_nodes):
        h(qv_in[pa])

    for dep in range(depth):
        
        for pa2 in range(num_nodes):
            rx(2 * gamma_beta[dep][0], qv_in[pa2])
        for pair in edge_list:
            rzz(3*gamma_beta[dep][1], qv_in[pair[0]], qv_in[pair[1]])
            rz(-gamma_beta[dep][1], qv_in[pair[0]])
            rz(-gamma_beta[dep][1], qv_in[pair[1]])
    
#NEED TO DO COMPLEMENT!!!
test_qv = QuantumFloat(num_nodes_out)
QAOA_to_apply(G_outer, test_qv,result, Depth)


test_qv_BIN = QuantumVariable(num_nodes_out)
QAOA_to_apply(G_outer, test_qv_BIN,result, Depth)

BIN_TEsT = test_qv_BIN.get_measurement()
sort_dict_BIN = dict(sorted(BIN_TEsT.items(), key=lambda item: item[1],reverse=True))
print(sort_dict_BIN)

restest = test_qv.get_measurement()


cost_f = max_clique_post_process(G_outer)
#print( "58 " + str(cost_f(58)))

sort_dict = dict(sorted(restest.items(), key=lambda item: item[1],reverse=True))
final_keys = list(sort_dict.keys())
for i in range(5): 
    print(final_keys[i] , cost_f(final_keys[i]))


print(G_outer.edges())
nx_part = nx.approximation.max_clique(G_outer)
print("nx clique " + str(nx_part))

"""






# %%

# Alternative cost_funtion formulaion
"""

true_indices = jnp.array(0)
#init_arrays = (jnp.array(true_indices[0]), true_indices,value, 0 )
# write for_i_loop
#true_indices = true_indices[true_indices != value]
liste = []
for i1 in range(len(true_indices)):
    for i2 in range(0,i1):
        predi = true_indices[i1] & true_indices[i2]
        def true_f_bf(vals):
            lisss, ind1, ind2 = vals
            lisss.append((ind2,ind1))
            return lisss,ind1, ind2
        def false_f_bf(vals):
            lisss, ind1, ind2 = vals
            lisss.append((0,0))
            return lisss,ind1, ind2
        
        liste_end, indend1, indend2 = jax.lax.cond(predi,true_f_bf,false_f_bf,  liste,i1,i2)
"""
#%%

# Conditional function used for:
# In inital formulation the true_index_comb function returns an array of a given length (n_nodes)
# --> has skip_vals as fillers in there, which we dont want to consider, need a conditional check for this 
""" 
def body_f(i, vals):
    temp_array, true_ix, skip_val,ix = vals
    pred_bf = jnp.bool(skip_val != true_ix[ix])
    def true_f_bf(cond_vals):
        some_arr = jnp.append(cond_vals[0], cond_vals[1][cond_vals[-1]])
        return some_arr, cond_vals[1], cond_vals[2], cond_vals[3]
    
    def false_f_bf(cond_vals):
        return cond_vals[0], cond_vals[1], cond_vals[2], cond_vals[3]
    cond_init = (temp_array, true_ix, skip_val,ix)
    temp_array_a, true_ix_a, skip_val_a,ix_a = jax.lax.cond(pred_bf, true_f_bf,false_f_bf, 
                                                    cond_init)
    ix_a +=1
    return temp_array_a, true_ix_a, skip_val_a,ix_a
final_array = jax.lax.fori_loop(1,size, body_f,init_arrays) """