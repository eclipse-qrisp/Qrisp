
# %%
import jax
import jax.numpy as jnp
from qrisp import jaspify, QuantumVariable, rx, sample, expectation_value, QuantumFloat, rzz, h
import networkx as nx
from qrisp.my_jasp_test.jax_coby import cobyla


"""
Benchmarking file for the jaspified QAOA 
this is very rudimentary and pretty much is a copy past of the whole MaxCut script
sorry for that, but the missing structure makes it hard to work with

returns: result, value, final_time, callback_list

also refer the to benchmark vanilla maxcut implementation

ALSO: CHANGES IN THE COBYLA OPTIMIZER ARE NECESSARY 
these are marked with the #CHANGE comment

"""

# %%

# start with large instances if True
largo = True

postproc_timelist = []
cost_op_timelist= []
exp_val_timelist = []

print(postproc_timelist )
print(cost_op_timelist)
print(exp_val_timelist )



#%%

@jax.jit
def extract_boolean_digit(integer, digit):
    return jnp.bool((integer>>digit & 1))


# get this to work
def create_maxcut_cost_op(G,depth):
    """
    Function that applies the circuit of a MaxCut instance.
    Returns an inner function, that returns a QuantumFloat, which has been created within the inner function
    Includes the init-function, mixer and cost-operator
    
    Parameters
    ----------
    G: networkx.graph
        Graph describing the problem instance
    depth: int
        depth of the circuit

    Returns
    -------
    max_cost_opeartor: function
        The function that applies the desired circuit
    """
    edge_list = G.edges()
    num_nodes = len(G.nodes())
    def maxcut_cost_operator(gamma_beta):
        
        gamma_beta = gamma_beta.reshape((depth,2)) # where to place this??
        qv = QuantumFloat(num_nodes)
        for p in range(num_nodes):
            h(qv[p])

        for d in range(depth):

            for pair in edge_list:
                rzz(2*gamma_beta[d][0], qv[pair[0]], qv[pair[1]])
                #rzz(2*gamma_beta[0], qv[pair[0]], qv[pair[1]])

            for p in range(num_nodes):
                rx(2 * gamma_beta[d][1], qv[p])
                #rx(2 * gamma_beta[1], qv[p])

        return qv
    
    return maxcut_cost_operator


def post_processor(G):
    """
    Post processor for expecatation_value function, which inputs the QuantumFloat ``x``. For the maxcut formulation
    
    Parameters
    ----------
    G: networkx.graph
        Graph describing the problem instance

    Returns
    -------
    post_process_inner: function
        The function that applies the desired postprocessing 
    """

    edge_list = G.edges()
    def post_process_inner(x):
        cut = 0
        #print(edge_list)
        #x_bin = jnp.unpackbits(jnp.uint8(x), bitorder="little") 
        for i, j in edge_list:
            #print(i,j)

            bool1 =  extract_boolean_digit(jnp.uint16(x), i)
            bool2 =  extract_boolean_digit(jnp.uint16(x), j)
            #def predi(b1,b2, cval):
            predi = bool1 != bool2
            def true_f():
                return 1
            def false_f():
                return 0
            
            cut +=  jax.lax.cond(predi,true_f, false_f)  

        #jax.debug.print("{cut} cut", cut=cut)
        return -cut
        
    return post_process_inner

def sample_array_post_processor(sample_array, G):
    
    cut_computer = post_processor(G)
    cut_values = jax.lax.map(cut_computer, sample_array)
    average_cut = jnp.sum(cut_values)/cut_values.size
    
    return average_cut
    
sample_array_post_processor = jax.jit(sample_array_post_processor, static_argnums = 1)


def maxcut_obj_JJ(G,depth):
    """
    Function that describes the optimization routine of a MaxCut instance, based on the sample function. 
    Returns an inner function, that returns the final state of the expecation_value execution.
    
    Parameters
    ----------
    G: networkx.graph
        Graph describing the problem instance
    depth: int
        depth of the circuit

    Returns
    -------
    maxcut_obj_jitted: function
        The function that executes the desired optimization routine
    """

    def maxcut_obj_jitted(x_angles):
        
        sample_array = sample(state_prep = create_maxcut_cost_op(G,depth),
                              shots = 1000)(x_angles)
        
        return sample_array_post_processor(sample_array, G)

    return maxcut_obj_jitted


    
# Append Dict as row to DataFrame
# %%

import pandas as pd 
import pickle
df_dict_list = []

for degr in range(2,4):
    for n_nodes in reversed(range(6,23,4)):
        for depth in range(1,4):
            for fun_evs in range(50,201,150):
            
                # HAVE!! i want the average (--> final) cost, the parameters (--> can be used to reconstruct the distribution) , 
                # NEED!! average cost_list over function evals would be cool, and timeit for each optimisation loop
                
                df = pd.DataFrame({"n_nodes": [],
                         "degr": [],
                         "depth": [],
                         "fun_evals": [],
                         "res": [],
                         "val": [],
                         "final_t": [],
                         "callback_list": []})
                
                G_out = nx.random_regular_graph(degr, n_nodes, seed=105)
                

                @jaspify(terminal_sampling = True)
                def maina():

                    # qarg = QuantumVariable(G.number_of_nodes())
                    #x0 = jnp.array([0.5, 0.5,  0.1,  0.3])
                    key = jax.random.key(11)
                    x0 = jax.random.uniform(key= key, shape=(2*depth,))
                    obj_func = maxcut_obj_JJ(G_out,depth)
                    # below
                    #result, value,final_time,callback_list = cobyla(obj_func, x0, cons=[],maxfun=fun_evs,callback=True                   )
                    #return result, value, final_time, callback_list
                    result, value = cobyla(obj_func, x0, cons=[]
                                           )
                    return result, value
                
                res_, val_,final_time_,callback_list_ = maina() 
                """ res_, val_,final_time_,callback_list_ = maina(degr,n_nodes,
                                                              depth,
                                                              #fun_evs
                                                              ) """

                new_row = {"n_nodes": n_nodes,
                         "degr": degr,
                         "depth": depth,
                         "fun_evals": fun_evs,
                         "res": res_,
                         "val": val_,
                         "final_t": final_time_,
                         "callback_list": callback_list_
                         }
                
                df_dict_list.append(new_row)
                print(len(df_dict_list))
                
                if not largo:
                    filename_list = 'list'
                else:
                    filename_list = 'list_largo'
                try:
                    with open(filename_list, 'wb') as file:
                        pickle.dump(df_dict_list, file)
                    print(f"Benchmark data saved to {filename_list}")
                except Exception as e:
                    print(f"Error saving benchmark data: {e}")
                


# save into a dataframe, this will probably not make it that far

##%%
df = pd.DataFrame(df_dict_list)   
import pickle
filename = 'dataframe_bm'
try:
    with open(filename, 'wb') as file:
        pickle.dump(df, file)
    print(f"Benchmark data saved to {filename}")
except Exception as e:
    print(f"Error saving benchmark data: {e}")
 # %%

with open(r"dataframe_bm", "rb") as input_file:
    e = pickle.load(input_file)
print(e)
# %%
