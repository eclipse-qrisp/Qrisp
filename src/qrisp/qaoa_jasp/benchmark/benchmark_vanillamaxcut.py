

#%%

from qrisp.algorithms.qaoa.qaoa_problem import QAOAProblem
from qrisp.algorithms.qaoa.problems.maxCut import *
from qrisp.algorithms.qaoa.mixers import *
from qrisp import QuantumVariable

import networkx as nx
import pandas as pd 
import pickle

"""
Benchmarking file for the vanilla QAOA
this unfortunately also requires changes in the original QAOA_problem optimization loop, to return the final time

if this is not desired, just set final_time == [] in the loop
callback_list is available in original QAOA via the keyword argument in the problem instanciation, so no changes are necessary


"""

# %%

largo = True

ind = 0
df_dict_list = []
for degr in range(2,4):
    for n_nodes in reversed(range(6,23,4)):
        for depth in range(1,4):
            for fun_evs in range(50,201,150):
            
                # HAVE!! i want the average (--> final) cost, the parameters (--> can be used to reconstruct the distribution) , 
                # NEED!! average cost_list over function evals would be cool, and timeit for each optimisation loop
                print(ind)
                df = pd.DataFrame({"n_nodes": [],
                         "degr": [],
                         "depth": [],
                         "fun_evals": [],
                         "res": [],
                         "val": [],
                         "final_t": [],
                         "callback_list": []})
                
                G_out = nx.random_regular_graph(degr, n_nodes, seed=105)
                
                qarg = QuantumVariable(n_nodes)

                maxcut_inst =QAOAProblem(cost_operator=create_maxcut_cost_operator(G_out),mixer=RX_mixer,
                                         cl_cost_function=create_maxcut_cl_cost_function(G_out)  )
                res_, val_,final_time_,callback_list_ = maxcut_inst.run(
                                                                        qarg, 
                                                                        depth,
                                                                        max_iter = fun_evs
                )

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
                #qarg.uncompute()
                qarg.delete()
                ind += 1
                if not largo:

                    filename_list = 'vanilla_list'
                else:
                    filename_list = 'vanilla_list_largo'
                try:
                    with open(filename_list, 'wb') as file:
                        pickle.dump(df_dict_list, file)
                    print(f"Benchmark data saved to {filename_list}")
                except Exception as e:
                    print(f"Error saving benchmark data: {e}")

#%%

import pickle
filename_list_bm = 'vanilla_list_bm'
try:
    with open(filename_list_bm, 'wb') as file:
        pickle.dump(df_dict_list, file)
    print(f"Benchmark data saved to {filename_list_bm}")
except Exception as e:
    print(f"Error saving benchmark data: {e}")



##%%
df = pd.DataFrame(df_dict_list)   
import pickle
filename = 'vanilla_dataframe_bm'
try:
    with open(filename, 'wb') as file:
        pickle.dump(df, file)
    print(f"Benchmark data saved to {filename}")
except Exception as e:
    print(f"Error saving benchmark data: {e}")
# %%

with open(r"vanilla_dataframe_bm", "rb") as input_file:
    e = pickle.load(input_file)
print(e)
# %%
