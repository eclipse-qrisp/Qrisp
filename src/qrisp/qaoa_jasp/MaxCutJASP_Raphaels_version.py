
import jax
import jax.numpy as jnp
from qrisp import jaspify, QuantumVariable, rx, sample, expectation_value, QuantumFloat
import networkx as nx
from qrisp.my_jasp_test.jax_coby import cobyla


num_nodes_out = 16
G_out = nx.erdos_renyi_graph(num_nodes_out, 0.3, seed=105)
edge_list_out = list(G_out.edges())


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



from qrisp import *

@jaspify(terminal_sampling = True)
def maina(depth=3):

    key = jax.random.key(11)
    x0 = jax.random.uniform(key= key, shape=(2*depth,))
    obj_func = maxcut_obj_JJ(G_out,depth)
    result, value = cobyla(obj_func, x0, cons=[])
    return result, value

import time
t0 = time.time()
result, value = (maina)()
print(time.time()-t0)