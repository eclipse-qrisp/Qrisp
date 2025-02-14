# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 13:54:59 2025

@author: sea
"""

import jax
import jax.numpy as jnp


def nelder_mead(f, x0, max_iter=1000, alpha=1, gamma=2, rho=0.5, sigma=0.5):
    n = len(x0)
    simplex = jnp.vstack([x0, jnp.array([x0 + jnp.eye(n)[i] for i in range(n)])])
    
    @jax.jit
    def step(simplex):
        
        f_values = jax.lax.map(f, simplex)
        
        ordered_indices = jnp.argsort(f_values)
        best, second_worst, worst = ordered_indices[0], ordered_indices[-2], ordered_indices[-1]
        
        centroid = jnp.mean(simplex[ordered_indices[:-1]], axis=0)
        reflected = centroid + alpha * (centroid - simplex[worst])
        f_reflected = f(reflected)
        
        def reflection_case(simplex):
            return jax.lax.cond(
                f_reflected < f_values[best],
                lambda _: expansion_case(simplex),
                lambda _: simplex.at[worst].set(reflected),
                None
            )
        
        def expansion_case(simplex):
            expanded = centroid + gamma * (reflected - centroid)
            return jax.lax.cond(
                f(expanded) < f_reflected,
                lambda _: simplex.at[worst].set(expanded),
                lambda _: simplex.at[worst].set(reflected),
                None
            )
        
        def contraction_case(simplex):
            contracted = centroid + rho * (simplex[worst] - centroid)
            return jax.lax.cond(
                f(contracted) < f_values[worst],
                lambda _: simplex.at[worst].set(contracted),
                lambda _: shrink_case(simplex),
                None
            )
        
        def shrink_case(simplex):
            return jnp.where(jnp.arange(n+1)[:, None] != best,
                             simplex[best] + sigma * (simplex - simplex[best]),
                             simplex)
        
        temp = f_values[best] <= f_reflected
        temp = temp & (f_reflected < f_values[second_worst])
        return jax.lax.cond(
            temp,
            lambda _: simplex.at[worst].set(reflected),
            lambda _: jax.lax.cond(
                f_reflected < f_values[best],
                reflection_case,
                contraction_case,
                simplex
            ),
            None
        )

    simplex = jax.lax.fori_loop(0, max_iter, lambda _, s: step(s), simplex)
    return simplex[jnp.argmin(jax.lax.map(f, simplex))]

def objective(x):
    return jnp.abs(x[0]**2 + x[1]**2 - 1337)


@jaspify
def main():
    x0 = jnp.array([1.0, 1.0])
    result = nelder_mead(objective, x0)
    return result

print(sum(main()**2))

#%%
@jaspify
def main2():
    
    x0 = jnp.array([1.0, 1.0])
    y0 = jnp.array([1.0, 1.0])
    
    z = jnp.vstack([x0, y0])
    
    return jax.lax.map(objective, z)
# jaxpr = make_jaxpr(main2)()

# print(jaxpr)
print(main())


#%%

from qrisp import *
from qrisp.jasp import *


def state_prep(k):
    a = QuantumFloat(4)
    b = QuantumFloat(4)
    
    qbl = QuantumBool()
    h(qbl)
    
    with control(qbl[0]):
        a[:] = k
        
    cx(a, b)
    
    return a, b

@jaspify
def main(k):
    
    ev_function = expectation_value(state_prep, shots = 50)
    
    return ev_function(k)

print(main(3))

#%%

from qrisp import *
from qrisp.jasp import *

def inner_f(i):
    qf = QuantumFloat(4)
    
    with conjugate(h)(qf):
        for k in jrange(i):
            t(qf[0])
    
    h(qf)
    qf -= 1
            
    return measure(qf) + 1


def sample(func, amount):
    
    def return_function(*args, **kwargs):
    
        @quantum_kernel
        def kernel(argtuple, dummy_arg):
            return argtuple, func(*argtuple, **kwargs)
    
        return jax.lax.scan(kernel, args, length = amount)[1]
    
    return return_function

def energy(x):
    return 2*x + 1

@jaspify
def main():
    
    res = sample(inner_f, 10)(2)
    # res = jax.lax.scan(inner_f, (), length = 5)
    # res = jax.vmap(inner_f, axis_size = 5)
    # return jax.vmap(energy)(res)
    
    return res

# jaxpr = make_jaxpr(main)()
print(main())
# print(jaxpr)

#%%


import networkx as nx

G = nx.erdos_renyi_graph(10, 0.2)

#%%
from qrisp import *
from qrisp.jasp import *

def phase_sep(phi, qv):
    
    for i, j in G.edges():
        cp(phi, qv[i], qv[j])
        
@qache
def qaoa_kernel(parameters):
    
    qv = QuantumFloat(len(G))
    
    h(qv)
    
    for i in jrange(len(parameters)//2):
        phase_sep(parameters[2*i], qv)
        rx(parameters[2*i+1], qv)
        
    return qv, QuantumFloat(2), QuantumBool()

# def double(x, y):
    # return x, y

# def qaoa_kernel(parameters):
    
#     a = QuantumFloat(2)
#     b = QuantumFloat(2)
#     c = QuantumFloat(3)
#     x(a)
#     x(b)
#     return a, b, c

def double(x, y, z):
    return jnp.int32(x)>>8, z

@jaspify(terminal_sampling = True)

# @make_jaspr
def main(i):
    res = 0
    for i in range(200):
        print(i)
        res+= expectation_value(qaoa_kernel, 100, post_processor = double)(jnp.array(range(12)))
    return res 
# main(10)
pass
print(main(10))

#%%

from qrisp.jasp import * 
from qrisp import* 

#https://quantum-journal.org/papers/q-2024-12-09-1552/pdf/
# should this one be done with qbl instead
@jaspify
def parallel_GHZ(n):
    # n=3
    qv = QuantumFloat(n)
    h(qv)
    anc = QuantumFloat(n-1)
    for ind in jrange(n-1):
        cx(qv[ind], anc[ind])
        cx(qv[ind+1], anc[ind])

    def parity(anc):
        return measure(anc[0])
        res = measure(anc)
        
        ctrl_bl = res %2 == 0
        return ctrl_bl
    
    with control(parity(anc)):
        x(qv)

    res =measure(qv)
    return res

# jaspr = parallel_GHZ(3)
print(parallel_GHZ(3))
# print(jaspr.to_qc(4)[0].to_qasm3())
#%%
from qrisp import *
import numpy as np

def U(qf):
    z(qf[1])
qv = QuantumFloat(3)
qv[:] = 1
# h(qv)

# U(qv)

print(qv.qs.statevector("array"))

#%%

def inner_f(i):
    qf = QuantumFloat(4)
    
    with conjugate(h)(qf):
        for k in jrange(i):
            t(qf[0])
            
    return qf

@jaspify
def main():
    res = sample(inner_f, 100)(2)
    return res

print(main())

#%%
from qrisp import *
from qrisp.jasp import *

def double(*args):
    if len(args) == 1:
        return 2*args[0]
    return tuple([2*x for x in args])

def inner_f(i):
    qf = QuantumFloat(4)
    
    with conjugate(h)(qf):
        for k in jrange(i):
            t(qf[0])
            
    return qf

@terminal_
def main():
    res = expectation_value(inner_f, 1000000, post_processor=double)(2)
    return res

print(main())
# assert abs(main()-1) < 0.05

#%%
@quantum_kernel
def inner():
    
    qf_a = QuantumFloat(4)
    qf_b = QuantumFloat(4)
    
    qf_a[:] = 5
    qf_b[:] = 3
    
    qf_a -= qf_b
    return measure(qf_a)

@boolean_simulation
def main():
    return inner()

print(main())

#%%

import numpy as np
import jax.numpy as jnp
from qrisp import *
from qrisp.jasp import make_jaspr

def diffuser(qv_list):

    qv1 = qv_list[0]
    qv2 = qv_list[1]
    h(qv1)
    h(qv2)

    m = len(qv_list)

    # tag {qv : 0, qv2 : 0} state ?
    temp_qf = QuantumFloat(m)
    for i in range(m):
        mcx(qv_list[i], 
            temp_qf[i], 
            method = "balauca", 
            ctrl_state = 0)
    
    h(temp_qf[temp_qf.size-1])
    mcx(temp_qf[:temp_qf.size-1], 
        temp_qf[temp_qf.size-1],
        method= "balauca",
        ctrl_state=-1)
    h(temp_qf[temp_qf.size-1])
    
    for i in range(m):
        mcx(qv_list[i], 
            temp_qf[i], 
            method = "balauca", 
            ctrl_state = 0)
        
    temp_qf.delete()

    h(qv1)
    h(qv2)


@terminal_sampling
def test_fun(i):

    qv1 = QuantumFloat(i)
    qv2 = QuantumFloat(i)
    qv_list = [qv1, qv2]

    diffuser(qv_list)

    return qv1, qv2

print(type(list(test_fun(1).values())[0]))
#%%
from qrisp import *
from qrisp.jasp import *
@terminal_sampling(shots = 50)
def main(phi, i):
    
    qv = QuantumFloat(i)

    x(qv[:qv.size-1])
    
    qbl = QuantumBool()
    h(qbl)
    
    with control(qbl[0]):
        with conjugate(h)(qv[qv.size-1]):
            mcp(phi, qv)
        
    return qv

print(main(np.pi, 5))

#%%

from qrisp import *
from qrisp.jasp import *
def state_prep(k):
    a = QuantumFloat(4)
    b = QuantumFloat(4)
    
    qbl = QuantumBool()
    h(qbl)
    
    with control(qbl[0]):
        a[:] = k
        
    cx(a, b)
    
    return a, b

def post_processor(x, y):
    return x*y

@jaspify
def main(k):
    
    ev_function = expectation_value(state_prep, shots = 1000, post_processor = post_processor)
    
    return ev_function(k)

print(main(3))

#%%

from qrisp import *
from qrisp.jasp import *

@stimulate
def main():
    qv = QuantumFloat(4)
    x(qv[0])
    cx(qv[0], qv[1])
    return measure(qv)
    
print(main())

#%%

from qrisp import *
from qrisp.jasp import *

@jaspify
def main(k):
    
    qf = QuantumFloat(6)
    
    def body_fun(i, val):
        acc, qf = val
        x(qf[i])
        acc += measure(qf[i])
        return acc, qf
    
    acc, qf = q_fori_loop(0, k, body_fun, (0, qf))
    
    return acc, measure(qf)

print(main(6))

@jaspify
def main(k):
    
    qf = QuantumFloat(6)
    
    def body_fun(i, qf):
        x(qf[i])
        return qf
    
    qf = q_fori_loop(0, k, body_fun, qf)
    
    return measure(qf)

print(main(6))


from qrisp import *
from qrisp.jasp import *

@jaspify
def main(k):
    
    qf = QuantumFloat(6)
    
    def body_fun(val):
        i, acc, qf = val
        x(qf[i])
        acc += measure(qf[i])
        i += 1
        return i, acc, qf
    
    def cond_fun(val):
        return val[0] < 5
    
    i, acc, qf = q_while_loop(cond_fun, body_fun, (0, 0, qf))
    
    return acc, measure(qf)

print(main(6))

@jaspify
def main(k):
    
    qf = QuantumFloat(6)
    
    def body_fun(val):
        i, qf = val
        x(qf[i])
        i += 1
        return (i, qf)
    
    def cond_fun(val):
        return val[0] < 5
    
    i, qf = q_while_loop(cond_fun, body_fun, (0, qf))
    
    return measure(qf)

print(main(6))


#%%
from qrisp import *
from qrisp.jasp import *

@jaspify
def main(k):
    
    def false_fun(qbl):
        qbl.flip()
        return qbl
    
    def true_fun(qbl):
        return qbl
    
    qbl = QuantumBool()
    h(qbl)
    pred = measure(qbl)
    
    qbl = q_cond(pred, 
                true_fun, 
                false_fun, 
                qbl)
    
    return measure(qbl)

print(main(5))

#%%

def uint_qq_less_than(a, b, inv_adder):
    comparison_anc = QuantumBool()
    comparison_res = QuantumBool()
    
    if a.size < b.size:
        temp_var = QuantumVariable(b.size-a.size)
        a = list(a) + list(temp_var)
    
    with conjugate(inv_adder, allocation_management = False)(b, list(a) + [comparison_anc]):
        cx(comparison_anc, comparison_res)
    
    comparison_anc.delete()
    
    try:
        temp_var.delete()
    except UnboundLocalError:
        pass
    
    return comparison_res

#%%
from qrisp import *
from qrisp.jasp import *

@jaspify
def main():
    
    qv = QuantumFloat(4)
    
    QFT(qv, exec_swap = False, inv = True)
    return measure(qv)

print(main())

#%%

from qrisp import *
from qrisp.jasp import *

@RUS
def test(n):

    qf = QuantumFloat(n)
    h(qf)

    qbl = QuantumBool()
    case_indicator = QuantumFloat(n)
    control_qbl = QuantumBool()

    def state_preparation(qv):
        h(qv)
    
    with conjugate(state_preparation)(case_indicator):
        for i in jrange(n):
            mcx(case_indicator[:i+1], 
                control_qbl[0], 
                method = "balauca", 
                ctrl_state = 2**i-1)
        
            mcx([control_qbl[0],qf[n-i-1]],
                qbl[0])
           
            mcx(case_indicator[:i+1], 
                control_qbl[0], 
                method = "balauca", 
                ctrl_state = 2**i-1)
            
    control_qbl.delete()
    
    return (measure(case_indicator) == 0) & (measure(qbl) == 1), qf


@terminal_sampling
# @jaspify
def main():
    # qf = QuantumFloat(2)
    # h(qf)
    
    qf =test(10)
    return qf
    return measure(qf)

res_dict = main()
# res_dict = {0: 0, 1: 0, 2: 0, 3: 0}
# shots = 100

# for i in range(shots):
    # res_dict[float(main())] +=1

# for k, v in res_dict.items():
    # res_dict[k] = v/shots

print(res_dict)

for k, v in res_dict.items():
    res_dict[k] = v**0.5

one_value = res_dict[1]
for k, v in res_dict.items():
    res_dict[k] = res_dict[k]/res_dict[1]

print(res_dict)

#%%
from qrisp import *

M = 30
@jaspify
def main():
    
    N = 9
    qv = QuantumFloat(N)
    x(qv[:N-1])
    # x(qv[k-1])
    for i in range(M):
        jasp_balauca_mcx(qv[:qv.size-1], [qv[qv.size-1]], ctrl_state = 2**(N-1) -1)
    
    
    return measure(qv)
    balauca_anc = QuantumVariable(N+N%2-1)
    ctrls = qv[:N]
        
    with QuantumEnvironment():
        
        for i in jrange(N//2):
            mcx([ctrls[2*i], ctrls[2*i+1]], balauca_anc[i])
            
        with control(N%2 != 0):
            cx(ctrls[N-1], balauca_anc[N//2-1+N%2])
        
        N_int = N
        N = make_tracer(N)
        n = jnp.int64(jnp.ceil(jnp.log2(N)))
        
        l = make_tracer(0)
        k = N
        from qrisp import barrier
        l = N//2+N%2
        for i in jrange(balauca_anc.size-l):
            mcx([balauca_anc[2*i], balauca_anc[2*i+1]], 
                balauca_anc[l+i],
                method = "gray")
            
        #     pass
        
        # for i in jrange(n-2):
            # k = jnp.int64(jnp.ceil(N/2**(i+1)))
            
            # l = (2**(n-1) - 1) ^ (2**(n -i-1) - 1)
            # print(l)
            
            # barrier([balauca_anc[k] for k in range(N_int-2-N_int%2+2)])
            # for j in jrange(k//2):
                # print(l+k+j)
                # mcx([balauca_anc[l+2*j], balauca_anc[l+2*j+1]], 
                    # balauca_anc[l+k+j], 
                    # method = "gray")
                # 
                # barrier([balauca_anc[k] for k in range(N-2-N%2+2)])
            
            # l += k - k%2


    # def temp(qv):
        # for i in jrange(0, qv.size//2):
            # cx(qv[2*i], qv[2*i+1])
    
    # with conjugate(temp)(qv):
        # x(qv)
    #     mcx(qv[:qv.size-1], qv[qv.size-1])
    # with invert():
        # mcx([qv[0], qv[1]], qv[2], method = "gidney")
    
    print(balauca_anc.qs)
    
    # print(qv.get_measurement())
    # return measure(qv)




import time
t0 = time.time()
jaspr = main()
print(jaspr)
# print(jaspr.to_qc())
# jaspr.to_qc().transpile(4).to_qiskit().draw("mpl")

print(t0 -time.time())
# qc = jaspr.to_qc()
# qc.measure(qc.qubits[10:])
# qc.transpile(0).to_qiskit().draw("mpl")
# print(qc.run())


#%%

from qrisp import *
from qrisp.jasp import *

def quantum_mult(a, b):
    return a*b

# @boolean_simulation(bit_array_padding = 2**10)

@make_jaspr
def main(i, j, iterations):

    a = QuantumFloat(10)
    b = QuantumFloat(10)

    a[:] = i
    b[:] = j

    c = QuantumFloat(30)

    for i in jrange(iterations): 

        # Compute the quantum product
        temp = quantum_mult(a,b)

        # add into c
        c += temp

        # Uncompute the quantum product
        with invert():
            # The << operator "injects" the quantum variable into
            # the function. This means that the quantum_mult
            # function, which was originally out-of-place, is
            # now an in-place function operating on temp.

            # It can therefore be used for uncomputation
            # Automatic uncomputation is not yet available within Jasp.
            (temp << quantum_mult)(a, b)

        # Delete temp
        temp.delete()

    return measure(c)

jaspr = (main)(1,2, 120)
qc = jaspr.to_qc(1, 2, 5)[0]
#%%

print(qc.transpile().count_ops())
ops_amount = sum(qc.transpile().count_ops().values())
#%%
import time
t0 = time.time()
main(1, 2, 5)
time_taken = time.time()-t0
#%%

print(time_taken/ops_amount)

#%%

from qrisp import *

import time
def test_function_0(qv):
    x(qv[0])
    time.sleep(1)

def test_function_1(qv):
    y(qv[0])
    time.sleep(1)
    
@qache
def qached_function_executor(qv, func):
    func(qv)

@boolean_simulation
def main():
    
    a = QuantumFloat(2)
    a[:] = 2
    qv = a**4
    
    # qached_function_executor(qv, test_function_0)
    # qached_function_executor(qv, test_function_0)
    # qached_function_executor(qv, test_function_1)
    # qached_function_executor(qv, test_function_1)
    
    return measure(qv)
t0 = time.time()
# main()
benchmark_function(main)(stat_amount = 20)
print(t0 - time.time())

#%%

import qrisp
def fake_inversion(qf, res=None):

    if res is None:
        res = qrisp.QuantumFloat(qf.size+1)

    for i in qrisp.jrange(qf.size):
        qrisp.cx(qf[i],res[qf.size-i])
        
    return res

def QPE(psi, U, precision=None, res=None):

    if res is None:
        res = qrisp.QuantumFloat(precision, -precision)

    qrisp.h(res)

    # Performs a loop with a dynamic bound in Jasp mode.
    for i in qrisp.jrange(res.size):
        with qrisp.control(res[i]):
            for j in qrisp.jrange(2**i):
                U(psi)

    return qrisp.QFT(res, inv=True)

# This decorator converts the function to be executed within a repeat-until-success procedure.
# The function must return a boolean value as first return value and is repeatedly executed until the first return value is True.
@qrisp.RUS(static_argnums = [0,1])
def HHL_encoding(prepare_b, hamiltonian_evolution, n, precision):

    # Prepare the state |b>.
    qf = qrisp.QuantumFloat(n)
    prepare_b(qf)

    qpe_res = QPE(qf, hamiltonian_evolution, precision=precision)
    inv_res = fake_inversion(qpe_res)

    n = inv_res.size
    qbl = qrisp.QuantumBool()
    case_indicator = qrisp.QuantumFloat(n)
    # Auxiliary variable to evalutate the case_indicator.
    control_qbl = qrisp.QuantumBool()

    with qrisp.conjugate(qrisp.h)(case_indicator):
        for i in qrisp.jrange(n):
            qrisp.mcx(case_indicator[:i+1], 
                    control_qbl[0], 
                    method = "balauca", 
                    ctrl_state = 2**i-1)
        
            qrisp.mcx([control_qbl[0],inv_res[n-1-i]],
                    qbl[0])
           
           # Uncompute the auxiliary variable.
            qrisp.mcx(case_indicator[:i+1], 
                    control_qbl[0], 
                    method = "balauca", 
                    ctrl_state = 2**i-1)
            
    control_qbl.delete()
    
    # The first return value is a boolean value. Additional return values are QuantumVaraibles.
    return (qrisp.measure(case_indicator) == 0) & (qrisp.measure(qbl) == 1), qf, qpe_res, inv_res

def HHL(prepare_b, hamiltonian_evolution, n, precision):

    qv, qpe_res, inv_res = HHL_encoding(prepare_b, hamiltonian_evolution, n, precision)
    
    with qrisp.invert():
        QPE(qv, hamiltonian_evolution, res=qpe_res)
        fake_inversion(qpe_res, res=inv_res)
    
    return qv

from qrisp.operators import QubitOperator
import numpy as np

A = np.array([[3/8, 1/8], 
            [1/8, 3/8]])

H = QubitOperator.from_matrix(A).to_pauli()

# By default e^{-itH} is performed. Therefore, we set t=-pi.
def U(qf):
    H.trotterization()(qf,t=-np.pi,steps=1)

b = np.array([1,1])
# prepare_b = qrisp.prepare(b, reversed=True)
def prepare_b(qv):
    qrisp.prepare(qv, b)

@qrisp.jaspify
def main():

    qv = HHL(prepare_b, U, 1, 3)
    return qrisp.measure(qv)

res_dict = main()

for k, v in res_dict.items():
    res_dict[k] = v**0.5

print(res_dict)
#%%
res_dict = main()
#%%
for k, v in res_dict.items():
    res_dict[k] = v**0.5

print(res_dict)

#%%
from qrisp import *
@qrisp.make_jaspr
def main():
    a = QuantumFloat(2)
    return measure(a)


print(main())

#%%

from qrisp.operators import *
def hermitian_matrix_with_power_of_2_eigenvalues(n):
    # Generate eigenvalues as inverse powers of 2.
    eigenvalues = 1/np.exp2(np.arange(n))
    
    # Generate a random unitary matrix.
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    
    # Construct the Hermitian matrix.
    A = Q @ np.diag(eigenvalues) @ Q.conj().T
    
    return A

# Example 
n = 2
A = hermitian_matrix_with_power_of_2_eigenvalues(2**n)

print("Hermitian matrix A:")
print(A)

H = QubitOperator.from_matrix(A).to_pauli()

def U(qf):
    H.trotterization()(qf,t=-np.pi,steps=10)

b = np.array([1,1,0,0])
# Reverse the endianness for compatibility with Hamiltonian simulation.
prep_b = prepare_b

def main():

    qv = HHL(prep_b, U, 2, 4)

    # Reverse the endianness for compatibility with Hamiltonian simulation.
    n = qv.size
    for i in qrisp.jrange(n//2):
        qrisp.swap(qv[i],qv[n-i-1])

    return qv


sampling_function = qrisp.terminal_sampling(main)
res_dict = sampling_function()

for k, v in res_dict.items():
    res_dict[k] = v**0.5

np.array([res_dict[key] for key in sorted(res_dict)])

#%%
from qrisp import *


# @jaspify
def main():
    
    qv = QuantumFloat(12)
    # for i in range(10):
    jasp_balauca_mcx(qv[:11], [qv[11]], ctrl_state = 0)

    # qv.get_measurement()
    print(len(qv.qs.compile().qubits))
    # print(len(qv.qs.qubits))
    return measure(qv)
    
main()    
pass

#%%

from qrisp.circuit import ControlledOperation
def mcx_transpile_predicate(op):
    
    if isinstance(op, ControlledOperation):
        if op.base_operation.name == "x":
            return False
    return True

qc = main().to_qc()[0]
qc = transpile(qc, transpile_predicate = mcx_transpile_predicate)

#%%

from qrisp import QuantumFloat, z, invert

a = QuantumFloat(5)
b = QuantumFloat(5)
a[:] = 3
b[:] = 4

# The multiplication operator is an out-of-place function
c = a*b

print(c)

# Perform some dummy phase-tag

z(c[0])

# Manual uncomputation by injecting the multiplication function 
# into the result and inverting the whole process.

mult_func = lambda x, y : x*y

with invert():
    (c << mult_func)(a, b)

print(c)
#%%

from qrisp import QuantumFloat, z, invert

a = QuantumFloat(5)
b = QuantumFloat(5)
a[:] = 3
b[:] = 4

# The multiplication operator is an out-of-place function
c = a*b

print(c)
# Yields: {12: 1.0}

# In an actual algorithm, something would be done now.
# After that, we can do manual uncomputation by injecting
# the multiplication function into the result and
# inverting the whole process.

mult_func = lambda x, y : x*y

with invert():
    (c << mult_func)(a, b)

print(c)
# Yields: {0: 1.0}

# c can now be deallocated.
c.delete()

#%%
from qrisp import *
def cca_mcx(ctrl, target, anc):
   
    n = jlen(ctrl)
   
    # STEP 1
    # This operation is always executed regardless of the lenght of ctrl
    mcx([ctrl[0], ctrl[1]], anc[0])
 
    for i in jrange((n - 2) // 2):
        mcx([ctrl[2 * i + 2], ctrl[2 * i + 3]], ctrl[2 * i + 1])
 
    x(ctrl[:-2])
   
    # STEP 2
    a = n - 3 + 2 * (n & 1)
    b = n - 5 + (n & 1)
    c = n - 6 + (n & 1)
 
    with control(c > -1):
        mcx([ctrl[a], ctrl[b]], ctrl[c])
 
    with invert():
        for i in jrange((c + 1) // 2):
            mcx([ctrl[2 * i + 2], ctrl[2 * i + 1]], ctrl[2 * i])
    return ctrl, target, anc
 
 
def khattar_mcx(ctrl, target):
    anc = QuantumVariable(1)
 
    with conjugate(cca_mcx)(ctrl, target, anc):
        # STEP 3
        mcx([anc[0], ctrl[0]], target[0])
    anc.delete()
 
 
@terminal_sampling
def main(n):
    # Target register
    target = QuantumVariable(1)
    # Control register
    ctrl = QuantumVariable(n)
    h(ctrl)
    khattar_mcx(ctrl, target)
 
    return ctrl, target
 
# target = QuantumVariable(1)
# # Control register
# ctrl = QuantumVariable(8)
# h(ctrl)
# khattar_mcx(ctrl, target)
# print(ctrl.qs)
 
jsp = main(9)
 
#print(jsp.to_qc(8).transpile(2))
#%%

from qrisp import *
from qrisp.jasp import *
from jax import make_jaxpr


def main(i):
    qf = QuantumFloat(i)
    h(qf[0])
    cx(qf[0], qf[1])

    meas_float = measure(qf)

    return meas_float
    

jaspr = make_jaspr(main)(5)

print(jaspr)

#%%

print(jaspr(5))
#%%
    import time

    @qache
    def inner_function(qv, i):
        cx(qv[0], qv[1])
        h(qv[i])
        # Complicated compilation, that takes a lot of time
        time.sleep(1)

    def main(i):
        qv = QuantumFloat(i)

        inner_function(qv, 0)
        inner_function(qv, 1)
        inner_function(qv, 2)

        return measure(qv)


    t0 = time.time()
    jaspr = make_jaspr(main)(5)
    print(time.time()- t0)
    
#%%

    @qache
    def inner_function(qv):
        x(qv)
        time.sleep(1)

    def main():
        qf = QuantumFloat(5)
        qbl = QuantumBool(5)

        inner_function(qf)
        inner_function(qf)
        inner_function(qbl)
        inner_function(qbl)

        return measure(qf)

    t0 = time.time()
    jaspr = make_jaspr(main)()
    print(time.time()- t0)
    
#%%
    from jax.core import Tracer

    def main(i):
        print("i is dynamic?: ", isinstance(i, Tracer))
        
        qf = QuantumFloat(5)
        j = qf.size
        print("j is dynamic?: ", isinstance(i, Tracer))
        
        h(qf)
        k = measure(qf)
        print("k is dynamic?: ", isinstance(k, Tracer))

        # Regular Python integers are not dynamic
        l = 5
        print("l is dynamic?: ", isinstance(l, Tracer))

        # Arbitrary Python objects can be used within Jasp
        # but they are not dynamic
        import networkx as nx
        G = nx.DiGraph()
        G.add_edge(1,2)
        print("G is dynamic?: ", isinstance(l, Tracer))
        
        return k

    jaspr = make_jaspr(main)(5)
    

#%%

    @jaspify
    def main(k):

        a = QuantumFloat(k)
        b = QuantumFloat(k)

        # Brings a into uniform superposition via Hadamard
        h(a)

        c = measure(a)

        # Excutes c iterations (i.e. depending the measurement outcome)
        for i in jrange(c):

            # Performs a quantum incrementation on b based on the measurement outcome
            b += c//5

        return measure(b)

    print(main(5))
    
#%%

    @jaspify
    def main(i, j, k):

        a = QuantumFloat(5)
        a[:] = i
        
        qbl = QuantumBool()

        # a[:j] is a dynamic amount of controls
        mcx(a[:j], qbl[0], ctrl_state = k)

        return measure(qbl)
    
#%%

    @jaspify
    def main():

        qf = QuantumFloat(3)
        h(qf)

        # This is a classical, dynamical int
        meas_res = measure(qf)

        # This is a classical, dynamical bool
        ctrl_bl = meas_res >= 4
        
        with control(ctrl_bl):
            qf -= 4

        return measure(qf)

    for i in range(5):
        print(main())
        
#%%

    from qrisp.jasp import RUS, make_jaspr
    from qrisp import QuantumFloat, h, cx, measure

    def init_GHZ(qf):
        h(qf[0])
        for i in jrange(1, qf.size):
            cx(qf[0], qf[i])

    @RUS
    def rus_trial_function():
        qf = QuantumFloat(5)

        init_GHZ(qf)
        
        cancelation_bool = measure(qf[0])
        
        return cancelation_bool, qf

    @jaspify
    def main():

        qf = rus_trial_function()

        return measure(qf)

    print(main())
    
#%%

    @RUS
    def rus_trial_function():
        qf = QuantumFloat(5)

        init_GHZ(qf)
        
        cancelation_bool = measure(qf[0])
        
        return cancelation_bool, qf

    @terminal_sampling
    def main():

        qf = rus_trial_function()
        h(qf[0])

        return qf

    print(main())
    
#%%

    from qrisp import *
    from qrisp.jasp import *

    def quantum_mult(a, b):
        return a*b

    @boolean_simulation(bit_array_padding = 2**10)
    def main(i, j, iterations):

        a = QuantumFloat(10)
        b = QuantumFloat(10)

        a[:] = i
        b[:] = j

        c = QuantumFloat(30)

        for i in jrange(iterations): 

            # Compute the quantum product
            temp = quantum_mult(a,b)

            # add into c
            c += temp

            # Uncompute the quantum product
            with invert():
                # The << operator "injects" the quantum variable into
                # the function. This means that the quantum_mult
                # function, which was originally out-of-place, is
                # now an in-place function operating on temp.

                # It can therefore be used for uncomputation
                # Automatic uncomputation is not yet available within Jasp.
                (temp << quantum_mult)(a, b)

            # Delete temp
            temp.delete()

        return measure(c)
    
#%%

from qrisp import *

# @jaspify
def main():
    
    qv = QuantumFloat(20)
    h(qv)
    
    with invert():
        for i in jrange(2,4,1):
            qv += i

    print(qv.get_measurement())            
    return measure(qv)

print(main())
#%%

from qrisp.jasp import * 
from qrisp import *

from qrisp import p, QuantumVariable, QPE, multi_measurement

def U(qv):
    x = 0.5
    y = 0.125

    p(x*2*np.pi, qv[0])
    p(y*2*np.pi, qv[1])


@jaspify
def main():
    qv = QuantumFloat(3)
    qv[:] = 3
    res = QPE(qv, U, precision = 3)
    return measure(res)

print(main())

#%%
from qrisp import *

from qrisp import *

def test(args, oracle_function):
    oracle_function(*args)

def oracle_function(qb):   
    z(qb)

@terminal_sampling
def main():

    qv = QuantumFloat(2)
    qv[:] = 3

    return qv**5

res_dict = main()
#%%
from qiskit import (QuantumCircuit, QuantumRegister,
ClassicalRegister, transpile)
from qiskit_aer import Aer
from qiskit.circuit.library import RGQFTMultiplier
n = 6
a = QuantumRegister(n)
b = QuantumRegister(n)
res = QuantumRegister(2*n)
cl_res = ClassicalRegister(2*n)
qc = QuantumCircuit(a, b, res, cl_res)
for i in range(len(a)):
        if 3 & 1<<i: qc.x(a[i])
for i in range(len(b)):
        if 4 & 1<<i: qc.x(b[i])
qc.append(RGQFTMultiplier(n, 2*n),
list(a) + list(b) + list(res))
qc.measure(res, cl_res)
backend = Aer.get_backend('qasm_simulator')
qc = transpile(qc, backend)
counts_dic = backend.run(qc).result().get_counts()
print({int(k, 2) : v for k, v in counts_dic.items()})
#Yields: {12: 1024}
#%%
from qrisp import QuantumFloat
n = 6
a = QuantumFloat(n)
b = QuantumFloat(n)
a[:] = 3
b[:] = 4
res = a*b
print(res)
#Yields: {12: 1.0}
#%%

from qrisp import *

def function(qf):
    return qf*qf

def distribution(qf):
    h(qf)

qf_x = QuantumFloat(3,-3)
qf_y = QuantumFloat(6,-6)

qbl = QuantumBool()

@auto_uncompute
def state_function(qf_x, qf_y, qbl):

    distribution(qf_x)
    h(qf_y)

    with(qf_y < function(qf_x)):
        x(qbl)
        
a = IQAE([qf_x,qf_y,qbl], state_function, eps=0.01, alpha=0.01)

#%%

from qrisp.jasp import jaspify, q_while_loop, jrange
from qrisp import QuantumVariable,ry, measure, QuantumFloat, invert, cx
from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_fourier_adder import qft as jasp_QFT
from qrisp import *

@jaspify
def main():
    
    qf = QuantumFloat(7)
    qf2 = QuantumFloat(3)
    
    def body_fun(val):
        i, acc, qf, qf2 = val
        cx(qf[i],qf2[1])
        jasp_QFT(qf)
        acc += measure(qf)

        i += 1
        return i, acc, qf, qf2
    
    def cond_fun(val):
        return val[0] < 5  
    
    i, acc, qf, qf2 = q_while_loop(cond_fun, body_fun, (0, 0, qf, qf2))
    
    return acc ,measure(qf)

main()
# pass

#%%

import jax
import jax.numpy as jnp
from jax import grad, jit
import optax

# Define the model
def model(params, x):
    W, b = params
    return jax.nn.sigmoid(jnp.dot(x, W) + b)

# Define the loss function (binary cross-entropy)
def loss_fn(params, x, y):
    preds = model(params, x)
    return -jnp.mean(y * jnp.log(preds) + (1 - y) * jnp.log(1 - preds))

# Initialize parameters
key = jax.random.PRNGKey(0)
W = jax.random.normal(key, (2, 1))
b = jax.random.normal(key, (1,))
params = (W, b)

# Create optimizer
optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(params)

# Define training step
@jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Generate some dummy data
key = jax.random.PRNGKey(0)
X = jax.random.normal(key, (1000, 2))
y = jnp.sum(X > 0, axis=1) % 2

# Training loop
for epoch in range(100):
    params, opt_state, loss = train_step(params, opt_state, X, y)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Make predictions
predictions = model(params, X)
accuracy = jnp.mean((predictions > 0.5) == y)
print(f"Final accuracy: {accuracy}")

#%%
from qrisp.jasp import *
from qrisp import *

@RUS
def rus_trial_function(params):

    # Sample data from two QuantumFloats.
    # This is a placeholder for an arbitrary quantum algorithm.
    qf_0 = QuantumFloat(5)
    h(qf_0)

    qf_1 = QuantumFloat(5)
    h(qf_1)

    meas_res_0 = measure(qf_0)
    meas_res_1 = measure(qf_1)

    # Turn the data into a Jax array
    X = jnp.array([meas_res_0,meas_res_1])/2**qf_0.size

    # Evaluate the model
    model_res = model(params, X)

    # Determine the cancelation
    cancelation_bool = (model_res > 0.5)[0]

    return cancelation_bool, qf_0

@jaspify
def main(params):

    qf = rus_trial_function(params)
    h(qf[0])

    return measure(qf)

print(main(params))

#%%

@jaspify
def main():
    
    k = 9
    a = QuantumFloat(k)
    #a[:] = 5
    h(a)
    
    b = a.duplicate()
    #cx(a, b)
    h(b)
    
    c = a*b
    
    return measure(c)

import time
t0 = time.time()
print(main())
print(time.time()-t0)

#%%

import jax
import jax.numpy as jnp


def cobyla(func, x0, cons, rhobeg=1.0, rhoend=1e-6, maxfun=1000):
    n = len(x0)
    m = len(cons)
    
    # Initialize the simplex
    sim = jnp.zeros((n + 1, n))
    sim = sim.at[0].set(x0)
    sim = sim.at[1:].set(x0 + jnp.eye(n) * rhobeg)
    
    # Initialize function values and constraint values
    f = jax.vmap(func)(sim)
    c = jax.vmap(lambda x: jnp.array([con(x) for con in cons]))(sim)
    
    def body_fun(state):
        sim, f, c, rho, nfeval = state
        
        # Find the best and worst points
        best = jnp.argmin(f)
        worst = jnp.argmax(f)
        
        # Calculate the centroid of the simplex excluding the worst point
        # Calculate the centroid of the simplex excluding the worst point
        mask = jnp.arange(n + 1) != worst
        centroid = jnp.sum(sim * mask[:, None], axis=0) / n
        
        # Reflect the worst point
        xr = 2 * centroid - sim[worst]
        fr = func(xr)
        cr = jnp.array([con(xr) for con in cons])
        nfeval += 1
        
        # Expansion
        xe = 2 * xr - centroid
        fe = func(xe)
        ce = jnp.array([con(xe) for con in cons])
        nfeval += 1
        
        # Contraction
        xc = 0.5 * (centroid + sim[worst])
        fc = func(xc)
        cc = jnp.array([con(xc) for con in cons])
        nfeval += 1
        
        # Update simplex based on conditions
        cond_reflect = (fr < f[best]) & jnp.all(cr >= 0)
        cond_expand = (fe < fr) & cond_reflect
        cond_contract = (fc < f[worst]) & jnp.all(cc >= 0)
        
        sim = jnp.where(cond_expand, sim.at[worst].set(xe), 
                jnp.where(cond_reflect, sim.at[worst].set(xr), 
                    jnp.where(cond_contract, sim.at[worst].set(xc), 
                        0.5 * (sim + sim[best]))))
        
        f = jax.vmap(func)(sim)
        c = jax.vmap(lambda x: jnp.array([con(x) for con in cons]))(sim)
        
        rho *= 0.5
        return sim, f, c, rho, nfeval
    
    def cond_fun(state):
        _, _, _, rho, nfeval = state
        return (rho > rhoend) & (nfeval < maxfun)
    
    # Main optimization loop
    state = (sim, f, c, rhobeg, n + 1)
    state = jax.lax.while_loop(cond_fun, body_fun, state)
    
    sim, f, _, _, _ = state
    best = jnp.argmin(f)
    return sim[best], f[best]

@jax.jit
def objective(x):
    return jnp.abs(x[0]**2 + x[1]**2 - 13.37)


@jax.jit
def constraint(x):
    return x[0] + x[1] - 1

x0 = jnp.array([0.5, 0.5])
cons = []

@jaspify
def main(x0):
    
    
    result, value = cobyla(objective, x0, cons)
    return result, value

result, value = main(x0)
    
print("Optimal solution:", result)
print("Optimal value:", value)
print(sum(result**2))

#%%


import numpy as np

from qrisp.jasp import qache, jrange
from qrisp.core import swap, h, cx, t, t_dg, s, p, measure, cz, cp, QuantumVariable
from qrisp.qtypes import QuantumBool, QuantumFloat
from qrisp.environments import control, custom_control, conjugate, invert
from qrisp.alg_primitives.arithmetic import gidney_adder
                
@qache
def jasp_mod_adder(a, b, modulus, inpl_adder = gidney_adder, ctrl = None):
    
    reduction_not_necessary = QuantumBool()
    # sign = QuantumBool()
    sign = b[-1]
    
    
    if isinstance(a, int):
        a = a%modulus
    
    # b = list(b) + [sign[0]]
    
    if ctrl is None:
        inpl_adder(a, b)
    else:
        with control(ctrl):
            inpl_adder(a, b)
            
    with invert():
        inpl_adder(modulus, b)

    cx(sign, reduction_not_necessary[0])
    
    with control(reduction_not_necessary[0]):
        inpl_adder(modulus, b)
        
    with invert():
        if ctrl is None:
            inpl_adder(a, b)
        else:
            with control(ctrl):
                inpl_adder(a, b)
    
    cx(sign, reduction_not_necessary[0])
    reduction_not_necessary.flip()
    
    if ctrl is None:
        inpl_adder(a, b)
    else:
        with control(ctrl):
            inpl_adder(a, b)
    
    reduction_not_necessary.delete()
    
@jaspify
def main():
    
    a = QuantumFloat(5)
    b = QuantumFloat(5)
    
    a[:] = 3
    b[:] = 6
    jasp_mod_adder(a, b, 7)
    
    return measure(b)
    
print(main())

#%%


from qrisp import *
from qrisp.operators import Z,A,C
from qrisp.algorithms.vqe.vqe_problem import *

def create_hamiltonian(n_fermions, J=-1, m=1, μ=.5):
    H=0
    for i in range(0,(2*n_fermions-3), 2):
        H+= -1*(2*A(i)*Z(i+1)*2*C(i+2))
    H += -2*A(2*n_fermions-2)
    return H 

n_fermions=3
n_qubits=2*n_fermions - 1
H=create_hamiltonian(n_fermions)

# generic Ansatz
def ansatz(qv,theta):
    nqubits=qv.size
    for i in range(nqubits):
        ry(theta[i],qv[i])
    for i in range(nqubits-1):
        cx(qv[i],qv[i+1])
    cx(qv[-1],qv[0])

vqe = VQEProblem(hamiltonian = H,
                 ansatz_function = ansatz,
                 num_params=n_qubits,
                 callback=True)
print(vqe.hamiltonian.ground_state_energy())
print(H)
energy = vqe.run(qarg = QuantumVariable(n_qubits),
              depth = 1,max_iter=300
              )
vqe.visualize_energy(exact=True)

#%%

import jax
import jax.numpy as jnp
from qrisp import jaspify, QuantumVariable, rx, sample, expectation_value, QuantumFloat
import networkx as nx
from numba import njit, prange


def cobyla(func, x0, 
           #func_args, 
           cons=[], rhobeg=1.0, rhoend=1e-6, maxfun=100):

    #func = func_in(func_args)

    n = len(x0)
    m = len(cons)
    
    # Initialize the simplex
    sim = jnp.zeros((n + 1, n))
    sim = sim.at[0].set(x0)
    sim = sim.at[1:].set(x0 + jnp.eye(n) * rhobeg)
    
    # Initialize function values and constraint values
    # f = jax.vmap(func)(sim)
    f = jax.lax.map(func, sim)
    c = jax.vmap(lambda x: jnp.array([con(x) for con in cons]))(sim)
    
    def body_fun(state):
        sim, f, c, rho, nfeval = state
        
        # Find the best and worst points
        best = jnp.argmin(f)
        worst = jnp.argmax(f)
        
        # Calculate the centroid of the simplex excluding the worst point
        # Calculate the centroid of the simplex excluding the worst point
        mask = jnp.arange(n + 1) != worst
        centroid = jnp.sum(sim * mask[:, None], axis=0) / n
        
        # Reflect the worst point
        xr = 2 * centroid - sim[worst]
        fr = func(xr)
        cr = jnp.array([con(xr) for con in cons])
        nfeval += 1
        
        # Expansion
        xe = 2 * xr - centroid
        fe = func(xe)
        ce = jnp.array([con(xe) for con in cons])
        nfeval += 1
        
        # Contraction
        xc = 0.5 * (centroid + sim[worst])
        fc = func(xc)
        cc = jnp.array([con(xc) for con in cons])
        nfeval += 1
        
        # Update simplex based on conditions
        cond_reflect = (fr < f[best]) & jnp.all(cr >= 0)
        cond_expand = (fe < fr) & cond_reflect
        cond_contract = (fc < f[worst]) & jnp.all(cc >= 0)
        
        sim = jnp.where(cond_expand, sim.at[worst].set(xe), 
                jnp.where(cond_reflect, sim.at[worst].set(xr), 
                    jnp.where(cond_contract, sim.at[worst].set(xc), 
                        0.5 * (sim + sim[best]))))
        
        # f = jax.vmap(func)(sim)
        f = jax.lax.map(func, sim)
        c = jax.vmap(lambda x: jnp.array([con(x) for con in cons]))(sim)
        
        rho *= 0.5
        return sim, f, c, rho, nfeval
    
    def cond_fun(state):
        _, _, _, rho, nfeval = state
        return (rho > rhoend) & (nfeval < maxfun)
    
    # Main optimization loop
    state = (sim, f, c, rhobeg, n + 1)
    state = jax.lax.while_loop(cond_fun, body_fun, state)
    
    sim, f, _, _, _ = state
    best = jnp.argmin(f)
    return sim[best], f[best]





@jax.jit
def objective(x):
    return jnp.abs(x[0]**2 + x[1]**2 + x[4] - x[3] - 13.37)


def apply_rx(x_angles):
    qv = QuantumFloat(6)
    for p in range(6):
        rx(x_angles[p], qv[p])
    return qv

def maxcut_obj_JJ(G):
    edge_list = G.edges()
    n_nodes = len(G.nodes)
    #qv = QuantumVariable(n_nodes)

    #@jax.jit
    def maxcut_obj_jitted(x_angles):
        
        #print(n_nodes)
        #print(x_angles.shape)
        
        for p in range(n_nodes):
            #print(p)
            #jax.debug.print("{x_angles} rr", x_angles=x_angles)
            #print(qv[p])
            #app_func= apply_rx(x_angles)
            res = expectation_value(state_prep = apply_rx, shots = 1000)(x_angles)

        """ cut = 0
        for i, j in edge_list:
            # the edge is cut
            #if ((x >> i) ^ (x >>j)) & 1:
            # yeah also... this is not gonna work huh...
            # try the injection from above?
            if qv[i] != qv[j]:                          
                cut -= 1 """
        return -res

    return maxcut_obj_jitted

@jax.jit
def constraint(x):
    return x[0] + x[1] - 1

#x0 = jnp.array([0.5, 0.5])
cons = []

from qrisp import *

@jaspify(terminal_sampling = True)
def main():
    num_nodes = 6
    G = nx.erdos_renyi_graph(num_nodes, 0.7, seed=99)
    # qarg = QuantumVariable(G.number_of_nodes())
    #x0 = jnp.array([0.5, 0.5,  0.1,  0.3])
    key = jax.random.key(0)
    x0 = jax.random.uniform(key= key, shape=(num_nodes,))
    obj_func = maxcut_obj_JJ(G)
    result, value = cobyla(obj_func, x0, cons=[])
    return result, value

result, value = main()
    
print("Optimal solution:", result)
print("Optimal value:", value)
print(sum(result**2))

#%%

from qrisp import *

def test(args, oracle_function):
    oracle_function(*args)

def oracle_function(qb):   
    z(qb)

def test(qv):
    pass

@terminal_sampling
def main():

    qv = QuantumFloat(1)

    test(*qv)
    # test([qv], oracle_function) # This works!
    # test(*qv, oracle_function) # This does NOT work! (Computation does not terminate.)

    return qv

res_dict = main()

#%%

from qrisp import *

@terminal_sampling
def main():

    # qv = QuantumBool() # Does NOT work
    qv = QuantumFloat(1) # Does work

    qv2 = QuantumFloat(1)

    mcx(qv, 
        qv2[0], 
        method = "balauca", 
        ctrl_state = 0)

    return qv

main()

#%%

from qrisp import *


@make_jaspr
def main():

    a = QuantumFloat(2)
    b = QuantumFloat(1)
    #b = QuantumFloat(2) # This does work!
    a[:] = 2
    b[:] = 1

    a+=b

    return a

print(main().to_qc()[0].transpile(1))

#%%

class Foo:
    
    def __enter__(self):
        raise Exception
        
    def __exit__(self, exception_type, exception_value, traceback):
        
        print(exception_type)
        print("test")
        return True
        
        
with Foo():
    pass
    

print("lumppi")
        

#%%

class Foo:
    def __init__(self, x):
        self.x = x

    def __enter__(self):
        # Evaluate the condition here
        if not self.should_execute():
            # Raise an exception to skip the block
            raise RuntimeError("Skipping block")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type == RuntimeError and str(exc_value) == "Skipping block":
            # Suppress the exception to skip the block silently
            return True
        # Return False for other exceptions to propagate them
        return False

    def should_execute(self):
        # Implement your condition here
        return self.x > 0  # Example condition

x = -5  # or any value you want to test
with Foo(x):
    print("This block will only execute if x > 0")

#%%

from qrisp import *
def main(i):
    
    qv = QuantumVariable(3)
    
    with control(i < 3):
        x(qv[i])
    
    print(qv.qs)

main(4)

#%%

import jax
import jax.numpy as jnp
from qrisp import jaspify, QuantumVariable, rx, sample, expectation_value, QuantumFloat
import networkx as nx
from numba import njit, prange


def cobyla(func, x0, 
           #func_args, 
           cons=[], rhobeg=1.0, rhoend=1e-6, maxfun=100):

    #func = func_in(func_args)

    n = len(x0)
    m = len(cons)
    
    # Initialize the simplex
    sim = jnp.zeros((n + 1, n))
    sim = sim.at[0].set(x0)
    sim = sim.at[1:].set(x0 + jnp.eye(n) * rhobeg)
    
    # Initialize function values and constraint values
    # f = jax.vmap(func)(sim)
    f = jax.lax.map(func, sim)
    c = jax.vmap(lambda x: jnp.array([con(x) for con in cons]))(sim)
    
    def body_fun(state):
        sim, f, c, rho, nfeval = state
        
        # Find the best and worst points
        best = jnp.argmin(f)
        worst = jnp.argmax(f)
        
        # Calculate the centroid of the simplex excluding the worst point
        # Calculate the centroid of the simplex excluding the worst point
        mask = jnp.arange(n + 1) != worst
        centroid = jnp.sum(sim * mask[:, None], axis=0) / n
        
        # Reflect the worst point
        xr = 2 * centroid - sim[worst]
        fr = func(xr)
        cr = jnp.array([con(xr) for con in cons])
        nfeval += 1
        
        # Expansion
        xe = 2 * xr - centroid
        fe = func(xe)
        ce = jnp.array([con(xe) for con in cons])
        nfeval += 1
        
        # Contraction
        xc = 0.5 * (centroid + sim[worst])
        fc = func(xc)
        cc = jnp.array([con(xc) for con in cons])
        nfeval += 1
        
        # Update simplex based on conditions
        cond_reflect = (fr < f[best]) & jnp.all(cr >= 0)
        cond_expand = (fe < fr) & cond_reflect
        cond_contract = (fc < f[worst]) & jnp.all(cc >= 0)
        
        sim = jnp.where(cond_expand, sim.at[worst].set(xe), 
                jnp.where(cond_reflect, sim.at[worst].set(xr), 
                    jnp.where(cond_contract, sim.at[worst].set(xc), 
                        0.5 * (sim + sim[best]))))
        
        # f = jax.vmap(func)(sim)
        f = jax.lax.map(func, sim)
        c = jax.vmap(lambda x: jnp.array([con(x) for con in cons]))(sim)
        
        rho *= 0.5
        return sim, f, c, rho, nfeval
    
    def cond_fun(state):
        _, _, _, rho, nfeval = state
        return (rho > rhoend) & (nfeval < maxfun)
    
    # Main optimization loop
    state = (sim, f, c, rhobeg, n + 1)
    state = jax.lax.while_loop(cond_fun, body_fun, state)
    
    sim, f, _, _, _ = state
    best = jnp.argmin(f)
    return sim[best], f[best]





@jax.jit
def objective(x):
    return jnp.abs(x[0]**2 + x[1]**2 + x[4] - x[3] - 13.37)


def apply_rx(x_angles):
    qv = QuantumFloat(6)
    for p in range(6):
        rx(x_angles[p], qv[p])
    return qv

def maxcut_obj_JJ(G):
    edge_list = G.edges()
    n_nodes = len(G.nodes)
    #qv = QuantumVariable(n_nodes)

    #@jax.jit
    def maxcut_obj_jitted(x_angles):
        
        #print(n_nodes)
        #print(x_angles.shape)
        
        for p in range(n_nodes):
            #print(p)
            #jax.debug.print("{x_angles} rr", x_angles=x_angles)
            #print(qv[p])
            #app_func= apply_rx(x_angles)
            res = expectation_value(state_prep = apply_rx, shots = 1000)(x_angles)

        """ cut = 0
        for i, j in edge_list:
            # the edge is cut
            #if ((x >> i) ^ (x >>j)) & 1:
            # yeah also... this is not gonna work huh...
            # try the injection from above?
            if qv[i] != qv[j]:                          
                cut -= 1 """
        return -res

    return maxcut_obj_jitted

@jax.jit
def constraint(x):
    return x[0] + x[1] - 1

#x0 = jnp.array([0.5, 0.5])
cons = []

from qrisp import *

@jaspify(terminal_sampling = True)
def main():
    num_nodes = 6
    G = nx.erdos_renyi_graph(num_nodes, 0.7, seed=99)
    # qarg = QuantumVariable(G.number_of_nodes())
    #x0 = jnp.array([0.5, 0.5,  0.1,  0.3])
    key = jax.random.key(0)
    x0 = jax.random.uniform(key= key, shape=(num_nodes,))
    obj_func = maxcut_obj_JJ(G)
    result, value = cobyla(obj_func, x0, cons=[])
    return result, value

result, value = main()
    
print("Optimal solution:", result)
print("Optimal value:", value)
print(sum(result**2))

#%%

# Create some dummy Qiskit quantum circuit to integrate into a jasp function
from qiskit import QuantumCircuit as qiskitQuantumCircuit

qiskit_qc = qiskitQuantumCircuit(2)
qiskit_qc.x(0)
qiskit_qc.cx(0, 1)


from qrisp import QuantumCircuit, jaspify, QuantumFloat, measure
# Convert the Qiskit QuantumCircuit to a Qrisp QuantumCircuit
qrisp_qc = QuantumCircuit.from_qiskit(qiskit_qc)
# Convert to a gate object
qrisp_gate_obj = qrisp_qc.to_gate()


@jaspify
def main():
    qv = QuantumFloat(2)
    # Append the gate object to the appropriate qubits
    qv.qs.append(qrisp_gate_obj, [qv[0], qv[1]])
    return measure(qv)

# Execute the function
print(main())


#%%

from qrisp import QuantumVariable, x
from qrisp import dicke_state

qv = QuantumVariable(4)
x(qv[2])
x(qv[3])

dicke_state(qv, 2)