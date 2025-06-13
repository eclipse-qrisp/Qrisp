"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************
"""

from qrisp import  cx,  x, ry,  control, invert, h, QuantumFloat, rz,  measure
from qrisp.alg_primitives.qft import QFT
from qrisp.jasp import jrange, terminal_sampling, check_for_tracing_mode, q_while_loop

import jax
from jax.scipy.special import gammaln
import jax.numpy as jnp



@jax.jit
def comb(N, k):
  integ = jnp.uint16(jnp.round(jnp.exp(gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1))))
  return integ


def dicke_state(qv,k):
    """
    Dicke State initialization of a QuantumVariable, based on the deterministic alogrithm in `Deterministic Preparation of Dicke States (2019)
    <https://arxiv.org/abs/1904.07358>`_. 

    This algorithm creates an equal superposition of Dicke states for a given Hamming weight. The initial input QuantumVariable has to be within this subspace.
    Works in JASP and non-JASP mode, see

    Parameters
    ----------
    qv : QuantumVariable
        Initial quantum variable to be prepared. Has to be in target subspace.
    k : Int
        The Hamming weight (i.e. number of "ones") for the desired dicke state
    little_endian : Boolean
        A flag to indicate a reversed qubit order. This is needed for the Divide-And-Conquer approach.

    Examples
    --------
    We initiate a QuantumVariable in the "0011" state and from this create the Dicke state with Hamming weight 2.

    ::
        
        from qrisp import QuantumVariable, x, dicke_state
        
        qv = QuantumVariable(4)
        x(qv[2])
        x(qv[3])

        dicke_state(qv, 2)

    """
    
    # jasp compatibility
    if check_for_tracing_mode():
        n = qv.size
    else:
        n = len(qv)

    # SCS cascade
    with invert():
        for index2 in jrange(k+1, n+1):
            split_cycle_shift(qv, index2, k,)
        #barrier(qv)
    with invert():
        for index in jrange(2,k+1):
            split_cycle_shift(qv, index, index-1, )
        #barrier(qv)
 

def split_cycle_shift(qv, highIndex, lowIndex):
    """

    Helper function for Dicke State initialization of a QuantumVariable, based on the deterministic alogrithm in `Deterministic Preparation of Dicke States (2019)
    <https://arxiv.org/abs/1904.07358>`_. 
    
    Parameters
    ----------
    qv : QuantumVariable
        Initial quantum variable to be prepared. Has to be in target subspace.
    highIndex : Int
        Index for indication of preparation steps, as seen in original algorithm.
    lowIndex : Int
        Index for indication of preparation steps, as seen in original algorithm.
    """

    with invert():
        # reversed jrange
        for i in jrange(lowIndex): 

            index = highIndex - i 
            param = 2 * jnp.arccos(jnp.sqrt((highIndex - index + 1 ) /(highIndex)) )

            ctrL_bool = index == highIndex
            ctrL_bool_false = index != highIndex

            # application of the c-ry rotations 
            with control(ctrL_bool):
                cx(qv[highIndex - 2], qv[highIndex-1]) 
                with control( qv[highIndex-1] ):
                    ry(param, qv[highIndex - 2])
                cx(qv[highIndex - 2], qv[highIndex -1])
            
            with control(ctrL_bool_false):
                cx(qv[index -2], qv[highIndex-1]) 
                with control([qv[highIndex -1],qv[index -1]]):
                    ry(param, qv[index - 2])
                cx(qv[index -2], qv[highIndex-1]) 
                



def dicke_divide_and_conquer(qv, k):
    """
    Dicke State initialization of a QuantumVariable, based on the deterministic alogrithm in `A Divide-and-Conquer Approach to Dicke State preparation (2021) <https://arxiv.org/abs/2112.12435>`_. 
    This algorithm creates an equal superposition of states for a given Hamming weight. The QuantumVariable has to be in the ``0``-State.
    Works in JASP and non-JASP mode, see
    
    Parameters
    ----------
    qv : QuantumVariable
        Initial quantum variable to be prepared. Has to be in target subspace.
    k : Int
        The Hamming weight (i.e. number of "ones") for the desired Dicke state.

    
    Examples
    --------

    We initiate a QuantumVariable with 7 qubits from this create the Dicke state with Hamming weight 3 in JASP mode.

    ::

        @terminal_sampling
        def main():
            n = 7
            qv_1 = QuantumVariable(n)
            dicke_divide_and_conquer_jasp(qv_1, 3)

            return qv_1

        res_jasp = main()

        
    Similarly, we can do the same thing in non-JASP mode: 
    
    ::

        n = 7
        qv_2 = QuantumVariable(n)
        dicke_divide_and_conquer_jasp(qv_2, 3)

        res = qv.get_measurement()

    """

    # jasp compatibility
    if check_for_tracing_mode():
        n = qv.size
    else:
        n = len(qv)

    n_1 = jnp.floor(n/2)
    n_2 = n - n_1

    # @gate_wrap
    # the divide step 
    def dicke_divide(qv):

        # precompute rotation angles
        l_xi = []
        rotation_angles = jnp.zeros(k)
        l_xi = jnp.zeros(k+1)
        for i1 in range(k+1):
            x_i = comb(n_1,i1)*comb(n_2,k-i1)
            l_xi = l_xi.at[i1].set(x_i)

        for i2 in range(k):
            temp_sum = jnp.sum(l_xi[i2:])
            rot_val = 2*jnp.acos(jnp.sqrt(l_xi[i2]/temp_sum))
            rotation_angles = rotation_angles.at[i2].set(rot_val)
        
        n_1h = n_1.astype(int)

        # start by applying the rotations
        ry(rotation_angles[0], qv[n_1h-1])
        for i in range(1,k):
            with control(qv[n_1h-i]):
                ry(rotation_angles[i], qv[n_1h-i-1])

        # then flip the correct qubits and ladder cx
        x(qv[n-k:n])
        for i in range(k):
            cx(qv[n_1h-k+i], qv[-(i+1)])


    dicke_divide(qv)
    # barrier(qv)
    n_1a = n_1.astype(int)
    n_2a = n_2.astype(int)
    dicke_state(qv[:n_1a], k) # needs to be big endian one
    # barrier(qv)
    dicke_state(qv[n-n_2a:], k)
    # barrier(qv)     

    return qv


def collective_hamming_measurement(qf, n):
    """
    Implementation of a collective Hamming weight measurement, based on `Shallow Quantum Circuit Implementation of
    Symmetric Functions with Limited Ancillary Qubits <https://arxiv.org/pdf/2404.06052>`_. 

    Parameters
    ----------
    qf: QuantumFloat
        The QuantumFloat, of which the Hamming weight is to be measured.
    n: Int
        Size of the QuantumFloat ``qf``.

    Returns
    -------
    ancillas: QuantumFloat
        The ancillas, in which the Hamming weight of ``qf`` is encoded

    """
    n_anc = jnp.ceil(jnp.log2(n)+1).astype(int)
    ancillas = QuantumFloat(n_anc)
    h(ancillas)
    for i in jrange(n_anc):
        rz(2*n*jnp.pi/(2**(n_anc+1-i)),ancillas[i])
        
    for i in jrange(n_anc):
        for k in jrange(n):
            cx(ancillas[n_anc-i-1], qf[k])
        for k in jrange(n):
            rz(-2*jnp.pi/(2**(i+2)), qf[k])
        for k in jrange(n):
            cx(ancillas[n_anc-i-1], qf[k])
    QFT(ancillas,inv=True)
    
    return ancillas

def iterative_dicke_state_sampling(qf, m_t):
    """
    Implementation of a Dicke state preparation routine, based on `Efficient preparation of Dicke states (2024)
    <https://arxiv.org/pdf/2411.03428>`_. 
    This algorithm creates an equal superposition of states for a given Hamming weight. The QuantumVariable has to be in the ``0``-State, other inputs have not been investigated.
    Intended to be used in JASP-mode. 

    Parameters
    ----------
    qf: QuantumFloat
        The QuantumFloat to be prepared.
    m_t: Int
        The Hamming weight of the Dicke state.

    Returns
    -------
    qf1: QuantumFloat
        The QuantumFloat which is prepared as a Dicke state.

    Examples
    --------

    We initiate a QuantumVariable with 7 qubits from this create the Dicke state with Hamming weight 3 in JASP mode.

    ::

        @terminal_sampling
        def main():
                
            n = 7
            k = 3
            qv_iter = QuantumFloat(n)
            qv_iter = iterative_dicke_state_sampling(qv_iter,k)

            return qv_iter

    """
    
    j = qf.size # 2*? 


    # algebra from paper
    r_mt = jnp.sqrt(j*(j+1)-m_t**2)
    r_0 = jnp.sqrt(j*(j+1))

    # jasp q_while loop body function
    def body_fun(val):
        m_t, qf1, theta,j, m = val
        # algebra from paper
        r_m = jnp.sqrt(j * (j+1) - m.astype(float) **2)
        theta = jnp.asin((m * r_mt - m_t.astype(float) * r_m) /r_0**2)

        # rotation towards desired state
        for t in jrange(j):
            ry(theta, qf1[t])

        # collective hamming weight measurement and uncomputation
        ancillas =collective_hamming_measurement(qf1,j)
        m = measure(ancillas)

        #with invert():
                #(ancillas << collective_hamming_measurement)(qf1,j)

        ancillas.delete()

        return m_t, qf1,theta ,j, m 
    
    def cond_fun(val):
        return val[0] != val[-1]


    thet_0 = 0
    
    m_t, qf1, thet_0, j, m  = q_while_loop(cond_fun, body_fun, (m_t, qf,thet_0 ,j,j))
    
    return qf1

