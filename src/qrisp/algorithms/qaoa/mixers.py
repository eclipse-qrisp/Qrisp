"""
\********************************************************************************
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
********************************************************************************/
"""

import numpy as np
from scipy.optimize import minimize
from sympy import Symbol

from qrisp import QuantumVariable, h, barrier, rz, rx , cx, QuantumArray, xxyy, p, invert, conjugate, mcp, auto_uncompute, control

def RX_mixer(qv, beta):
    """
    Applies an RX gate to each qubit in ``qv``.

    The RX gate is a single-qubit rotation about the x-axis. It is used as a mixer in QAOA to drive transitions between different states.

    Parameters
    ----------
    qv : QuantumVariable
        The quantum variable to which the RX gate is applied.
    beta : float or sympy.Symbol
        The phase shift value for the RX gate.

    Returns
    -------
    qv : QuantumVariable
        The quantum variable after applying the RX gate.
    """
    for i in range(qv.size):
        rx(2 * beta, qv[i])
    return qv


def XY_mixer(qv, beta):
    """
    Applies multiple XX+YY gates to ``qv`` such that each qubit has interacted with it's neighbour at least once.

    The XX+YY gate is a two-qubit gate that performs rotations around the XY plane. It is used as a mixer in QAOA to drive transitions between different states.
    
    A defining feature of this mixer is the fact, that it keeps the number of ones (or equivalently zeros) in the binary representation of the state invariant.

    Parameters
    ----------
    qv : QuantumVariable
        The quantum variable to which the XY gate is applied.
    beta : float or sympy.Symbol
        The phase shift value for the XY gate.

    Returns
    -------
    qv : QuantumVariable
        The quantum variable after applying the XY gate.
    """
    N = qv.size
    
    for i in range(0, N//2):
        q1 = qv[2*i]
        q2 = qv[2*i+1]
        xxyy(4*beta, 0, q1, q2)
    
    for i in range(0, (N-2+N%2)//2):
        q1 = qv[2*i+1]
        q2 = qv[2*i+2]
        xxyy(4*beta, 0, q1, q2)
        
    xxyy(4*beta, 0, qv[N-1], qv[0])

    return qv


def apply_XY_mixer(quantumcolor_array, beta):
    for qcolor in quantumcolor_array:
       XY_mixer(qcolor, beta)
    return quantumcolor_array


def RZ_mixer(qv, beta):
    """
    This function applies an RZ gate with a negative phase shift to a given quantum variable.

    Parameters
    ----------
    qv : QuantumVariable
        The quantum variable to which the RZ gate is applied.
    beta : float or sympy.Symbol
        The phase shift value for the RZ gate.

    """
    rz(-beta, qv)
    
def grover_mixer(qv, beta):
    """
    Performs the parametrized Grover diffuser.

    Parameters
    ----------
    qv : QuantumVariable
        The QuantumVariable to be mixed.
    beta : float or sympy.Symbol
        The mixing parameter.

    """
    
    
    from qrisp.grover import diffuser
    diffuser(qv, phase = beta)
    
def constrained_mixer_gen(constraint_oracle, winner_state_amount):
    r"""
    Generates a customized mixer function that leaves arbitrary constraints intact. 
    The constraints are specified via a ``constraint_oracle`` function, which
    is taking a :ref:`QuantumVariable` or :ref:`QuantumArray` and apply a phase $\phi$
    (specified by the keyword argument ``phase``) to the states that are allowed
    by the constraints.
    
    Additionally the amount of winner states needs to be known. For this the user
    needs to provide the function ``winner_state_amount``, that returns the number
    of winner states for a given qubit amount. This number can be an approximation,
    however faulty values can cause leakage into the state-space that is forbidden
    by the constraints.
    
    For more details regarding implementation specifics please check the 
    corresponding :ref:`tutorial <ConstrainedMixers>`.

    Parameters
    ----------
    constraint_oracle : function
        A function of a :ref:`QuantumVariable` or :ref:`QuantumArray`. Also needs to
        support the keyword argument ``phase``. This function should apply the phase
        specified by the keyword argument to the allowed states.
    winner_state_amount : function
        A function of a QuantumVariable or QuantumArray, that returns the amount 
        of winner states for that QuantumVariable.


    Returns
    -------
    constrained_mixer : function
        A mixer function that does not leave the allowed space specified by the oracle.
        
    Examples
    --------
    
    We create a mixer function that only mixes among the states where the first and the
    last qubit disagree. In more mathematical terms - they satisfy the following 
    constraint function.
    
        
    .. math::

        f: \mathbb{F}_2^n \rightarrow \mathbb{F}_2, x \rightarrow (x_{n-1} \neq x_0)
        
    ::
        
        from qrisp.qaoa import constrained_mixer_gen
        from qrisp import QuantumVariable, auto_uncompute, cx, p

        @auto_uncompute
        def constraint_oracle(qarg, phase):
        
            predicate = QuantumBool()        
            
            cx(qarg[0], predicate)
            cx(qarg[-1], predicate)
            p(phase, predicate)
          
        def winner_state_amount(qarg):
            return 2**(len(qarg) - 1)
          
        mixer = constrained_mixer_gen(constraint_oracle, winner_state_amount)
        
    To test the mixer, we create a :ref:`QuantumVariable`:
        
    ::
        
        import numpy as np
        beta = np.pi
        
        qv = QuantumVariable(3)
        qv[:] = "101"
        mixer(qv, beta)
        print(qv)
        #Yields: {'101': 1.0} 
        #Leaves forbidden states invariant
        
        qv = QuantumVariable(3)
        qv[:] = "100"
        mixer(qv, beta)
        print(qv)
        #Yields: {'100': 0.25, '110': 0.25, '001': 0.25, '011': 0.25}
        #Only mixes among allowed states


    """
    
    from qrisp.grover import grovers_alg
    
    def prep_psi(qarg):
        
        if isinstance(qarg, QuantumVariable):
            qubit_amount = len(qarg)
        elif isinstance(qarg, QuantumArray):
            qubit_amount = len(qarg.qtype)*len(qarg.flatten())
        else:
            raise Exception(f"Argument type {type(qarg)} not supported for constrained mixer")
        
        grovers_alg(qarg,
                    constraint_oracle,
                    exact = True,
                    winner_state_amount = winner_state_amount(qarg))
        
        
    def inv_prep_psi(qarg):

        with invert():
            prep_psi(qarg)
            
    def constrained_mixer(qarg, beta):

        with conjugate(inv_prep_psi)(qarg):
            mcp(beta, qarg, ctrl_state = 0)
    
    return constrained_mixer
    

def controlled_RX_mixer_gen(predicate):
    r"""
    Generate a controlled RX mixer for a given predicate function.

    Parameters
    ----------
    predicate : function
        A function receiving a ``QuantumVariable`` and an index $i$.
        This function returns a ``QuantumBool`` indicating if the predicate is satisfied for ``qv[i]``,
        that is, if the element ``qv[i]`` should be swapped in. 

    Returns
    -------
    controlled_RX_mixer : function
        A function receiving a ``QuantumVariable`` and a real parameter $\beta$.
        This function performs the application of the mixing operator.

    Examples
    --------

    We define the predicate function for the :ref:`MaxIndepSet <maxIndepSetQAOA>` problem. It returns ``True`` for the index (node) $i$ if 
    all neighbors $j$ of the node $i$ in the graph $G$ are not selected, and ``False`` otherwise.

    ::

        from qrisp import QuantumVariable, QuantumBool, h, mcx, auto_uncompute, multi_measurement
        import networkx as nx

        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])  
        neighbors_dict = {node: list(G.adj[node]) for node in G.nodes()}

        def predicate(qv,i):
            qbl = QuantumBool()
            if len(neighbors_dict[i])==0:
                x(qbl)
            else:
                mcx([qv[j] for j in neighbors_dict[i]],qbl,ctrl_state='0'*len(neighbors_dict[i]))
            return qbl

        qv = QuantumVariable(3)
        h(qv)
        qbl = predicate(qv,0)
        multi_measurement([qv,qbl])
        # Yields: {('000', True): 0.125,('100', True): 0.125,('010', False): 0.125,('110', False): 0.125,('001', False): 0.125,('101', False): 0.125,('011', False): 0.125,('111', False): 0.125}

    The resulting ``controlled_RX_mixer`` then only swaps the node $i$ in if all neighbors $j$ in the graph $G$ are not selected.

    """

    @auto_uncompute
    def controlled_RX_mixer(qv, beta):
        m = qv.size
        for i in range(m):
            with control(predicate(qv,i)):
                rx(beta,qv[i])

    return controlled_RX_mixer


""" from qrisp import as_hamiltonian
@as_hamiltonian
def mcp_as_hamiltonian(qv, beta):
    if qv == "0001":
        p(beta, qv)
    elif qv == "0011":
        p(beta, qv)
    elif qv == "0111":
        p(beta, qv)
    elif qv == "1111":
        p(beta, qv) """

    

#formulate on q_array
def portfolio_mixer():
    """
    Multi-Channel constrained mixer to be applied for a discrete portfolio rebalancing problem, as seen in https://arxiv.org/pdf/2006.00354.pdf.
    This Mixer keeps the constraints in terms of lots on the portfolio intact. This is achieved by mixing between Dicke States.
    

    Returns:
    --------

    apply_mixer : function 
        The Mixer to be applied to a QuantumVariable 

    Examples:
    ---------

    We initiate a QuantumVariable in the "0011" state and from this partially mix into the Dicke state space with Hamming weight 2.

    ::

        from qrisp import QuantumVariable, x
        import numpy as np

        qv = QuantumVariable(4)
        x(qv[2])
        x(qv[3])

        from qrisp.qaoa.mixers import portfolio_mixer
        mixer_op = portfolio_mixer()
        mixer_op(qv, np.pi/8)

    """
    from qrisp.alg_primitives import dicke_state

    def inv_prepare_dicke(qv, k):
        with invert():
            dicke_state(qv, k)

    def apply_mixer(q_array, beta):
        half = int(len(q_array[0]))
        qv1 = q_array[0]
        qv2 = q_array[1]

        #omfg this is harcoded-- problematic one 

        with conjugate(inv_prepare_dicke)(qv1, half):
            # mehrere mcp-gates, as hamiltonian
            #mcp_as_hamiltonian(qv1, beta=beta)
            for i in range(half):
                ctrl_state = "0" * (half-i-1) + ("1"*(i+1))
                #print(ctrl_state)
                mcp(beta, qv1, ctrl_state = ctrl_state)
            """ mcp(beta, qv1, ctrl_state = "0001")
            mcp(beta, qv1, ctrl_state = "0011")
            mcp(beta, qv1, ctrl_state = "0111")
            mcp(beta, qv1, ctrl_state = "1111") """

        with conjugate(inv_prepare_dicke)(qv2, half):
            for i in range(half):
                ctrl_state = "0" * (half-i-1) + ("1"*(i+1))
                #print(ctrl_state)
                mcp(beta, qv2, ctrl_state = ctrl_state)
            """ mcp(beta, qv2, ctrl_state = "0001")
            mcp(beta, qv2, ctrl_state = "0011")
            mcp(beta, qv2, ctrl_state = "0111")
            mcp(beta, qv2, ctrl_state = "1111") """
        
    return apply_mixer





