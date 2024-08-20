"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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


# This file implements the carry generation art of the algorithm presented here:
# https://arxiv.org/abs/2304.02921

# However instead of two quantum inputs, this file deals with the case of one quantum
# and one classical input.

# For two quantum inputs, see qrisp.alg_primitives.arithmetic.adders.qcla.carry_path

# Many functions and ideas are adapted from this file, so to get an understanding
# for this file, we recommend going through the quantum-quantum file first


# We implement the Brent-Kung-Tree in a recursive manner. See:
# https://web.stanford.edu/class/archive/ee/ee371/ee371.1066/lectures/lect_04.pdf Page 13

import numpy as np

from qrisp.alg_primitives import x, cx, mcx, barrier
from qrisp.qtypes import QuantumVariable, QuantumBool
from qrisp.circuit import QuantumCircuit
from qrisp.permeability import auto_uncompute
from qrisp.misc.utility import bin_rep, check_if_fresh
from qrisp.alg_primitives.mcx_algs import hybrid_mcx


# Returns the PROPAGATE status of a group of entries
def calc_P_group(P):
    new_p = QuantumBool(name = "p_group*", qs = P[0].qs())
    # Due to the semi-classical nature of the algorithm, it is possible
    # that some propagate values are known to be 0 (because some of the 
    # input values are known to be 0)
    # The following loop checks if they are |0> and if so, the corresponding
    # mcx gate is skipped
    for p in P:
        if isinstance(p, QuantumBool):
            p = p[0]
        if check_if_fresh([p], p.qs()):
            return new_p
    if len(P) == 2:
        mcx(P, new_p, method = "gidney")
    else:
        hybrid_mcx(P, new_p)
    return new_p[0]
    

# Returns the GENERATE status of a group of entries
# A group has GENERATE status True, if any of the members
# has GENERATE status True and all of the following members
# have PROPAGATE status True.

# The GENERATE status of the group is calculated into the last
# member of the group (G[-1])
def calc_G_group(P, G):
    
    for i in range(len(G)-1):
        
        controls = [G[i]] + P[i+1:]
        for p in controls:
            if isinstance(p, QuantumBool):
                p = p[0]
            if check_if_fresh([p], p.qs()):
                break
        else:
            
            if len(P[i+1:]) == 1:
                mcx(controls, G[-1], method = "jones")
            else:
                hybrid_mcx(controls, G[-1])
        
    return G[-1]

# Receives the values PROPAGATE and GENERATE values of a group.
# It is assumed that the 0-th entry of G already contains the carry status
# This function then calculates the carry status for every other entry of G
def propagate_carry(P, G):
    
    # This loop iteratively calculates the CARRY status into all entries
    # of G.
    
    # We calculate the CARRY status of position i by considering the CARRY status of
    # position 0 and the GENERATE status of any position above that. To successfully
    # generate a CARRY status of True, all the other positions in between need to have
    # PROPAGATE status True.
    
    # Note that in order to calculate the CARRY status of position i, we need the
    # GENERATE status (not the CARRY status) of j for all j < i.
    # This is why we reverse the loop, such that the position with the highest i
    # are calculated first (the computation can access all the GENERATE 
    # information because none of the CARRY status has been calculated yet)
    
    
    for i in range(1, len(G))[::-1]:
        
        for j in range(i):
            if len(P[j+1:i+1]) == 1:
                method = "jones"
            else:
                method = "auto"
            
            controls = [G[j]] + P[j+1:i+1]
            for p in controls:
                if isinstance(p, QuantumBool):
                    p = p[0]
                if check_if_fresh([p], p.qs()):
                    break
            else:
                if len(P[j+1:i+1]) == 1:
                    mcx(controls, G[i], method = method)
                else:
                    hybrid_mcx(controls, G[i])
                

        

# This function recursively calculates the layers of the Brent-Kung tree.
# It receives the GENERATE and CARRY status of a set of positions and returns
# the evaluated CARRY status

# The function takes the parameters P and G, which represent the PROPAGATE and
# the GENERATE status of the corresponding digit.

# The parameter r describes the radix of the Brent-Kung tree.

# Finally the parameter k_out describes a the layer where the carry calculation
# should be ended prematurely. That is if k_out = 2 and r = 4, the output will
# be every 4**2 = 16-th bit of G, that contains a properly calculated carry value
def brent_kung_tree(P, G, r, k_out = np.inf):
    
    if r <= 1 or not isinstance(r, int):
        raise Exception("Provided invalid radix base (needs to be integer > 1)")
    
    # We group the positions and calculate the GENERATE and PROPAGATE status
    # of the groups. Subsequently the function is called recursively to evaluate the
    # CARRY status of the respective groups.
    
    # Finally the CARRY status of the groups is propagated to the remaining positions
    # to yield the complete set of carries (not only the groups)    
    
    # Recursion cancellation condition
    if len(G) == 0:
        return []

    # Set up list for grouped GENERATE and PROPAGATE states
    grouped_P = []
    grouped_G = []
    for i in range(len(G)//r):
        interval = slice(r*i,r*(i+1))
        
        if i == 0:
            # We can skip the 0-th position
            # This is because the propagate_carry function and the calc_G_group
            # function are not taking the 0-th position into consideration
            grouped_P.append(None)
        else:
            grouped_P.append(calc_P_group(P[interval]))
            
        grouped_G.append(calc_G_group(P[interval], G[interval]))
    
    # Call the function recursively
    grouped_carry = brent_kung_tree(grouped_P, grouped_G, r, k_out = k_out - 1)
    
    if k_out > 0:
        return grouped_carry
    
    # If working with a radix > 2, the carry also needs to be propagated
    # to the position of the first group. In the case of r = 2, this is not neccessary
    # because the first group consists just of the 0-th position (which already contains 
    # the carry in its GENERATE status) and the 1st position which contains the CARRY status
    # due to the GENERATE computation of this group.
    initial_interval = slice(0, r-1)
    propagate_carry(P[initial_interval], G[initial_interval])

    # Go through all the groups and propagate the carry.
    #Note that the last interval has not neccessarily size r (depending on the size of G)
    for i in range(1, len(G)//r+1):
        interval = slice(r*i-1, r*(i+1)-1)
        propagate_carry(P[interval], G[interval])
    
    # Return G (now contains the carry positions)
    return G


# The following function calculates the carry status a QuantumVariable and a classical
# integer

# The function takes a classical integer a, QuantumVariable b, the radix base and the radix exponent.
# The function returns a QuantumVariable c, that contains the CARRY status of every
# radix_base**radix_exponent digit of CARRY(a,b)

# Example:
    
# a = 3 = 0011
# b = 6 = 0110
# CARRY(a,b) = 0110

# Assume we choose radix_base = 2. If we choose radix_exponent = 0,
# this function returns

# c = 0110

#If we choose radix_exponent = 1

# c = 01

# ie. position 1 and 3 of CARRY(a,b)

# If we chose radix_exponent = 2

# c = 0

# ie. position 3 of CARRY(a,b)


# Furthermore this function uncomputes all garbage. What is the garbage here?
# The cancelation of the brent kung tree at an early layer produces many GENERATE
# entries, that are not holding CARRY values. Furthermore, also the PROPAGATE values
# of groups are uncomputed.

@auto_uncompute
def cq_calc_carry(a, b, radix_base = 2, radix_exponent = 0, ctrl = None):
    
    # Convert the classical integer into a bitstring
    if isinstance(a, int):
        a = bin_rep(a, len(b))[::-1]
    elif not isinstance(a, str):
        raise Exception(f"Tried to call semi-classical carry calculator with invalid type {type(a)}")
    
    R = radix_base**radix_exponent
    # How can we achieve that the GENERATE entries, that don't contain a relevant
    # CARRY information are uncomputed while the CARRY entries stay?
    
    # We do this by creating two QuantumVariables, one which holds the carry and
    # one which holds the GENERATE garbage. We then arrange both of them in a list g,
    # that represents the GENERATE array.
    
    # This variable will hold the result
    # If b can be divided into k blocks of size R,
    # we only need k-1 ancillae qubit, because we have no need for
    # the carry of the last bock.
    c = QuantumVariable(int(np.ceil(len(b)/R))-1, name = "carry*", qs = b[0].qs())
    
    # This variable will hold the intermediate GENERATE values, that are supposed 
    # to be uncomputed. The uncomputation is performed using the auto_uncompute 
    # decorator. This decorator uncomputes all local variables.
    if R > 1:
        brent_kung_ancilla = QuantumVariable(c.size*(R-1), name = "bk_ancilla*", qs = b[0].qs())
        anc_list = list(brent_kung_ancilla)
    else:
        anc_list = []
    
    #Create the g list
    c_list = list(c)
    g = []
    for i in range(len(c)):
        for j in range(R-1):
            g.append(anc_list.pop(0))
        g.append(c_list.pop(0))
    
    # Append the remaining ancillae
    g.extend(anc_list)
    
    # We now compute the GENERATE and the PROPAGATE status of each digit
    # g_i = a_i * b_i
    # p_i = a_i XOR b_i
    
    # Since a_i is a classical value that is known to us, the following loop will
    # only execute the iterations where a_i is True
    
    use_parallel = False
    
    if not ctrl is None:
        if sum(k == "1" for k in a) > 1:
            parallel_anc_var = QuantumVariable(sum(k == "1" for k in a), name = "parll_qbl*", qs = b[0].qs())
            parallel_ancillae = list(parallel_anc_var)
    
    
    for i in range(min(len(g), len(a), len(b))):
        
        if a[i] == "1":
            
            
            # To get p_i = a_i XOR b_i we can simply flip b_i (because we know
            # that a_i = 1)
            x(b[i])
            
            # We now need to compute
            # g_i = a_i * b_i
            # Since we flipped b_i already, we need to adjust the control of the mcx gates.
            # We did this because using this order, there is no cycle in the uncomputation
            # DAG.
            
            # In the case there is no control qubit given, we can simply execute
            # a cnot gate from b_i to g_i to get g_i = a_i * b_i
            if ctrl is None:
                mcx([b[i]], g[i], ctrl_state = "0")
            
            # In the case of the a control qubit it is sufficient to set all 
            # GENERATE entries to 0 if the control qubit is in the |0> state.
            # This is achieved by deploying a x gate controlled by
            # the control qubit and b_i.
            else:
                # Instead of this command:
                
                if not use_parallel:
                    mcx([ctrl, b[i]], g[i], method = "gidney", ctrl_state = "10")
                    use_parallel = True
                else:
                    # To achieve further parallelization, we "copy" the value of
                    # the control value.
                    # The permutation of the controls  that is necessary for 
                    # actual parallelization will be done by the the compiler.
                    # parll_qbl = QuantumBool(name = "parll_qbl*", qs = b[0].qs())
                    parll_qbl = parallel_ancillae.pop(0)
                    cx(ctrl, parll_qbl)
                    mcx([parll_qbl, b[i]], g[i], method = "gidney", ctrl_state = "10")
                    # parll_qbl.uncompute(recompute = True)
    
    try:
        parallel_anc_var.uncompute(recompute = True)
    except UnboundLocalError:
        pass
    
    p = b
    
    # Calculate compute the carry using the Brent-Kung tree
    brent_kung_tree(p, g, radix_base, radix_exponent)

    # Undo the previous x to keep b unchanged
    for i in range(min(len(g), len(a), len(b))):
        if a[i] == "1":
            x(b[i])

    # Return the result
    return c

# This function returns a gate, that performs the toffoli and cx gate. 
# We mark the second qubit as permeable to "hack" the uncomputation algorithm.
def anc_gate():
    from qrisp.alg_primitives.mcx_algs import GidneyLogicalAND
    anc_qc = QuantumCircuit(3)
    anc_qc.append(GidneyLogicalAND(), [0,1,2])
    anc_qc.cx(0,1)
    res = anc_qc.to_gate(name = "anc_gate")
    res.permeability = {0 : True, 1: True, 2: False}
    res.is_qfree = True
    return res