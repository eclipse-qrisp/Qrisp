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

# This file implements the Gidney-Adder https://arxiv.org/abs/1709.06648
# but in a semi-classical manner. That is, the adder accepts one classical
# and one quantum input.

# For the quantum-quantum version please check qrisp.alg_primitives.arithmetic.adders.gidney_adder


from qrisp.qtypes import QuantumBool
from qrisp.core import QuantumVariable, merge
from qrisp.misc.utility import bin_rep
from qrisp.core.gate_application_functions import x, cx, mcx
from qrisp.circuit import fast_append
from qrisp.environments import invert


# This function inverts a bit represented by a character
def c_inv(c):
    if c == "0":
        return "1"
    elif c == "1":
        return "0"
    else:
        raise Exception

# This function perform semi-classical Gidney-Addition

def cq_gidney_adder(a, b, c_in = None, c_out = None, ctrl = None):
    
    # Call merge to make the QuantumSession enter the inversion environment
    # (doesn't happen automatically with fast_append = 3)
    
    qs_list = [b[0].qs()]
    
    for qb in [c_in, c_out, ctrl]:
        if qb is not None:
            qs_list.append(qb.qs())
    
    merge(qs_list)
    
    with fast_append(3):
    
        if isinstance(a, int):
            a = bin_rep(a%2**len(b), len(b))[::-1]
        elif not isinstance(a, str):
            raise Exception(f"Tried to call semi-classical carry calculator with invalid type {type(a)}")
        
        if len(a) != len(b):
            raise Exception("Tried to call Gidney adder with inputs of unequal length")
    
        if c_out is not None:
            
            # Convert to qubit if neccessary
            if isinstance(c_out, QuantumBool):
                c_out = c_out[0]
            
            b = list(b) + [c_out]
            a = a + "0"
            
        if c_in is not None:
            
            # Convert to qubit if neccessary
            if isinstance(c_in, QuantumBool):
                c_in = c_in[0]
            
            b = [c_in] + list(b)
            a = "1" + a
            
        #Treat the case that b is only a single qubit
        if len(b) == 1:
            if a[0] == "1":
                if ctrl is None:
                    x(b[0])
                else:
                    cx(ctrl, b[0])
            return
        
        gidney_anc = QuantumVariable(len(b) - 1, name = "gidney_anc*", qs = b[0].qs())
        
        # This loop now perform the iterations that are required to build up the circuit described
        # in the paper
        for i in range(len(b)-1):
            
            # The relevant observation to turn Gidney's adder into a semi-classical adder
            # is that the "a_i" qubit (in his paper these qubits are called "i") 
            # is only involved in two interactions:
            #   1. The CNOT gate from the ancilla qubit of the previous iteration
            #   2. The the quasi-toffoli gate into the ancilla of the current interaction
            # If we consider "a_i" as a classically known value, we can recover the quantum
            # value of the first step by applying an x gate onto the ancilla of the previous 
            # iteration if a_i is True.
            
            # To "simulate" the quasi-toffoli we then simply use this ancilla as a control
            # value instead.
            
            # Combining these steps results in a quasi-toffoli with a modified control-state
            # If a is True, the quasi-toffoli receives the flipped value of the ancilla
            if i != 0:
                
                if ctrl is not None:
                    if "1" == a[i]:
                        cx(ctrl, gidney_anc[i-1])
                    mcx([gidney_anc[i-1], b[i]], gidney_anc[i], method = "gidney", ctrl_state = "11")
                    if "1" == a[i]:
                        cx(ctrl, gidney_anc[i-1])
                else:
                    mcx([gidney_anc[i-1], b[i]], gidney_anc[i], method = "gidney", ctrl_state = c_inv(a[i]) + "1")
                    
                cx(gidney_anc[i-1], gidney_anc[i])
                
            else:
                # For the first iteration, we can simply execute a CNOT gate if a_0 is True
                if a[i] == "1":
                    if ctrl is None:
                        cx(b[i], gidney_anc[i])
                    else:
                        mcx([ctrl, b[i]], gidney_anc[i], method = "gidney")
            
            
            # To realize the controlled version of this adder, note that the V-shaped
            # part of the circuit in the paper is consisting of two branches that are
            # almost inverse to each other.
            
            # The one thing that stops the from being truly inverse is the CNOT gate
            # from the ancilla into b (in his paper "t").
            
            # That is: If we only control this gate (and the "tip"), the two branches will be their
            # perfect inverse and the whole computation will result in the identity
            # if the control qubit is in the |0> state.
            if i != len(b) -2:
                cx(gidney_anc[i], b[i+1])
    
        # This snippet makes sure that the "tip" of the V shaped circuit is also
        # controlled
        cx(gidney_anc[-1], b[-1])
    
        # We now perform the right branch of the V shaped circuit
        
        with invert():
            
            # Call merge to make the QuantumSession enter the inversion environment
            # (doesn't happen automatically with fast_append = 3)
            merge(b[0].qs())
            
            for i in range(len(b)-1):
                
                # Regarding the treatment of the classical values, a similar logic as
                # above is applied
                if i != 0:
                    if ctrl is not None:
                        
                        if "1" == a[i]:
                            cx(ctrl, gidney_anc[i-1])
                        mcx([gidney_anc[i-1], b[i]], gidney_anc[i], method = "gidney", ctrl_state = "11")
                        if "1" == a[i]:
                            cx(ctrl, gidney_anc[i-1])
                    
                    else:
                        mcx([gidney_anc[i-1], b[i]], gidney_anc[i], method = "gidney", ctrl_state = c_inv(a[i]) + "1")
                        
                    cx(gidney_anc[i-1], gidney_anc[i])
                    
                else:
                    if a[i] == "1":
                        if ctrl is None:
                            cx(b[i], gidney_anc[i])
                        else:
                            mcx([ctrl, b[i]], gidney_anc[i], method = "gidney")
        
        # Finally, the CNOT gates on the right side of the V shape are executed.
        
        for i in range(len(a)):
            # Since the values of a (in his paper "i") are classically known we
            # can use a classical condition
            if a[i] == "1":
                
                if c_in is not None and i == 0:
                    continue
                # Here we need to again treat the control qubit, because in the 
                # controlled case, the V shaped part of the circuit might be properly
                # controlled because of the above construction but the right most
                # CNOT gates are not.
                if ctrl is None:
                    x(b[i])
                else:
                    cx(ctrl, b[i])
        
        gidney_anc.delete(verify = False)