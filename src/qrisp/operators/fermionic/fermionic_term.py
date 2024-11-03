"""
\********************************************************************************
* Copyright (c) 2024 the Qrisp authors
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

from qrisp.operators.fermionic.visualization import a_,c_
from qrisp.operators.pauli import A,C,Z

#
# FermionicTerm
#

import numpy as np

class FermionicTerm:
    r"""
    
    """

    def __init__(self, ladder_list=[]):
        
        self.ladder_list = ladder_list
        
        # Compute the hash value such that 
        # terms receive the same hash as their hermitean conjugate
        # this way the FermionicHamiltonian does not have
        # to track both the term and it's dagger
        index_list = [index for index, is_creator in ladder_list]
        is_creator_hash = 0
        for i in range(len(ladder_list)):
            is_creator_hash += ladder_list[i][1]*2**i
        
        self.hash_value = hash(tuple(index_list + [is_creator_hash]))

    #def update(self, update_dict):
    #    self.pauli_dict.update(update_dict)
    #    self.hash_value = hash(tuple(sorted(self.pauli_dict.items())))

    def __hash__(self):
        return self.hash_value

    def __eq__(self, other):
        return self.hash_value == other.hash_value
    
    def copy(self):
        return FermionicTerm(self.ladder_list.copy())
    
    def dagger(self):
        return FermionicTerm([(index, not is_creator) for index, is_creator in self.ladder_list[::-1]])
    #
    # Printing
    #

    def __str__(self):
        # Convert the sympy expression to a string and return it
        expr = self.to_expr()
        return str(expr)
    
    def __repr__(self):
        return str(self)

    def to_expr(self):
        """
        Returns a SymPy expression representing the FermionicTerm.

        Returns
        -------
        expr : sympy.expr
            A SymPy expression representing the FermionicTerm.

        """

        def to_ladder(value, index):
            if value:
                return c_(index)
            else:
                return a_(index)
        
        expr = 1
        for index,value in self.ladder_list[::-1]:
            expr *= to_ladder(value,str(index))

        return expr

    #
    # Arithmetic
    #

    def __mul__(self, other):
        result_ladder_list = other.ladder_list + self.ladder_list
        return FermionicTerm(result_ladder_list)
    
    def order(self):
        """
        Not that important, since relevant Hamiltonians (e.g., electronic structure) consist of ordered terms.
        What is needed for trotterization?

        Fermionic commutation relations:

        {a_i,a_j^dagger} = a_i*a_j^dagger + a_j^dagger*a_i = delta_{ij}
        {a_i^dagger,a_j^dagger} = {a_i,a_j} = 0


        Order ladder terms such that 
            1) Raising operators preceed lowering operators
            2) Operators are ordered in descending order of fermionic modes

        Example: a_5^dagger a_2^dagger a_3 a_1

        """
        pass
    
    
    def simulate(self, coeff, qv):
        
        from qrisp import h, cx, rz, conjugate, control, QuantumBool, mcx, x, p, QuantumEnvironment, gphase
        
        sorted_term, flip_sign = self.sort()
        
        coeff *= flip_sign
        
        ladder_list = sorted_term.ladder_list
        
        active_indices = [index for index, is_creator in ladder_list]
        
        # Some hamiltonians contain terms of the for a(1)*c(1), ie.
        # two ladder operators, which operate on the same qubit.
        # We filter them out and discuss their precise treatment below
        
        active_indices = []
        active_index_is_creator = []
        
        double_indices = []
        double_index_is_creator = []
        i = 1
        
        for i in range(len(ladder_list)):
            
            ladder_op = ladder_list[i]
            ladder_index = ladder_op[0]
            is_creator = ladder_op[1]
            
            if i > 0 and ladder_index == ladder_list[i-1][0]:
                double_indices.append(active_indices.pop(-1))
                double_index_is_creator.append(ladder_list[i-1][1])
                continue
            
            active_indices.append(ladder_index)
            active_index_is_creator.append(is_creator)
        
        if len(active_indices) == 0 and len(double_indices) == 0:
            gphase(coeff)
            return
        elif len(active_indices) == 0:
            for i in range(len(double_indices)):
                if double_index_is_creator[i]:
                    x(qv[double_indices[i]])
                p(coeff, qv[double_indices[i]])
                if double_index_is_creator[i]:
                    x(qv[double_indices[i]])
            return
        
        
        # In the Jordan-Wigner transform annihilation/creation operators are
        # represented as operators of the form
        
        # ZZZZA111
        
        # or
        
        # ZZC11111
        
        # Where Z is the Z Operator, A/C are creation/annihilation operators and
        # 1 is the identity.
        
        # We now are given a list of these types of operators.
        # The first step is now to identify on which qubits there are actually
        # Z operators acting and where they cancel out.
        
        # We do this by creating an array that operates under boolean arithmetic        
        Z_qubits = np.zeros(qv.size)
        
        
        for i in active_indices:
            # This array contains the Z Operators, which need to be executed.
            Z_incr = np.zeros(qv.size)
            Z_incr[:i] = 1
            
            # Update the overall tracker of Z gates
            Z_qubits = (Z_qubits + Z_incr)%2
        
        # We can assume the ladder_list is sorted, so every Z operator
        # that acts on an A or C is acting from the left.
        # We have and Z*C = -C and Z*A = A 
        # because of C = |1><0| and A = |0><1|
        
        # Therefore we flip the sign of the coefficient for every creator
        # that has an overall Z acting on it.
        
        for i in range(len(ladder_list)):
            if Z_qubits[ladder_list[i][0]] == 1:
                if ladder_list[i][1]:
                    coeff *= -1
                Z_qubits[ladder_list[i][0]] = 0
        
        

        # We now start implementing performing the quantum operations
                
        # There are three challenges-
        
        # 1. Implement the "double_indices" i.e. creation/annihilation
        # operators where two act on the same qubit.
        # 2. Implement the other creation annihilation operators.
        # 3. Implement the Z operators.
        
        # For step 2 we recreate the circuit in https://arxiv.org/abs/2310.12256
        
        # The circuit on page 4 looks like this
        
        #               ┌───┐                                                          »
        #         qv.0: ┤ X ├─────────────────■────────────────────────────────■───────»
        #               └─┬─┘┌───┐            │                                │       »
        #         qv.1: ──┼──┤ X ├────────────■────────────────────────────────■───────»
        #                 │  └─┬─┘┌───┐       │                                │       »
        #         qv.2: ──┼────┼──┤ X ├───────o────■──────────────────────■────o───────»
        #                 │    │  └─┬─┘┌───┐  │  ┌─┴─┐┌────────────────┐┌─┴─┐  │  ┌───┐»
        #         qv.3: ──■────■────■──┤ H ├──┼──┤ X ├┤ Rz(-0.5*theta) ├┤ X ├──┼──┤ H ├»
        #                              └───┘┌─┴─┐└───┘└───────┬────────┘└───┘┌─┴─┐└───┘»
        # hs_ancilla.0: ────────────────────┤ X ├─────────────■──────────────┤ X ├─────»
        #                                   └───┘                            └───┘     »
        # «                        ┌───┐
        # «        qv.0: ──────────┤ X ├
        # «                   ┌───┐└─┬─┘
        # «        qv.1: ─────┤ X ├──┼──
        # «              ┌───┐└─┬─┘  │  
        # «        qv.2: ┤ X ├──┼────┼──
        # «              └─┬─┘  │    │  
        # «        qv.3: ──■────■────■──
        # «                             
        # «hs_ancilla.0: ───────────────
        
        # In it's essence this circuit is a conjugation with an inverse GHZ state
        # preparation and a multi controlled RZ gate.

        def inv_ghz_state(qb_list):
            if operator_ctrl_state[-1] == "1":
                x(qb_list[-1])
            for qb in qb_list[:-1]:
                cx(qb_list[-1], qb)
            if operator_ctrl_state[-1] == "1":
                x(qb_list[-1])
            h(qb_list[-1])
            
        # Determine ctrl state and the qubits the creation/annihilation
        # operators act on
        operator_ctrl_state = ""
        operator_qubits = []
        for i in range(len(active_indices)):
            operator_ctrl_state += str(int(active_index_is_creator[i]))
            operator_qubits.append(qv[active_indices[i]])
            
        # The qubit that receives the RZ gate will be called anchor qubit.
        anchor_index = active_indices[-1]
        
                
        with conjugate(inv_ghz_state)(operator_qubits):
            
            # To realize the behavior of the "double_indices" i.e. the qubits
            # that receive two creation annihilation operators, not that such a
            # term produces a hamiltonian of the form
            
            # H = c(0)*a(0)*a(1)*a(2) + h.c.
            #   = |1><1| (H_red + h.c.)
            
            # where H_red = a(1)*a(2)
            
            # Simulating this H is therefore the controlled version of H_red:
                
            # exp(itH) = |0><0| ID + |1><1| exp(itH_red)
            
            # We can therefore add the "double_indices" to this conjugation.
            
            double_index_ctrl_state = ""
            double_index_qubits = []
            for i in range(len(double_index_is_creator)):
                if double_index_is_creator[i]:
                    double_index_ctrl_state += "1"
                else:
                    double_index_ctrl_state += "0"
            
                double_index_qubits.append(qv[double_indices[i]])
                
            if len(active_indices) == 2:
                hs_ancilla = qv[active_indices[0]]
                if operator_ctrl_state[0] == "0":
                    env = conjugate(x)(hs_ancilla)
                else:
                    env = QuantumEnvironment()
            else:
                # We furthermore allocate an ancillae to perform an efficient
                # multi controlled rz.
                hs_ancilla = QuantumBool()
                
                env = conjugate(mcx)(operator_qubits[:-1] + double_index_qubits, 
                                    hs_ancilla, 
                                    ctrl_state = operator_ctrl_state[:-1] + double_index_ctrl_state, 
                                    method = "gray_pt")
            
            with env:
                
                # Before we execute the RZ, we need to deal with the Z terms (next to the anihilation/
                # creation) operators.
                
                # The qubits that only receive a Z operator will now be called "passive qubits"
                
                # By inspection we see that the Z operators are equivalent to the
                # term (-1)**(q_1 (+) q_2)
                # It therefore suffices to compute the parity value of all the passive
                # qubits and flip the sign of the coefficient based on the parity.

                # To this end we perform several CX gate on-to the anchor qubit.
                # Since the anchor qubit will receive an RZ gate, each bitflip will
                # induce a sign flip of the phase.

                # We achieve this by executing a conjugation with the following function
                def flip_anchor_qubit(qv, anchor_index, Z_qubits):
                    for i in range(qv.size):
                        # Z_qubits will contain a 1 if a flip should be executed.
                        if Z_qubits[i]:
                            cx(qv[i], qv[anchor_index])
                
                # Perform the conjugation
                with conjugate(flip_anchor_qubit)(qv, anchor_index, Z_qubits):
                
                    # Perform the controlled RZ
                    with control(hs_ancilla):
                            rz(coeff, qv[anchor_index])
        
        if len(active_indices) != 2:
            # Delete ancilla
            hs_ancilla.delete()
        
    def sort(self):
        # Sort ladder operators (ladder operator semantics are order in-dependent)
        sorting_list = [index for index, is_creator in self.ladder_list]
        perm = np.argsort(sorting_list, kind = "stable")
        ladder_list = [self.ladder_list[i] for i in perm]
        
        return FermionicTerm(ladder_list), permutation_signature(perm)
    
    def fermionic_swap(self, permutation):
        
        new_ladder_list = [(permutation[i], is_creator) for i, is_creator in self.ladder_list]
        
        return FermionicTerm(new_ladder_list)
    
    def to_JW(self):
        res = 1
        for i in range(len(self.ladder_list)):
            temp = 1
            for j in range(self.ladder_list[i][0]):
                temp = Z(j)*temp
            
            if self.ladder_list[i][1]:
                temp = temp*C(self.ladder_list[i][0])
            else:
                temp = temp*A(self.ladder_list[i][0])
        
            res = temp*res
            
        return res
            
        
        

def permutation_signature(perm):
    
    k = 0
    for i in range(len(perm)):
        for j in range(i):
            k += int(perm[i] < perm[j])
            
    return (-1)**(k%2)
            

            
            
            
        
        
    
