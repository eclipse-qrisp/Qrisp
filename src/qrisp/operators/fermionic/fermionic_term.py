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
        is_creator_temp_hash = 0
        for i in range(len(ladder_list)):
            is_creator_temp_hash += ladder_list[i][1]*2**i
        is_creator_temp_hash_dg = 0
        # for i in range(len(ladder_list)):
        #     is_creator_temp_hash_dg += (not ladder_list[::-1][i][1])*2**i
        
        is_creator_hash = is_creator_temp_hash + is_creator_temp_hash_dg
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
        
        from qrisp import h, cx, rz, conjugate, control, QuantumBool, mcx, gphase
        
        sorted_term, flip_sign = self.sort()
        
        coeff *= flip_sign
        
        ladder_list = sorted_term.ladder_list
        
        active_indices = [index for index, is_creator in ladder_list]
        
        
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
        
        for i in active_indices:
            if Z_qubits[i] == 1:
                if ladder_list[i][1]:
                    coeff *= -1
                    Z_qubits[i] = 0
        
        

        # We now implement an operator like
        
        # 5*AZZC11A
        
        # To this end we split the procedure in two steps:
        
        # 1. Implement the creation annihilation operators.
        # 2. Implement the Z operators.
        
        # For step 1 we recreate the circuit in https://arxiv.org/abs/2310.12256
        
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
            for qb in qb_list[:-1]:
                cx(qb_list[-1], qb)
            h(qb_list[-1])
            
        # Determine ctrl state and the qubits the creation/annihilation
        # operators act on
        ctrl_state = ""
        operator_qubits = []
        for i in active_indices:
            ctrl_state += str(str(int(ladder_list[i][1])))
            operator_qubits.append(qv[i])
            
        # The qubit that receives the RZ gate will be called anchor qubit.
        anchor_index = active_indices[-1]
        
        # We furthermore allocate an ancillae to perform an efficient
        # multi controlled rz.
        hs_ancilla = QuantumBool()
                
        with conjugate(inv_ghz_state)(operator_qubits):
            with conjugate(mcx)(operator_qubits[:-1], hs_ancilla, ctrl_state = ctrl_state[:-1], method = "gray_pt"):
                
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
                        if ctrl_state[0] == "0":
                            rz(coeff, qv[anchor_index])
                        else:
                            rz(-coeff,qv[anchor_index])
        
        # Delete ancilla
        hs_ancilla.delete()
        
    def sort(self):
        # Sort ladder operators (ladder operator semantics are order in-dependent)
        sorting_list = [index for index, is_creator in self.ladder_list]
        perm = np.argsort(sorting_list, kind = "stable")
        ladder_list = [self.ladder_list[i] for i in perm]
        from sympy.combinatorics import Permutation
        return FermionicTerm(ladder_list), Permutation(perm).signature()

        
                    

                
            
            
            
        
        
    
