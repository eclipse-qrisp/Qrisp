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

from qrisp import gphase, rz, cx, conjugate, lifted
from qrisp.operators.qubit.visualization import X_,Y_,Z_

from sympy import Symbol

PAULI_TABLE = {("I","I"):("I",1),("I","X"):("X",1),("I","Y"):("Y",1),("I","Z"):("Z",1),("I","A"):("A",1),("I","C"):("C",1),("I","P0"):("P0",1),("I","P1"):("P1",1),("X","I"):("X",1),("X","X"):("I",1),("X","Y"):("Z",1j),("X","Z"):("Y",(-0-1j)),("X","A"):("P0",1),("X","C"):("P1",1),("X","P0"):("A",1),("X","P1"):("C",1),("Y","I"):("Y",1),("Y","X"):("Z",(-0-1j)),("Y","Y"):("I",1),("Y","Z"):("X",1j),("Y","A"):("P0",(-0-1j)),("Y","C"):("P1",1j),("Y","P0"):("A",1j),("Y","P1"):("C",(-0-1j)),("Z","I"):("Z",1),("Z","X"):("Y",1j),("Z","Y"):("X",(-0-1j)),("Z","Z"):("I",1),("Z","A"):("A",-1),("Z","C"):("C",1),("Z","P0"):("P0",1),("Z","P1"):("P1",-1),("A","I"):("A",1),("A","X"):("P1",1),("A","Y"):("P1",(-0-1j)),("A","Z"):("A",1),("A","A"):("I",0),("A","C"):("P1",1),("A","P0"):("A",1),("A","P1"):("I",0),("C","I"):("C",1),("C","X"):("P0",1),("C","Y"):("P0",1j),("C","Z"):("C",-1),("C","A"):("P0",1),("C","C"):("I",0),("C","P0"):("I",0),("C","P1"):("C",1),("P0","I"):("P0",1),("P0","X"):("C",1),("P0","Y"):("C",(-0-1j)),("P0","Z"):("P0",1),("P0","A"):("I",0),("P0","C"):("C",1),("P0","P0"):("P0",1),("P0","P1"):("I",0),("P1","I"):("P1",1),("P1","X"):("A",1),("P1","Y"):("A",1j),("P1","Z"):("P1",-1),("P1","A"):("A",1),("P1","C"):("I",0),("P1","P0"):("I",0),("P1","P1"):("P1",1)}

#
# QubitTerm
#

class QubitTerm:
    r"""
    
    """

    def __init__(self, pauli_dict={}):
        self.pauli_dict = pauli_dict
        self.hash_value = hash(tuple(sorted(pauli_dict.items())))

    def update(self, update_dict):
        self.pauli_dict.update(update_dict)
        self.hash_value = hash(tuple(sorted(self.pauli_dict.items())))

    def __hash__(self):
        return self.hash_value

    def __eq__(self, other):
        return self.hash_value == other.hash_value
    
    def copy(self):
        return QubitTerm(self.pauli_dict.copy())
    
    def is_identity(self):
        return len(self.pauli_dict)==0
    
    #
    # Simulation
    #
    
    # Assume that the operator is diagonal after change of basis
    # Implements exp(i*coeff*\prod_j Z_j) where the product goes over all indices j in self.pauli_dict
    # @lifted
    def simulate(self, coeff, qarg):

        def parity(qarg, indices):
            n = len(indices)
            for i in range(n-1):
                cx(qarg[indices[i]],qarg[indices[i+1]])

        if not self.is_identity():
            indices = list(self.pauli_dict.keys())
            with conjugate(parity)(qarg, indices):
                rz(-2*coeff,qarg[indices[-1]])
        else:
            gphase(coeff,qarg[0])

    # @lifted
    def simulate(self, coeff, qv):

        from qrisp import h, cx, rz, mcp, conjugate, control, QuantumBool, mcx, x, p, QuantumEnvironment, gphase
        
        Z_indices = []
        ladder_indices = []
        is_creator_list = []
        projector_indices = []
        projector_state = []
        
        pauli_dict = self.pauli_dict
        for i in pauli_dict.keys():
            if pauli_dict[i] in ["X", "Y", "Z"]:
                Z_indices.append(i)
            elif pauli_dict[i] == "A":
                ladder_indices.append(i)
                is_creator_list.append(False)
            elif pauli_dict[i] == "C":
                ladder_indices.append(i)
                is_creator_list.append(True)
            elif pauli_dict[i] == "P0":
                projector_indices.append(i)
                projector_state.append(False)
            elif pauli_dict[i] == "P1":
                projector_indices.append(i)
                projector_state.append(True)
        
        
        if len(Z_indices + ladder_indices + projector_indices) == 0:
            gphase(coeff, qv[0])
            return
        elif len(Z_indices + ladder_indices) == 0:
            
            flip_qubits = [qv[projector_indices[i]] for i in range(len(projector_indices)) if not projector_state[i]]
            
            if len(flip_qubits) == 0:
                env = QuantumEnvironment()
            else:
                env = conjugate(x)(flip_qubits)
            
            with env:
                
                if len(projector_indices) == 1:
                    p(coeff, qv[projector_indices[0]])
                else:
                    mcp(coeff, [qv[i] for i in projector_indices])
                
            return
        # Some hamiltonians contain terms of the for a(1)*c(1), ie.
        # two ladder operators, which operate on the same qubit.
        # We filter them out and discuss their precise treatment below
        

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
            if ladder_ctrl_state[-1] == "1":
                x(qb_list[-1])
            for qb in qb_list[:-1]:
                cx(qb_list[-1], qb)
            if ladder_ctrl_state[-1] == "1":
                x(qb_list[-1])
            h(qb_list[-1])
            
        # Determine ctrl state and the qubits the creation/annihilation
        # operators act on
        ladder_ctrl_state = ""
        ladder_qubits = []
        for i in range(len(ladder_indices)):
            ladder_ctrl_state += str(int(not is_creator_list[i]))
            ladder_qubits.append(qv[ladder_indices[i]])
            
        # The qubit that receives the RZ gate will be called anchor qubit.
        anchor_index = (Z_indices + ladder_indices)[-1]
        
        if len(ladder_qubits) == 0:
            env = QuantumEnvironment()
        else:
            env = conjugate(inv_ghz_state)(ladder_qubits)
                
        with env:
            
            # To realize the behavior of the "double_indices" i.e. the qubits
            # that receive two creation annihilation operators, not that such a
            # term produces a hamiltonian of the form
            
            # H = c(0)*a(0)*a(1)*a(2) + h.c.
            #   = |1><1| (H_red + h.c.)
            
            # where H_red = a(1)*a(2)
            
            # Simulating this H is therefore the controlled version of H_red:
                
            # exp(itH) = |0><0| ID + |1><1| exp(itH_red)
            
            # We can therefore add the "double_indices" to this conjugation.
            
            projector_ctrl_state = ""
            projector_qubits = []
            
            
            for i in range(len(projector_indices)):
                projector_ctrl_state += str(int(projector_state[i]))
                projector_qubits.append(qv[projector_indices[i]])
            
            
            control_qubit_available = False
            
            if len(ladder_indices + projector_qubits) == 0:
                env = QuantumEnvironment()
            elif len(ladder_indices) == 1 and len(projector_indices) == 0:
                env = QuantumEnvironment()
            elif len(ladder_indices) == 2 and len(projector_indices) == 0:
                hs_ancilla = qv[ladder_indices[0]]
                control_qubit_available = True
                if ladder_ctrl_state[0] == "0":
                    env = conjugate(x)(hs_ancilla)
                else:
                    env = QuantumEnvironment()
            else:
                # We furthermore allocate an ancillae to perform an efficient
                # multi controlled rz.
                hs_ancilla = QuantumBool()
                control_qubit_available = True
                
                env = conjugate(mcx)(ladder_qubits[:-1] + projector_qubits, 
                                    hs_ancilla, 
                                    ctrl_state = ladder_ctrl_state[:-1] + projector_ctrl_state, 
                                    method = "gray")
            
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
                def flip_anchor_qubit(qv, anchor_index, Z_indices):
                    for i in Z_indices:
                        if i != anchor_index:
                            cx(qv[i], qv[anchor_index])
                # Perform the conjugation
                with conjugate(flip_anchor_qubit)(qv, anchor_index, Z_indices):
                    
                    if control_qubit_available:
                        env = control(hs_ancilla)
                    else:
                        env = QuantumEnvironment()
                    
                    if len(ladder_indices) == 0:
                        coeff *= 2
                    # Perform the controlled RZ
                    with env:
                        rz(-coeff, qv[anchor_index])
        
        if len(ladder_indices) > 2:
            # Delete ancilla
            hs_ancilla.delete()

    #
    # Printing
    #

    def __str__(self):
        # Convert the sympy expression to a string and return it
        expr = self.to_expr()
        return str(expr)
    
    def __repr__(self):
        return str(self)
    
    def non_trivial_indices(self):
        res = set()
        for index, P in self.pauli_dict.items():
            res.add(index)
        return res

    def to_expr(self):
        """
        Returns a SymPy expression representing the QubitTerm.

        Returns
        -------
        expr : sympy.expr
            A SymPy expression representing the QubitTerm.

        """

        def to_spin(P, index):
            if P=="I":
                return 1
            if P=="X":
                return X_(index)
            if P=="Y":
                return Y_(index)
            if P=="Z":
                return Z_(index)
            if P=="A":
                return Symbol("A_" + str(index))
            if P=="C":
                return Symbol("C_" + str(index))
            if P=="P0":
                return Symbol("P0_" + str(index))
            if P=="P1":
                return Symbol("P1_" + str(index))
        
        expr = 1
        for index,P in self.pauli_dict.items():
            expr *= to_spin(P,str(index))

        return expr

    #
    # Arithmetic
    #

    def __pow__(self, e):
        if isinstance(e, int) and e>=0:
            if e%2==0:
                return QubitTerm({():1})
            else:
                return self
        else:
            raise TypeError("Unsupported operand type(s) for ** or pow(): "+str(type(self))+" and "+str(type(e)))

    def __mul__(self, other):
        result_pauli_dict={}
        result_coeff = 1
        a = self.pauli_dict
        b = other.pauli_dict

        keys = set(a.keys()) | set(b.keys())
        for key in keys:
            pauli, coeff = PAULI_TABLE[a.get(key,"I"),b.get(key,"I")]
            if pauli!="I":
                result_pauli_dict[key]=pauli
                result_coeff *= coeff
        return QubitTerm(result_pauli_dict), result_coeff
    
    def subs(self, subs_dict):
        """
        
        Parameters
        ----------
        subs_dict : dict
            A dictionary with indices (int) as keys and numbers (int, float, complex) as values.

        Returns
        -------
        QubitTerm
            The resulting QubitTerm.
        result_coeff : int, float, complex
            The resulting coefficient.
        
        """
        result_pauli_dict=self.pauli_dict.copy()
        result_coeff = 1

        for key, value in subs_dict.items():
            if key in result_pauli_dict:
                del result_pauli_dict[key]
                result_coeff *= value

        return QubitTerm(result_pauli_dict), result_coeff
    
    #
    # Commutativity checks
    #

    def commute(self, other):
        """
        Checks if two QubitTerms commute.

        """
        a = self.pauli_dict
        b = other.pauli_dict

        keys = set()
        keys.update(set(a.keys()))
        keys.update(set(b.keys()))

        # Count non-commuting Pauli operators
        commute = True

        for key in keys:
            if a.get(key,"I")!="I" and b.get(key,"I")!="I" and a.get(key,"I")!=b.get(key,"I"):
                commute = not commute
        return commute

    def commute_qw(self, other):
        """
        Checks if two QubitTerms commute qubit-wise.

        """
        a = self.pauli_dict
        b = other.pauli_dict

        keys = set()
        keys.update(set(a.keys()))
        keys.update(set(b.keys()))

        for key in keys:
            if a.get(key,"I")!="I" and b.get(key,"I")!="I" and a.get(key,"I")!=b.get(key,"I"):
                return False
        return True
    

