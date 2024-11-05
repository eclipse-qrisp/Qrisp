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

from qrisp import gphase, rz, cx, conjugate
from qrisp.operators.qubit.visualization import X_,Y_,Z_

from sympy import Symbol

PAULI_TABLE = {("I","I"):("I",1),("I","X"):("X",1),("I","Y"):("Y",1),("I","Z"):("Z",1),("I","A"):("A",1),("I","C"):("C",1),("I","P0"):("P0",1),("I","P1"):("P1",1),("X","I"):("X",1),("X","X"):("I",1),("X","Y"):("Z",1j),("X","Z"):("Y",(-0-1j)),("X","A"):("P0",1),("X","C"):("P1",1),("X","P0"):("A",1),("X","P1"):("C",1),("Y","I"):("Y",1),("Y","X"):("Z",(-0-1j)),("Y","Y"):("I",1),("Y","Z"):("X",1j),("Y","A"):("P0",(-0-1j)),("Y","C"):("P1",1j),("Y","P0"):("A",1j),("Y","P1"):("C",(-0-1j)),("Z","I"):("Z",1),("Z","X"):("Y",1j),("Z","Y"):("X",(-0-1j)),("Z","Z"):("I",1),("Z","A"):("A",-1),("Z","C"):("C",1),("Z","P0"):("P0",1),("Z","P1"):("P1",-1),("A","I"):("A",1),("A","X"):("P1",1),("A","Y"):("P1",(-0-1j)),("A","Z"):("A",1),("A","A"):("I",0),("A","C"):("P1",1),("A","P0"):("A",1),("A","P1"):("I",0),("C","I"):("C",1),("C","X"):("P0",1),("C","Y"):("P0",1j),("C","Z"):("C",-1),("C","A"):("P0",1),("C","C"):("I",0),("C","P0"):("I",0),("C","P1"):("C",1),("P0","I"):("P0",1),("P0","X"):("C",1),("P0","Y"):("C",(-0-1j)),("P0","Z"):("P0",1),("P0","A"):("I",0),("P0","C"):("C",1),("P0","P0"):("P0",1),("P0","P1"):("I",0),("P1","I"):("P1",1),("P1","X"):("A",1),("P1","Y"):("A",1j),("P1","Z"):("P1",-1),("P1","A"):("A",1),("P1","C"):("I",0),("P1","P0"):("I",0),("P1","P1"):("P1",1)}

#
# QubitTerm
#

class QubitTerm:
    r"""
    
    """

    def __init__(self, factor_dict={}):
        self.factor_dict = factor_dict
        self.hash_value = hash(tuple(sorted(factor_dict.items())))

    def update(self, update_dict):
        self.factor_dict.update(update_dict)
        self.hash_value = hash(tuple(sorted(self.factor_dict.items())))

    def __hash__(self):
        return self.hash_value

    def __eq__(self, other):
        return self.hash_value == other.hash_value
    
    def copy(self):
        return QubitTerm(self.factor_dict.copy())
    
    def is_identity(self):
        return len(self.factor_dict)==0
    
    #
    # Simulation
    #
    
    # Assume that the operator is diagonal after change of basis
    # Implements exp(i*coeff*\prod_j Z_j) where the product goes over all indices j in self.factor_dict
    # @lifted
    def simulate(self, coeff, qarg):

        def parity(qarg, indices):
            n = len(indices)
            for i in range(n-1):
                cx(qarg[indices[i]],qarg[indices[i+1]])

        if not self.is_identity():
            indices = list(self.factor_dict.keys())
            with conjugate(parity)(qarg, indices):
                rz(-2*coeff,qarg[indices[-1]])
        else:
            gphase(coeff,qarg[0])

    # @lifted
    def simulate(self, coeff, qv, do_change_of_basis = True):

        from qrisp import h, cx, rz, mcp, conjugate, control, QuantumBool, mcx, x, p, s, QuantumEnvironment, gphase
        
        # If required, do change of basis.
        
        
        if do_change_of_basis:
            
            def change_of_basis(qv, terms_dict):
                for index, factor in terms_dict.items():
                    if factor=="X":
                        h(qv[index])
                    if factor=="Y":
                        s(qv[index])
                        h(qv[index])
                        x(qv[index])
                        
            with conjugate(change_of_basis)(qv, self.factor_dict):
                self.simulate(coeff, qv, do_change_of_basis=False)
            return
        
        # We group the term into 3 types of components:
        
        # 1. Pauli-indices, i.e. indices where a Pauli-Z is supposed to be applied.
        # 2. Ladder indices, i.e. A or C Operators
        # 3. Projector indices, i.e. P0 or P1
        
        # Contains the indices of Pauli Z
        Z_indices = []
        
        # Contains the indices of the ladder Operators
        ladder_indices = []
        # Contains the booleans, which specify whether A or C
        is_creator_list = []
        
        # Contains the indices of the projection Operators
        projector_indices = []
        # Contains the booleans, which specify whether P0 or P1
        projector_state = []
        
        factor_dict = self.factor_dict
        for i in factor_dict.keys():
            if factor_dict[i] in ["X", "Y", "Z"]:
                Z_indices.append(i)
            elif factor_dict[i] == "A":
                ladder_indices.append(i)
                is_creator_list.append(False)
            elif factor_dict[i] == "C":
                ladder_indices.append(i)
                is_creator_list.append(True)
            elif factor_dict[i] == "P0":
                projector_indices.append(i)
                projector_state.append(False)
            elif factor_dict[i] == "P1":
                projector_indices.append(i)
                projector_state.append(True)
        
        # If no non-trivial indices are found, we perform a global phase
        # and are done.
        if len(Z_indices + ladder_indices + projector_indices) == 0:
            gphase(coeff, qv[0])
            return
        
        # If there are only projectors, the circuit is a mcp gate
        elif len(Z_indices + ladder_indices) == 0:
            
            # Flip the relevant qubits to ensure the right states are projected.
            flip_qubits = [qv[projector_indices[i]] for i in range(len(projector_indices)) if not projector_state[i]]
            
            if len(flip_qubits) == 0:
                env = QuantumEnvironment()
            else:
                env = conjugate(x)(flip_qubits)
            
            # Perform the mcp            
            with env:
                if len(projector_indices) == 1:
                    p(coeff, qv[projector_indices[0]])
                else:
                    mcp(coeff, [qv[i] for i in projector_indices])
                
            return
        
        # For your reference, here is the circuit that that implements
        # exp(i*phi*H)
        # For 
        # H = A(0)*C(1)*C(2)*Z(3)*X(4)
        
        #           ┌───┐                                                                  ┌───┐
        #     qv.0: ┤ X ├────────────■───────────────────────────────────■─────────────────┤ X ├
        #           └─┬─┘┌───┐       │                                   │            ┌───┐└─┬─┘
        #     qv.1: ──┼──┤ X ├───────o───────────────────────────────────o────────────┤ X ├──┼──
        #             │  └─┬─┘┌───┐  │  ┌───┐┌───┐┌──────────────┐┌───┐  │  ┌───┐┌───┐└─┬─┘  │
        #     qv.2: ──■────■──┤ H ├──┼──┤ X ├┤ X ├┤ Rz(-1.0*phi) ├┤ X ├──┼──┤ X ├┤ H ├──■────■──
        #                     └───┘  │  └─┬─┘└─┬─┘└──────┬───────┘└─┬─┘  │  └─┬─┘└───┘
        #     qv.3: ─────────────────┼────■────┼─────────┼──────────┼────┼────■─────────────────
        #           ┌───┐            │         │         │          │    │  ┌───┐
        #     qv.4: ┤ H ├────────────┼─────────■─────────┼──────────■────┼──┤ H ├───────────────
        #           └───┘          ┌─┴─┐                 │             ┌─┴─┐└───┘
        # hs_anc.0: ───────────────┤ X ├─────────────────■─────────────┤ X ├────────────────────
        #                          └───┘                               └───┘
        
        
        # To implement the ladder operators, we use the ideas from https://arxiv.org/abs/2310.12256
        
        # In it's essence the circuits described there are a conjugation with an inverse GHZ state
        # preparation and a multi controlled RZ gate.

        def inv_ghz_state(qb_list):
            if ladder_ctrl_state[-1] == "1":
                x(qb_list[-1])
            for qb in qb_list[:-1]:
                cx(qb_list[-1], qb)
            if ladder_ctrl_state[-1] == "1":
                x(qb_list[-1])
            h(qb_list[-1])
        
        # We determine the ctrl state and the qubits of the MCRZ 
        ladder_ctrl_state = ""
        ladder_qubits = []
        for i in range(len(ladder_indices)):
            ladder_ctrl_state += str(int(not is_creator_list[i]))
            ladder_qubits.append(qv[ladder_indices[i]])
        
        # The qubit that receives the RZ gate will be called anchor qubit.
        anchor_index = (Z_indices + ladder_indices)[-1]
        
        if len(ladder_qubits):
            # If there are ladder qubits, we conjugate with the above
            # GHZ state preparation function
            env = conjugate(inv_ghz_state)(ladder_qubits)
        else:
            env = QuantumEnvironment()
        
                
        with env:
            
            # To realize the behavior of the projector indices, not that the
            # Hamiltonian has the form 
            
            # H = P1*H_red
            #   = |1><1|*H_red
            
            # Simulating this H is therefore the controlled version of H_red:
                
            # exp(itH) = |0><0| ID + |1><1| exp(itH_red)
            
            # We can therefore add the projector indices to the computation of
            # the control qubit. If the projector indices are in the wrong state,
            # no phase will be performed, which is precisely what we want
            
            # Determine the control qubits and the control state
            projector_ctrl_state = ""
            projector_qubits = []
            
            for i in range(len(projector_indices)):
                projector_ctrl_state += str(int(projector_state[i]))
                projector_qubits.append(qv[projector_indices[i]])
            
            
            control_qubit_available = False
            
            # If the term has neither ladder operators nor projectors,
            # we don't need a controlled RZ
            if len(ladder_indices + projector_qubits) == 0:
                env = QuantumEnvironment()
            elif len(ladder_indices) == 1 and len(projector_indices) == 0:
                # If there is only a single ladder index, we achieve the behavior
                # with a single RZ and H gate
                env = QuantumEnvironment()
            elif len(ladder_indices) == 2 and len(projector_indices) == 0:
                # If there are two ladder indices, we don't need and mcx to compute
                # the control value but instead we use a CRZ gate
                
                # hs_anc is the qubit where RZ gate is controlled on
                hs_anc = qv[ladder_indices[0]]
                
                
                control_qubit_available = True
                
                if ladder_ctrl_state[0] == "0":
                    # Flip the control if required by the ladder state
                    env = conjugate(x)(hs_anc)
                else:
                    env = QuantumEnvironment()
                    
            else:
                # In any other case, we allocate an additional quantum bool,
                # perform a mcx to compute the control value.
                # To achieve the multi-controlled RZ behavior, we control the RZ
                # on that quantum bool.

                hs_anc = QuantumBool()
                control_qubit_available = True
                
                # Compute the control value
                env = conjugate(mcx)(ladder_qubits[:-1] + projector_qubits, 
                                     hs_anc, 
                                     ctrl_state = ladder_ctrl_state[:-1] + projector_ctrl_state, 
                                     method = "gray")
            
            
            # Perform the conjugation
            with env:
                
                # Before we execute the RZ, we need to deal with the Z indices.
                
                # By inspection we see that acting on qubit 1 & 2 with Z operators,
                # is equivalent to the term (-1)**(q_1 (+) q_2)
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
                    
                    
                    # Set up the control environment
                    if control_qubit_available:
                        env = control(hs_anc)
                    else:
                        env = QuantumEnvironment()
                    
                    if len(ladder_indices) == 0:
                        coeff *= 2
                    
                    # Perform the controlled RZ
                    with env:
                        rz(-coeff, qv[anchor_index])
        
        if len(ladder_indices) > 2:
            # Delete ancilla
            hs_anc.delete()
                
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
        for index, P in self.factor_dict.items():
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
        for index,P in self.factor_dict.items():
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
        result_factor_dict={}
        result_coeff = 1
        a = self.factor_dict
        b = other.factor_dict

        keys = set(a.keys()) | set(b.keys())
        for key in keys:
            pauli, coeff = PAULI_TABLE[a.get(key,"I"),b.get(key,"I")]
            if pauli!="I":
                result_factor_dict[key]=pauli
                result_coeff *= coeff
        return QubitTerm(result_factor_dict), result_coeff
    
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
        result_factor_dict=self.factor_dict.copy()
        result_coeff = 1

        for key, value in subs_dict.items():
            if key in result_factor_dict:
                del result_factor_dict[key]
                result_coeff *= value

        return QubitTerm(result_factor_dict), result_coeff
    
    #
    # Commutativity checks
    #

    def commute(self, other):
        """
        Checks if two QubitTerms commute.

        """
        a = self.factor_dict
        b = other.factor_dict

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
        a = self.factor_dict
        b = other.factor_dict

        keys = set()
        keys.update(set(a.keys()))
        keys.update(set(b.keys()))

        for key in keys:
            if a.get(key,"I")!="I" and b.get(key,"I")!="I" and a.get(key,"I")!=b.get(key,"I"):
                return False
        return True
    

