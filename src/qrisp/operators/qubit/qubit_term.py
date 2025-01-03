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

from qrisp import gphase, rz, cx, conjugate, custom_control
from qrisp.operators.qubit.visualization import X_,Y_,Z_

from sympy import Symbol
import numpy as np

PAULI_TABLE = {("I","I"):("I",1),("I","X"):("X",1),("I","Y"):("Y",1),("I","Z"):("Z",1),
               ("I","A"):("A",1),("I","C"):("C",1),("I","P0"):("P0",1),("I","P1"):("P1",1),
               ("X","I"):("X",1),("X","X"):("I",1),("X","Y"):("Z",1j),("X","Z"):("Y",(-0-1j)),
               ("X","A"):("P1",1),("X","C"):("P0",1),("X","P0"):("C",1),("X","P1"):("A",1),
               ("Y","I"):("Y",1),("Y","X"):("Z",(-0-1j)),("Y","Y"):("I",1),("Y","Z"):("X",1j),
               ("Y","A"):("P1",1j),("Y","C"):("P0",(-0-1j)),("Y","P0"):("C",1j),
               ("Y","P1"):("A",(-0-1j)),("Z","I"):("Z",1),("Z","X"):("Y",1j),
               ("Z","Y"):("X",(-0-1j)),("Z","Z"):("I",1),("Z","A"):("A",1),
               ("Z","C"):("C",-1),("Z","P0"):("P0",1),("Z","P1"):("P1",-1),
               ("A","I"):("A",1),("A","X"):("P0",1),("A","Y"):("P0",1j),("A","Z"):("A",-1),
               ("A","A"):("I",0),("A","C"):("P0",1),("A","P0"):("I",0),("A","P1"):("A",1),
               ("C","I"):("C",1),("C","X"):("P1",1),("C","Y"):("P1",(-0-1j)),("C","Z"):("C",1),
               ("C","A"):("P1",1),("C","C"):("I",0),("C","P0"):("C",1),("C","P1"):("I",0),
               ("P0","I"):("P0",1),("P0","X"):("A",1),("P0","Y"):("A",(-0-1j)),
               ("P0","Z"):("P0",1),("P0","A"):("A",1),("P0","C"):("I",0),("P0","P0"):("P0",1),
               ("P0","P1"):("I",0),("P1","I"):("P1",1),("P1","X"):("C",1),("P1","Y"):("C",1j),
               ("P1","Z"):("P1",-1),("P1","A"):("I",0),("P1","C"):("C",1),("P1","P0"):("I",0),
               ("P1","P1"):("P1",1)}

#
# QubitTerm
#

class QubitTerm:
    r"""
    
    """

    def __init__(self, factor_dict={}):
        self.factor_dict = dict(factor_dict)
        
        self.hash_value = hash(tuple(sorted(factor_dict.items(), key = lambda x : x[0])))

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
    
    def binary_representation(self, n):
        x_vector = np.zeros(n, dtype=int)
        z_vector = np.zeros(n, dtype=int)
        for index in range(n):
            curr_factor = self.factor_dict.get(index,"I")
            if curr_factor=="X":
                x_vector[index] = 1
            elif curr_factor=="Z":
                z_vector[index] = 1
            elif curr_factor=="Y":
                x_vector[index] = 1
                z_vector[index] = 1

        return x_vector, z_vector
    
    def serialize(self):
        # This function serializes the QubitTerm in a way that facilitates the 
        # measurement post-processing. To learn more details about the reasoning
        # behind this Serialisation check QubitOperator.conjugation_circuit
        
        # The idea here is to serialize the operator via 3 integers.
        # These integers specify how the energy of a measurement sample should
        # be computed.
        # They are processed in the function "evaluate_observable".
            
        # 1. The Z-int: The binary representation of this integer has a 1 at
        # every digit, where there is a Pauli term in self.
        
        # 2. The and_int: The binary representation has a 1 at every digit, which
        # should participate in an AND evaluation. If the evaluation of the AND
        # does not return True, the energy of this measurement is 0.

        # 3. The ctrl_int: Thi binary representation has a 1 at every digit,
        # which should be flipped before evaluating the AND value.
        
        z_int = 0
        and_int = 0
        ctrl_int = 0
        last_ladder_factor = None
        factor_dict = self.factor_dict
        for i in factor_dict.keys():
            bit = (1<<i)
            
            # Go through the cases and update the appropriate integers
            if factor_dict[i] in ["X", "Y", "Z"]:
                z_int |= bit
                continue
            elif factor_dict[i] == "C":
                ctrl_int |= bit
                last_ladder_factor = bit
                pass
            elif factor_dict[i] == "A":
                last_ladder_factor = bit
                pass
            elif factor_dict[i] == "P0":
                pass
            elif factor_dict[i] == "P1":
                ctrl_int |= bit
                pass
            else:
                continue
            
            and_int |= bit
        
        # The last ladder factor should not participate in the AND but should
        # instead be treated like a Z Operator (see QubitOperator.get_conjugation_circuit
        # to learn why).
        if last_ladder_factor is not None:
            and_int ^= last_ladder_factor
            z_int ^= last_ladder_factor
        
        # Returns a tuple, which contains the relevant integers and a boolean
        # indicating whether this term contains any ladder operators.
        
        return (z_int, and_int, ctrl_int, int(last_ladder_factor is not None))
    
    def to_pauli(self):
        
        from qrisp.operators import X, Y, Z
        res = 1
        
        for i, factor in self.factor_dict.items():
            if factor == "X":
                res *= X(i)
            elif factor == "Y":
                res *= Y(i)
            elif factor == "Z":
                res *= Z(i)
            elif factor == "A":
                res *= (X(i) + 1j*Y(i))*0.5
            elif factor == "C":
                res *= (X(i) - 1j*Y(i))*0.5
            elif factor == "P0":
                res *= (Z(i) + 1)*0.5
            elif factor == "P1":
                res *= (Z(i) - 1)*(-0.5)
        
        return res
    
    def adjoint(self):
        new_factor_dict = {}
        for i, factor in self.factor_dict.items():
            if factor in ["X", "Y", "Z", "P0", "P1"]:
                new_factor_dict[i] = factor
            elif factor == "A":
                new_factor_dict[i] = "C"
            elif factor == "C":
                new_factor_dict[i] = "A"
            
        return QubitTerm(new_factor_dict)
    
    #
    # Simulation
    #
    @custom_control(static_argnums = 0)
    def simulate(self, coeff, qv, ctrl = None):

        from qrisp import h, cx, rz, mcp, conjugate, control, QuantumBool, mcx, x, p, s, QuantumEnvironment, gphase, QuantumVariable, find_qs
        from qrisp.operators import QubitOperator
        import numpy as np
        # If required, do change of basis. Change of basis here means, that
        # the quantum argument is conjugated with a function that makes the
        # Operator diagonal. Please refer to the comments in QubitOperator.change_of_basis
        # to learn how this works.
        for factor in self.factor_dict.values():
            if factor not in ["I", "Z", "P0", "P1"]:
                qubit_op = QubitOperator({self : coeff})
                with conjugate(qubit_op.change_of_basis)(qv) as diagonal_op:
                    for diagonal_term, coeff in diagonal_op.terms_dict.items():
                        diagonal_term.simulate(coeff, qv)
                return        
        
        qs = find_qs(qv)
        
        # We group the term into 2 types:
        
        # 1. Z-indices, where a Z-Operator is applied
        # 2. Projector indices, i.e. P0 or P1
        
        # Contains the indices of Pauli Z
        Z_indices = []
        
        # Contains the indices of the projection Operators
        projector_indices = []
        # Contains the booleans, which specify whether P0 or P1
        projector_state = []
        
        factor_dict = self.factor_dict
        for i in factor_dict.keys():
            if factor_dict[i] == "Z":
                Z_indices.append(i)
            elif factor_dict[i] == "P0":
                projector_indices.append(i)
                projector_state.append(False)
            elif factor_dict[i] == "P1":
                projector_indices.append(i)
                projector_state.append(True)
                
        # Determine the control qubits and the control state
        projector_ctrl_state = ""
        projector_qubits = []
        
        if ctrl is not None:
            projector_qubits.append(ctrl)
            projector_ctrl_state = "1"
        
        for i in range(len(projector_indices)):
            projector_ctrl_state += str(int(projector_state[i]))
            projector_qubits.append(qv[projector_indices[i]])
        
        
        # If no non-trivial indices are found, we perform a global phase
        # and are done.
        if len(Z_indices + projector_qubits) == 0:
            gphase(-coeff, qv[0])
            return
        
        # If there are only projectors, the circuit is a mcp gate
        elif len(Z_indices) == 0:
            
            # Perform the mcp            
            if len(projector_qubits) == 1:
                if projector_ctrl_state[0] == "0":
                    p(coeff, projector_qubits[0])
                    gphase(-coeff, projector_qubits[0])
                else:
                    p(-coeff, projector_qubits[0])
            else:
                mcp(-coeff, projector_qubits, method = "balauca", ctrl_state = projector_ctrl_state)
                    
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
        
        # Since we performed the change of basis however already at the beginning
        # of this function, we don't have to perform the GHZ preparation anymore.
        # We only need to consider the projectors and the Z-indices

        
        # To realize the behavior of the projector indices, not that the
        # Hamiltonian has the form 
        
        # H = P1*H_red
        #   = |1><1|*H_red
        
        # Simulating this H is therefore the controlled version of H_red:
            
        # exp(itH) = |0><0| ID + |1><1| exp(itH_red)
        
        # We can therefore add the projector indices to the computation of
        # the control qubit. If the projector indices are in the wrong state,
        # no phase will be performed, which is precisely what we want
        
        flip_control_phase = False
        # If the term has no projectors,
        # we don't need a controlled RZ
        if len(projector_qubits) == 0:
            env = QuantumEnvironment()
            control_qubit_available = False
        elif len(projector_qubits) == 1:
            # If there is only one projector qubit, we can use this as control value
            hs_anc = projector_qubits[0]
            control_qubit_available = True
            if not projector_ctrl_state[0] == "1":
                flip_control_phase = True
            env = QuantumEnvironment()
        else:
            # In any other case, we allocate an additional quantum bool,
            # perform a mcx to compute the control value.
            # To achieve the multi-controlled RZ behavior, we control the RZ
            # on that quantum bool.

            hs_anc_qbl = QuantumBool(qs = qs, name = "hs_anc*")
            hs_anc = hs_anc_qbl[0]
            control_qubit_available = True
            
            # Compute the control value
            
            from qrisp.alg_primitives.mcx_algs import balauca_layer
            
            def semi_balauca_mcx(projector_qubits, target, ctrl_state, ancillae):
                
                reduction_qubits = list(projector_qubits)
                fresh_ancillae = list(ancillae)
                ctrl_list = [ctrl_state[i] for i in range(len(reduction_qubits))]
            
                while len(reduction_qubits) > 2:
                    
                    k = len(reduction_qubits)//2
                    
                    balauca_layer(reduction_qubits[:2*k], 
                                  fresh_ancillae[:k], 
                                  structure=k*[2], 
                                  ctrl_list = [ctrl_list[i] for i in range(2*k)],
                                  use_mcm = True)
                    
                    reduction_qubits = reduction_qubits[2*k:] + fresh_ancillae[:k]
                    ctrl_list = ctrl_list[2*k:] + k*["1"]
                    fresh_ancillae = fresh_ancillae[k:]
                
                
                mcx(reduction_qubits, target, method = "gray", ctrl_state = "".join(ctrl_list))
                    
            balauca_ancillae = QuantumVariable(len(projector_qubits)-1, qs = qs, name = "balauca_ancilla*")
            
            env = conjugate(semi_balauca_mcx)(projector_qubits,
                                              hs_anc,
                                              projector_ctrl_state,
                                              ancillae = [balauca_ancillae[k] for k in range(len(projector_qubits)-1)])
        
        # Perform the conjugation
        with env:
            
            # The following qubit will be the target of the RZ gate
            anchor_index = Z_indices[-1]
            
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
                    cx(qv[i], qv[anchor_index])
            
            # Perform the conjugation
            with conjugate(flip_anchor_qubit)(qv, anchor_index = anchor_index, Z_indices = Z_indices[:-1]):
                
                # Perform the controlled RZ
                if control_qubit_available:
                    # Use Selinger's circuit (page 5)
                    with conjugate(cx)(qv[anchor_index], hs_anc):
                        rz(coeff, qv[anchor_index])
                        if flip_control_phase:
                            coeff = -coeff
                        rz(-coeff, hs_anc)
                else:
                    rz(coeff*2, qv[anchor_index])
                
    
        if len(projector_indices) >= 2:
            # Delete ancilla
            hs_anc_qbl.delete()
            balauca_ancillae.delete()
                    
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
                return Symbol("A_" + str(index), commutative = False)
            if P=="C":
                return Symbol("C_" + str(index), commutative = False)
            if P=="P0":
                return Symbol("P^0_" + str(index), commutative = False)
            if P=="P1":
                return Symbol("P^1_" + str(index), commutative = False)
        
        expr = 1
        index_list = sorted(list(self.factor_dict.keys()))
        
        for i in index_list:
            expr = expr*to_spin(self.factor_dict[i],str(i))

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
            factor, coeff = PAULI_TABLE[a.get(key,"I"),b.get(key,"I")]
            if coeff == 0:
                return QubitTerm({}), 0
            if factor != "I":
                result_factor_dict[key]=factor
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
    # Commutativity
    #
    def commutator(self, other):
        from qrisp.operators.qubit import QubitOperator
        
        term_0, coeff_0 = self*other
        term_1, coeff_1 = other*self
        
        temp =  QubitOperator({term_0 : coeff_0}) - QubitOperator({term_1 : coeff_1})
        temp.apply_threshold(0.5)
        
        return temp

    def commute(self, other):
        """
        Checks if two QubitTerms commute.

        """
        a = self.factor_dict
        b = other.factor_dict

        keys = set()
        keys.update(set(a.keys()))
        keys.update(set(b.keys()))

        sign_flip = 1
        
        for key in keys:
            factor_a = PAULI_TABLE[a.get(key, "I"), b.get(key, "I")]
            factor_b = PAULI_TABLE[b.get(key, "I"), a.get(key, "I")]
            
            if factor_a[1] == 0 and factor_b[1] == 0:
                return True
            if factor_a[0] == factor_b[0]:
                if factor_a[1] == factor_b[1]:
                    continue
                elif factor_a[1] == -factor_b[1]:
                    sign_flip *= -1
                    continue
            else:
                return len(self.commutator(other).terms_dict) == 0
                
        return sign_flip == 1

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
            if not PAULI_TABLE[a.get(key,"I"), b.get(key,"I")] == PAULI_TABLE[b.get(key,"I"), a.get(key,"I")]:
                return False
        return True
    
    def intersect(self, other):
        """
        Checks if two QubitTerms operate on the same qubit.

        """
        return len(set(self.factor_dict.keys()).intersection(other.factor_dict.keys())) != 0
    
    def ladders_agree(self, other):
        """
        Checks if the ladder operators of two QubitTerms operate on the same set of qubits.

        Parameters
        ----------
        other : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        ladder_indices_self = [factor[0] for factor in self.factor_dict.items() if factor[1] in ["A", "C"]]
        ladder_indices_other = [factor[0] for factor in other.factor_dict.items() if factor[1] in ["A", "C"]]
        return set(ladder_indices_self) == set(ladder_indices_other)
        