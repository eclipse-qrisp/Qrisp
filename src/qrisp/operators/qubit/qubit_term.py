"""
********************************************************************************
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
********************************************************************************
"""

from qrisp import gphase, rz, cx, conjugate, custom_control
from qrisp.operators.qubit.visualization import X_, Y_, Z_

from sympy import Symbol
import numpy as np

import jax
from jax import tree_util

PAULI_TABLE = {
    (0, 0): (0, 1),
    (0, 1): (1, 1),
    (0, 2): (2, 1),
    (0, 3): (3, 1),
    (0, 4): (4, 1),
    (0, 5): (5, 1),
    (0, 6): (6, 1),
    (0, 7): (7, 1),
    (1, 0): (1, 1),
    (1, 1): (0, 1),
    (1, 2): (3, 1j),
    (1, 3): (2, (-0 - 1j)),
    (1, 4): (7, 1),
    (1, 5): (6, 1),
    (1, 6): (5, 1),
    (1, 7): (4, 1),
    (2, 0): (2, 1),
    (2, 1): (3, (-0 - 1j)),
    (2, 2): (0, 1),
    (2, 3): (1, 1j),
    (2, 4): (7, 1j),
    (2, 5): (6, (-0 - 1j)),
    (2, 6): (5, 1j),
    (2, 7): (4, (-0 - 1j)),
    (3, 0): (3, 1),
    (3, 1): (2, 1j),
    (3, 2): (1, (-0 - 1j)),
    (3, 3): (0, 1),
    (3, 4): (4, 1),
    (3, 5): (5, -1),
    (3, 6): (6, 1),
    (3, 7): (7, -1),
    (4, 0): (4, 1),
    (4, 1): (6, 1),
    (4, 2): (6, 1j),
    (4, 3): (4, -1),
    (4, 4): (0, 0),
    (4, 5): (6, 1),
    (4, 6): (0, 0),
    (4, 7): (4, 1),
    (5, 0): (5, 1),
    (5, 1): (7, 1),
    (5, 2): (7, (-0 - 1j)),
    (5, 3): (5, 1),
    (5, 4): (7, 1),
    (5, 5): (0, 0),
    (5, 6): (5, 1),
    (5, 7): (0, 0),
    (6, 0): (6, 1),
    (6, 1): (4, 1),
    (6, 2): (4, (-0 - 1j)),
    (6, 3): (6, 1),
    (6, 4): (4, 1),
    (6, 5): (0, 0),
    (6, 6): (6, 1),
    (6, 7): (0, 0),
    (7, 0): (7, 1),
    (7, 1): (5, 1),
    (7, 2): (5, 1j),
    (7, 3): (7, -1),
    (7, 4): (0, 0),
    (7, 5): (5, 1),
    (7, 6): (0, 0),
    (7, 7): (7, 1),
}

#
# QubitTerm
#


class QubitTerm:
    r""" """

    def __init__(self, factor_dict={}, hash_value=None):
        self.factor_dict = dict(factor_dict)

        if hash_value is not None:
            self.hash_value = hash_value
        else:
            self.hash_value = hash(tuple(sorted(factor_dict.items(), key=lambda x: x[0])))

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
        return len(self.factor_dict) == 0

    def binary_representation(self, n):
        x_vector = np.zeros(n, dtype=int)
        z_vector = np.zeros(n, dtype=int)
        for index in range(n):
            curr_factor = self.factor_dict.get(index, 0)
            if curr_factor == 1:
                x_vector[index] = 1
            elif curr_factor == 3:
                z_vector[index] = 1
            elif curr_factor == 2:
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
            bit = 1 << i

            # Go through the cases and update the appropriate integers
            if factor_dict[i] in [1, 2, 3]:
                z_int |= bit
                continue
            elif factor_dict[i] == 5:
                ctrl_int |= bit
                last_ladder_factor = bit
                pass
            elif factor_dict[i] == 4:
                last_ladder_factor = bit
                pass
            elif factor_dict[i] == 6:
                pass
            elif factor_dict[i] == 7:
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
            if factor == 1:
                res *= X(i)
            elif factor == 2:
                res *= Y(i)
            elif factor == 3:
                res *= Z(i)
            elif factor == 4:
                res *= (X(i) + 1j * Y(i)) * 0.5
            elif factor == 5:
                res *= (X(i) - 1j * Y(i)) * 0.5
            elif factor == 6:
                res *= (Z(i) + 1) * 0.5
            elif factor == 7:
                res *= (Z(i) - 1) * (-0.5)

        return res

    def adjoint(self):
        new_factor_dict = {}
        for i, factor in self.factor_dict.items():
            if factor in [1, 2, 3, 6, 7]:
                new_factor_dict[i] = factor
            elif factor == 4:
                new_factor_dict[i] = 5
            elif factor == 5:
                new_factor_dict[i] = 4

        return QubitTerm(new_factor_dict)

    #
    # Simulation
    #
    from qrisp.jasp import qache

    @qache
    def jasp_simulate(self, qv):
        #indices = jax.tree.leaves(self)
        #print(indices)

        rz(1,qv[0])

        #def flip_anchor_qubit(qv, anchor_index, Z_indices):
        #        for i in Z_indices:
        #            cx(qv[i], qv[anchor_index])

        #with conjugate(flip_anchor_qubit)(qv, indices[-1],indices[:-1]):
        #    rz(coeff, qv[indices[-1]])



    @custom_control(static_argnums=0)
    def simulate(self, coeff, qv, ctrl=None):

        from qrisp import (
            h,
            cx,
            rz,
            mcp,
            conjugate,
            control,
            QuantumBool,
            mcx,
            x,
            p,
            s,
            QuantumEnvironment,
            gphase,
            QuantumVariable,
            find_qs,
        )
        from qrisp.operators import QubitOperator
        import numpy as np

        # If required, do change of basis. Change of basis here means, that
        # the quantum argument is conjugated with a function that makes the
        # Operator diagonal. Please refer to the comments in QubitOperator.change_of_basis
        # to learn how this works.
        for factor in self.factor_dict.values():
            if factor not in [0, 3, 6, 7]:
                qubit_op = QubitOperator({self: coeff})
                with conjugate(qubit_op.change_of_basis)(qv) as diagonal_op:
                    for diagonal_term, coeff in diagonal_op.terms_dict.items():
                        if ctrl is None:
                            diagonal_term.simulate(coeff, qv)
                        else:
                            with control(ctrl):
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
            if factor_dict[i] == 3:
                Z_indices.append(i)
            elif factor_dict[i] == 6:
                projector_indices.append(i)
                projector_state.append(False)
            elif factor_dict[i] == 7:
                projector_indices.append(i)
                projector_state.append(True)

        # Determine the control qubits and the control state
        projector_ctrl_state = 0
        projector_qubits = []

        if ctrl is not None:
            projector_qubits.append(ctrl)
            projector_ctrl_state = 1
            # projector_ctrl_state = "1"

        for i in range(len(projector_indices)):
            # projector_ctrl_state += str(int(projector_state[i]))
            projector_ctrl_state += int(projector_state[i]) * 2 ** (
                len(projector_qubits)
            )
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
                if projector_ctrl_state == 0:
                    p(coeff, projector_qubits[0])
                    gphase(-coeff, projector_qubits[0])
                else:
                    p(-coeff, projector_qubits[0])
            else:
                mcp(
                    -coeff,
                    projector_qubits,
                    method="balauca",
                    ctrl_state=projector_ctrl_state,
                )

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
            if not projector_ctrl_state == 1:
                flip_control_phase = True
            env = QuantumEnvironment()
        else:
            # In any other case, we allocate an additional quantum bool,
            # perform a mcx to compute the control value.
            # To achieve the multi-controlled RZ behavior, we control the RZ
            # on that quantum bool.

            hs_anc_qbl = QuantumBool(qs=qs, name="hs_anc*")
            hs_anc = hs_anc_qbl[0]
            control_qubit_available = True

            # Compute the control value

            from qrisp.alg_primitives.mcx_algs import balauca_layer

            def semi_balauca_mcx(projector_qubits, target, ctrl_state, ancillae):

                reduction_qubits = list(projector_qubits)
                fresh_ancillae = list(ancillae)
                ctrl_list = [
                    str(int((ctrl_state >> i) % 2))
                    for i in range(len(reduction_qubits))
                ]

                while len(reduction_qubits) > 2:

                    k = len(reduction_qubits) // 2

                    balauca_layer(
                        reduction_qubits[: 2 * k],
                        fresh_ancillae[:k],
                        structure=k * [2],
                        ctrl_list=[ctrl_list[i] for i in range(2 * k)],
                        use_mcm=True,
                    )

                    reduction_qubits = reduction_qubits[2 * k :] + fresh_ancillae[:k]
                    ctrl_list = ctrl_list[2 * k :] + k * ["1"]
                    fresh_ancillae = fresh_ancillae[k:]

                mcx(
                    reduction_qubits,
                    target,
                    method="gidney",
                    ctrl_state="".join(ctrl_list),
                )

            balauca_ancillae = QuantumVariable(
                len(projector_qubits) - 1, qs=qs, name="balauca_ancilla*"
            )

            env = conjugate(semi_balauca_mcx)(
                projector_qubits,
                hs_anc,
                projector_ctrl_state,
                ancillae=[
                    balauca_ancillae[k] for k in range(len(projector_qubits) - 1)
                ],
            )

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
            with conjugate(flip_anchor_qubit)(
                qv, anchor_index=anchor_index, Z_indices=Z_indices[:-1]
            ):

                # Perform the controlled RZ
                if control_qubit_available:
                    # Use Selinger's circuit (page 5)
                    with conjugate(cx)(qv[anchor_index], hs_anc):
                        rz(coeff, qv[anchor_index])
                        if flip_control_phase:
                            coeff = -coeff
                        rz(-coeff, hs_anc)
                else:
                    rz(coeff * 2, qv[anchor_index])

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
            if P == 0:
                return 1
            if P == 1:
                return X_(index)
            if P == 2:
                return Y_(index)
            if P == 3:
                return Z_(index)
            if P == 4:
                return Symbol("A(" + str(index) + ")", commutative=False)
            if P == 5:
                return Symbol("C(" + str(index) + ")", commutative=False)
            if P == 6:
                return Symbol("P0(" + str(index) + ")", commutative=False)
            if P == 7:
                return Symbol("P1(" + str(index) + ")", commutative=False)

        expr = 1
        index_list = sorted(list(self.factor_dict.keys()))

        for i in index_list:
            expr = expr * to_spin(self.factor_dict[i], str(i))

        return expr

    #
    # Arithmetic
    #

    def __pow__(self, e):
        if isinstance(e, int) and e >= 0:
            if e % 2 == 0:
                return QubitTerm({(): 1})
            else:
                return self
        else:
            raise TypeError(
                "Unsupported operand type(s) for ** or pow(): "
                + str(type(self))
                + " and "
                + str(type(e))
            )

    def __mul__(self, other):
        result_factor_dict = {}
        result_coeff = 1
        a = self.factor_dict
        b = other.factor_dict

        keys = set(a.keys()) | set(b.keys())
        for key in keys:
            factor, coeff = PAULI_TABLE[a.get(key, 0), b.get(key, 0)]
            if coeff == 0:
                return QubitTerm({}), 0
            if factor != 0:
                result_factor_dict[key] = factor
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
        result_factor_dict = self.factor_dict.copy()
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

        term_0, coeff_0 = self * other
        term_1, coeff_1 = other * self

        temp = QubitOperator({term_0: coeff_0}) - QubitOperator({term_1: coeff_1})
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
            factor_a = PAULI_TABLE[a.get(key, 0), b.get(key, 0)]
            factor_b = PAULI_TABLE[b.get(key, 0), a.get(key, 0)]

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

    def commute_pauli(self, other):
        """
        Checks if the Pauli factors of two QubitTerms commute and the ladder factors commute qubit-wise.

        """
        a = self.factor_dict
        b = other.factor_dict

        keys = set()
        keys.update(set(a.keys()))
        keys.update(set(b.keys()))

        sign_flip = 1

        for key in keys:
            factor_a = PAULI_TABLE[a.get(key, 0), b.get(key, 0)]
            factor_b = PAULI_TABLE[b.get(key, 0), a.get(key, 0)]

            if not factor_a == factor_b:
                if factor_a[0] in [0, 1, 2, 3] and factor_b[0] in [
                    0,
                    1,
                    2,
                    3,
                ]:
                    if factor_a[1] == -factor_b[1]:
                        sign_flip *= -1
                else:
                    return False

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
            if (
                not PAULI_TABLE[a.get(key, 0), b.get(key, 0)]
                == PAULI_TABLE[b.get(key, 0), a.get(key, 0)]
            ):
                return False
        return True

    def intersect(self, other):
        """
        Checks if two QubitTerms operate on the same qubit.

        """
        return (
            len(set(self.factor_dict.keys()).intersection(other.factor_dict.keys()))
            != 0
        )

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
        ladder_indices_self = [
            factor[0] for factor in self.factor_dict.items() if factor[1] in [4, 5]
        ]
        ladder_indices_other = [
            factor[0] for factor in other.factor_dict.items() if factor[1] in [4, 5]
        ]
        return set(ladder_indices_self) == set(ladder_indices_other)

    def ladders_intersect(self, other):
        """
        Checks if the ladder operators of two QubitTerms operate on the same qubit.

        Parameters
        ----------
        other : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        ladder_indices_self = [
            factor[0] for factor in self.factor_dict.items() if factor[1] in [4, 5]
        ]
        ladder_indices_other = [
            factor[0] for factor in other.factor_dict.items() if factor[1] in [4, 5]
        ]
        return len(set(ladder_indices_self).intersection(ladder_indices_other)) != 0


# Function to flatten QubitTerm
def flatten_qubit_term(term):
    keys = tuple(term.factor_dict.keys())
    vals = tuple(term.factor_dict.values())
    hash = term.hash_value
    leaves = (keys, hash)
    aux_data = vals
    return leaves, aux_data


# Function to unflatten QubitTermfrom leaves and auxiliary data
def unflatten_qubit_term(aux_data, leaves):
    keys, hash = leaves
    vals = aux_data
    terms_dict = dict(zip(keys, vals))
    return QubitTerm(terms_dict, hash)


# Register QubitTerm as a PyTree node
tree_util.register_pytree_node(QubitTerm, flatten_qubit_term, unflatten_qubit_term)