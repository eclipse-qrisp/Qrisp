# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Tests the custom_inversion decorator combined with invert and control environments in Jasp."""

import jax.numpy as jnp

from qrisp import *
from qrisp.alg_primitives.state_preparation import prepare
from qrisp.jasp import *


def test_custom_inverse():
    """Check that custom_inversion picks the backward version under inversion.

    Covers a function whose inverse is not its gate reversal, the same under nested
    inversion environments, and the combination with custom_control.
    """

    @custom_inversion
    def c_inv_function(qbl, inv=False):
        if inv:
            pass
        else:
            qbl.flip()

    def recursion(qbl, recursion_level):

        if recursion_level == 0:
            c_inv_function(qbl)
        else:
            with QuantumEnvironment():
                with invert():
                    recursion(qbl, recursion_level=recursion_level - 1)

    @jaspify
    def main():
        qbl = QuantumBool()
        recursion(qbl, 6)
        return measure(qbl)

    assert main() == True

    @jaspify
    def main():
        qbl = QuantumBool()
        recursion(qbl, 5)
        return measure(qbl)

    assert main() == False

    @custom_control
    @custom_inversion
    def c_inv_control_function(qf, inv=False, ctrl=None):

        if inv and ctrl is None:
            qf[:] = 1
        if not inv and ctrl is not None:
            qf[:] = 2
        if inv and ctrl is not None:
            qf[:] = 3
        if not inv and ctrl is None:
            qf[:] = 4

    @jaspify
    def main(j):

        qbl = QuantumBool()
        qf = QuantumFloat(4)

        for i in range(8):
            if i & 1:
                env_0 = invert()
            else:
                env_0 = QuantumEnvironment()

            if i & 2:
                env_1 = control(qbl)
            else:
                env_1 = QuantumEnvironment()

            if i & 4:
                env_0, env_1 = env_1, env_0

            with control(j == i):
                with env_0:
                    with env_1:
                        c_inv_control_function(qf)

        return measure(qf)

    for i in range(8):
        assert main(i) % 4 == i % 4

    @custom_inversion
    @custom_control
    def c_inv_control_function(qf, inv=False, ctrl=None):

        if inv and ctrl is None:
            qf[:] = 1
        if not inv and ctrl is not None:
            qf[:] = 2
        if inv and ctrl is not None:
            qf[:] = 3
        if not inv and ctrl is None:
            qf[:] = 4

    @jaspify
    def main(j):

        qbl = QuantumBool()
        qf = QuantumFloat(4)

        for i in range(8):
            if i & 1:
                env_0 = invert()
            else:
                env_0 = QuantumEnvironment()

            if i & 2:
                env_1 = control(qbl)
            else:
                env_1 = QuantumEnvironment()

            if i & 4:
                env_0, env_1 = env_1, env_0

            with control(j == i):
                with env_0:
                    with env_1:
                        c_inv_control_function(qf)

        return measure(qf)

    for i in range(8):
        assert main(i) % 4 == i % 4

    @custom_inversion
    def inner_f(qv, inv=False):
        if inv:
            measure(qv[0])
            x(qv[0])
        else:
            measure(qv[0])
            z(qv[0])

    def main():

        qv = QuantumBool()
        qbl = QuantumBool()
        qbl.flip()

        with control(qbl):
            with QuantumEnvironment():
                with conjugate(h)(qv):
                    with invert():
                        with conjugate(h)(qv):
                            inner_f(qv)

        return measure(qv)

    jsp = make_jaspr(main)()
    assert jsp() == True


def test_double_inversion_preserves_custom_inverse():
    """Inverting a custom_inversion function twice must return the original.

    An inverted Jaspr carries a back-pointer to the Jaspr it inverts, which is how
    a second inversion recovers the original instead of deriving one. Inverting can
    reclassify invars as constvars, and folding them back rewraps the Jaspr; the
    rewrapped one used to be handed on without the back-pointer, so the second
    inversion fell back to inverting the body structurally. For a custom_inversion
    user that derivation is exactly the one that does not apply, and here it failed
    outright, because the body is a loop that carries no jrange marker.

    Amplitudes that are only known at run time are what trigger the reclassification:
    prepare then builds its state symbolically out of a q_switch over the traced
    array, rather than emitting a single opaque gate.
    """
    for qubit_amount in (1, 2, 3):
        size = 2**qubit_amount

        def amplitudes(scale, size=size):
            # Scaling by a traced value and renormalizing leaves the state alone but
            # makes the array traced, which is what puts prepare on its symbolic path.
            weights = jnp.arange(1, size + 1, dtype=float) * scale
            return weights / jnp.linalg.norm(weights)

        def build(inversion_depth, qubit_amount=qubit_amount):
            @terminal_sampling
            def main(scale):
                qv = QuantumFloat(qubit_amount)
                if inversion_depth == 0:
                    prepare(qv, amplitudes(scale))
                else:
                    with invert():
                        with invert():
                            prepare(qv, amplitudes(scale))
                return qv

            return main(1.0)

        plain = build(0)
        double_inverted = build(2)

        for state in range(size):
            assert abs(plain.get(state, 0) - double_inverted.get(state, 0)) < 1e-4, (
                f"double inversion changed the state for {qubit_amount} qubits: {plain} vs {double_inverted}"
            )


def test_double_inversion_through_a_control_preserves_custom_inverse():
    """A control between the two inversions must not change the outcome either.

    The two inversions need not be adjacent. control_transform rewraps a Jaspr the
    same way invert_eqn does, so this covers the nesting where the inner inversion
    is reached through a controlled equation. Pure nested control never hit the
    problem, because deriving a controlled version from the body is valid where
    deriving an inverse is not.
    """
    qubit_amount = 2
    size = 2**qubit_amount

    def amplitudes(scale):
        weights = jnp.arange(1, size + 1, dtype=float) * scale
        return weights / jnp.linalg.norm(weights)

    @terminal_sampling
    def main(scale):
        condition = QuantumBool()
        condition.flip()
        qv = QuantumFloat(qubit_amount)
        with invert():
            with control(condition[0]):
                with invert():
                    prepare(qv, amplitudes(scale))
        return qv

    @terminal_sampling
    def reference(scale):
        qv = QuantumFloat(qubit_amount)
        prepare(qv, amplitudes(scale))
        return qv

    # The control qubit is held at |1>, so the controlled body acts unconditionally
    # and the two inversions cancel, leaving the plain state preparation.
    through_control = main(1.0)
    plain = reference(1.0)

    for state in range(size):
        assert abs(plain.get(state, 0) - through_control.get(state, 0)) < 1e-4, (
            f"inversion through a control changed the state: {plain} vs {through_control}"
        )


def test_double_inversion_of_gidney_mcx_round_trips():
    """Two inversions of the Gidney MCX must restore the compute gate sequence.

    The Gidney MCX is the sharpest custom_inversion user: its inverse uncomputes
    via a measurement, so deriving an inverse from the body is impossible rather
    than merely wrong. One inversion must yield the measurement-based uncomputation
    and two must yield the original gates back.
    """

    def gate_names(inversion_depth):
        def main():
            control_qf = QuantumFloat(2)
            h(control_qf)
            target = QuantumBool()

            def body():
                mcx([control_qf[0], control_qf[1]], target[0], method="gidney")

            if inversion_depth == 0:
                body()
            elif inversion_depth == 1:
                with invert():
                    body()
            else:
                with invert():
                    with invert():
                        body()
            return target

        return [instruction.op.name for instruction in make_jaspr(main)().to_qc()[-1].data]

    forward = gate_names(0)

    assert "measure" in gate_names(1), "a single inversion must uncompute via measurement"
    assert "measure" not in forward
    assert gate_names(2) == forward
