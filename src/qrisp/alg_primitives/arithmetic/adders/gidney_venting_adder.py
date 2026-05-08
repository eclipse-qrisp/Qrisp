"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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

from qrisp import cx, x, z, h, mcx, control


def bit_inverted_cx(ctrl_qubit, target_qubit, b):
    """
    CX gate whose control polarity is inverted when classical bit *b* is True.

    Equivalent to: if ctrl_qubit ⊕ b == 1  →  X(target).

    Parameters
    ----------
    ctrl_qubit : Qubit
        The control qubit.
    target_qubit : Qubit
        The target qubit (receives X when condition met).
    b : bool or jax traced bool
        Classical bit that inverts the control polarity.
    """
    with control(b):
        x(ctrl_qubit)
    cx(ctrl_qubit, target_qubit)
    with control(b):
        x(ctrl_qubit)


def bit_inverted_cz(ctrl_qubit, target_qubit, b):
    """
    CZ gate whose control polarity is inverted when classical bit *b* is True.

    Parameters
    ----------
    ctrl_qubit : Qubit
        The control qubit.
    target_qubit : Qubit
        The target qubit (receives Z when condition met).
    b : bool or jax traced bool
        Classical bit that inverts the control polarity.
    """
    with control(b):
        x(ctrl_qubit)
    h(target_qubit)
    cx(ctrl_qubit, target_qubit)
    h(target_qubit)
    with control(b):
        x(ctrl_qubit)


def bit_inverted_controlled_gate(ctrl_qubit, b, body_func):
    """
    Run *body_func* controlled on ctrl_qubit with polarity flipped by classical bit *b*.

    Parameters
    ----------
    ctrl_qubit : Qubit
        The control qubit.
    b : bool or jax traced bool
        Classical bit that inverts the control polarity.
    body_func : callable
        Zero-argument callable that applies the desired gate(s) to the target(s).
    """
    with control(b):
        x(ctrl_qubit)
    with control(ctrl_qubit):
        body_func()
    with control(b):
        x(ctrl_qubit)


def zz_parity_controlled_x(q0, q1, target):
    """
    X gate on *target*, conditioned on the ZZ parity of q0 and q1.
    Fires when q0 and q1 have even parity (both |0⟩ or both |1⟩).

    Parameters
    ----------
    q0, q1 : Qubit
        The two qubits whose parity (ZZ eigenvalue) controls the gate.
    target : Qubit
        Receives an X gate when the parity condition is met.
    """
    cx(q0, q1)
    mcx([q1], target, ctrl_state="0")
    cx(q0, q1)


def zz_parity_controlled_z(q0, q1, target):
    """
    Z gate on *target*, conditioned on the ZZ parity of q0 and q1.

    Parameters
    ----------
    q0, q1 : Qubit
        The two qubits whose parity (ZZ eigenvalue) controls the gate.
    target : Qubit
        Receives a Z gate when the parity condition is met.
    """
    cx(q0, q1)
    h(target)
    mcx([q1], target, ctrl_state="0")
    h(target)
    cx(q0, q1)


def zz_parity_controlled_gate(q0, q1, body_func):
    """
    Execute *body_func* controlled on the ZZ parity of q0 and q1.
    Fires when q0 and q1 have even parity.

    Parameters
    ----------
    q0, q1 : Qubit
        The two parity-control qubits.
    body_func : callable
        Zero-argument callable applying the target operation(s).
    """
    cx(q0, q1)
    with control(q1, ctrl_state="0"):
        body_func()
    cx(q0, q1)


def dual_zz_controlled_x(a0, a1, b0, b1, target):
    """
    X gate on *target*, conditioned on two simultaneous ZZ parity checks.
    Fires when parity(a0,a1)=even AND parity(b0,b1)=even.

    Parameters
    ----------
    a0, a1 : Qubit
        First ZZ parity pair.
    b0, b1 : Qubit
        Second ZZ parity pair.
    target : Qubit
        Receives X when both parity conditions are met.
    """
    cx(a0, a1)
    cx(b0, b1)
    mcx([a1, b1], target, ctrl_state="00")
    cx(b0, b1)
    cx(a0, a1)


def dual_zz_controlled_z(a0, a1, b0, b1, target):
    """
    Z gate on *target*, conditioned on two simultaneous ZZ parity checks.

    Parameters
    ----------
    a0, a1 : Qubit
        First ZZ parity pair.
    b0, b1 : Qubit
        Second ZZ parity pair.
    target : Qubit
        Receives Z when both parity conditions are met.
    """
    cx(a0, a1)
    cx(b0, b1)
    h(target)
    mcx([a1, b1], target, ctrl_state="00")
    h(target)
    cx(b0, b1)
    cx(a0, a1)


def dual_zz_controlled_gate(a0, a1, b0, b1, body_func):
    """
    Execute *body_func* when both ZZ parity conditions are satisfied.

    Parameters
    ----------
    a0, a1 : Qubit
        First ZZ parity pair (even parity required).
    b0, b1 : Qubit
        Second ZZ parity pair (even parity required).
    body_func : callable
        Zero-argument callable applying the target operation(s).
    """
    cx(a0, a1)
    cx(b0, b1)
    with control([a1, b1], ctrl_state="00"):
        body_func()
    cx(b0, b1)
    cx(a0, a1)