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

Contains deliberately broken variants of the Cuccaro adder for
demonstrating how ``inpl_adder_test`` catches different classes of bugs.

Each bad adder mirrors the structure of the working
:func:`~qrisp.alg_primitives.arithmetic.adders.cuccaro_adder.cuccaro_adder`
but introduces a single surgical flaw that causes it to fail a specific
assertion in ``_static_inpl_adder_test`` / ``_dynamic_inpl_adder_test``.
"""

from qrisp.core import QuantumVariable, x, cx, mcx
from qrisp.qtypes import QuantumFloat, QuantumBool
from qrisp.environments import conjugate, custom_control
from qrisp.misc import int_encoder
from qrisp.jasp import jrange, jlen
import jax.numpy as jnp

@custom_control
def flip_first_maj_cuccaro(a, b, c_in=None, c_out=None, ctrl=None):
    """Like ``cuccaro_adder`` but the first MAJ gate uses ``cx(b[0], a[0])``
    instead of ``cx(a[0], b[0])``.  This corrupts the carry chain and makes
    every sum wrong when the low bit of ``a`` is set.

    Bug: one cx target is wrong in the first MAJ gate.
    Fails: QQ arithmetic assertion (step 1)."""

    if not isinstance(a, QuantumVariable):
        q_a = b.duplicate()
        with conjugate(int_encoder)(q_a, a):
            flip_first_maj_cuccaro(q_a, b, c_in=c_in, c_out=c_out, ctrl=ctrl)
        q_a.delete()
        return

    if not isinstance(b, QuantumVariable):
        raise ValueError("The second argument must be of type QuantumVariable.")

    dim_a = a.size
    dim_b = b.size
    max_size = jnp.maximum(dim_a, dim_b)
    effective_size_a = jnp.minimum(dim_a, dim_b)
    a = a[:effective_size_a]
    extension_size = jnp.maximum(0, dim_b - dim_a)
    extension_anc_a = QuantumVariable(extension_size)
    extended_a = a[:] + extension_anc_a[:]
    a = extended_a
    dim_a = jlen(a)
    dim_b = jlen(b)
    ancilla = QuantumFloat(max_size)

    if c_in is not None:
        if isinstance(c_in, QuantumBool):
            c_in = c_in[0]
        cx(c_in, ancilla[0])

    if c_out is not None:
        ancilla2 = c_out

    cx(b[0], a[0])
    cx(a[0], ancilla[0])
    mcx([ancilla[0], b[0]], a[0])

    for i in jrange(1, dim_a):
        cx(a[i], b[i])
        cx(a[i], a[i - 1])
        mcx([a[i - 1], b[i]], a[i])

    if c_out is not None:
        cx(a[-1], ancilla2[0])

    if ctrl is None:
        for j in jrange(dim_a - 1):
            i = dim_a - j - 1
            x(b[i])
            cx(a[i - 1], b[i])
            mcx([a[i - 1], b[i]], a[i])
            x(b[i])
            cx(a[i], a[i - 1])
            cx(a[i], b[i])

        x(b[0])
        cx(ancilla[0], b[0])
        mcx([ancilla[0], b[0]], a[0])
        x(b[0])
        cx(a[0], ancilla[0])
        cx(a[0], b[0])
    else:
        for j in jrange(dim_a - 1):
            i = dim_a - j - 1
            mcx([a[i - 1], b[i]], a[i])
            mcx([ctrl, a[i - 1]], b[i])
            cx(a[i], a[i - 1])
            cx(a[i], b[i])

        mcx([ancilla[0], b[0]], a[0])
        mcx([ctrl, ancilla[0]], b[0])
        cx(a[0], ancilla[0])
        cx(a[0], b[0])

    if c_in is not None:
        cx(c_in, ancilla[0])

    ancilla.delete()
    extension_anc_a.delete()


@custom_control
def skip_uma_cuccaro(a, b, c_in=None, c_out=None, ctrl=None):
    """Like ``cuccaro_adder`` but the UMA (un-majority) section is
    skipped entirely. Without UMA the carry bits stay entangled in the
    output, producing garbage.

    Bug: MAJ gates are applied but the UMA section is completely omitted.
    Fails: QQ arithmetic assertion (step 1)."""

    if not isinstance(a, QuantumVariable):
        q_a = b.duplicate()
        with conjugate(int_encoder)(q_a, a):
            skip_uma_cuccaro(q_a, b, c_in=c_in, c_out=c_out, ctrl=ctrl)
        q_a.delete()
        return

    if not isinstance(b, QuantumVariable):
        raise ValueError("The second argument must be of type QuantumVariable.")

    dim_a = a.size
    dim_b = b.size
    max_size = jnp.maximum(dim_a, dim_b)
    effective_size_a = jnp.minimum(dim_a, dim_b)
    a = a[:effective_size_a]
    extension_size = jnp.maximum(0, dim_b - dim_a)
    extension_anc_a = QuantumVariable(extension_size)
    extended_a = a[:] + extension_anc_a[:]
    a = extended_a
    dim_a = jlen(a)
    dim_b = jlen(b)
    ancilla = QuantumFloat(max_size)

    if c_in is not None:
        if isinstance(c_in, QuantumBool):
            c_in = c_in[0]
        cx(c_in, ancilla[0])
    if c_out is not None:
        ancilla2 = c_out

    cx(a[0], b[0])
    cx(a[0], ancilla[0])
    mcx([ancilla[0], b[0]], a[0])

    for i in jrange(1, dim_a):
        cx(a[i], b[i])
        cx(a[i], a[i - 1])
        mcx([a[i - 1], b[i]], a[i])

    if c_out is not None:
        cx(a[-1], ancilla2[0])

    if c_in is not None:
        cx(c_in, ancilla[0])
    ancilla.delete()
    extension_anc_a.delete()


@custom_control
def ctrl_ignorant_cuccaro(a, b, c_in=None, c_out=None, ctrl=None):
    """Like ``cuccaro_adder`` but ignores the ``ctrl`` qubit: the addition
    is applied unconditionally even when a control qubit is present.
    Fails the controlled-test assertion ``c == b`` when the control qubit
    is ``|0>``.

    Bug: the ctrl condition is ignored — addition is always applied.
    Fails: controlled QQ / controlled CQ assertions (steps 3 & 4)."""

    if not isinstance(a, QuantumVariable):
        q_a = b.duplicate()
        with conjugate(int_encoder)(q_a, a):
            ctrl_ignorant_cuccaro(q_a, b, c_in=c_in, c_out=c_out,
                                  ctrl=ctrl)
        q_a.delete()
        return

    if not isinstance(b, QuantumVariable):
        raise ValueError("The second argument must be of type QuantumVariable.")

    dim_a = a.size
    dim_b = b.size
    max_size = jnp.maximum(dim_a, dim_b)
    effective_size_a = jnp.minimum(dim_a, dim_b)
    a = a[:effective_size_a]
    extension_size = jnp.maximum(0, dim_b - dim_a)
    extension_anc_a = QuantumVariable(extension_size)
    extended_a = a[:] + extension_anc_a[:]
    a = extended_a
    dim_a = jlen(a)
    dim_b = jlen(b)
    ancilla = QuantumFloat(max_size)

    if c_in is not None:
        if isinstance(c_in, QuantumBool):
            c_in = c_in[0]
        cx(c_in, ancilla[0])
    if c_out is not None:
        ancilla2 = c_out

    cx(a[0], b[0])
    cx(a[0], ancilla[0])
    mcx([ancilla[0], b[0]], a[0])

    for i in jrange(1, dim_a):
        cx(a[i], b[i])
        cx(a[i], a[i - 1])
        mcx([a[i - 1], b[i]], a[i])

    if c_out is not None:
        cx(a[-1], ancilla2[0])

    for j in jrange(dim_a - 1):
        i = dim_a - j - 1
        x(b[i])
        cx(a[i - 1], b[i])
        mcx([a[i - 1], b[i]], a[i])
        x(b[i])
        cx(a[i], a[i - 1])
        cx(a[i], b[i])

    x(b[0])
    cx(ancilla[0], b[0])
    mcx([ancilla[0], b[0]], a[0])
    x(b[0])
    cx(a[0], ancilla[0])
    cx(a[0], b[0])

    if c_in is not None:
        cx(c_in, ancilla[0])
    ancilla.delete()
    extension_anc_a.delete()


@custom_control
def z_phase_cuccaro(a, b, c_in=None, c_out=None, ctrl=None):
    """Like ``cuccaro_adder`` but applies a phase gate ``p(0.3)`` to the
    carry ancilla after the MAJ section.  The arithmetic is correct, but
    the phase shift is detected by the angle-sum check.

    Bug: an extra ``p(0.3)`` phase gate on the carry ancilla after MAJ.
    Fails: phase-angle assertion in every test block."""

    from qrisp import p as phase_gate

    if not isinstance(a, QuantumVariable):
        q_a = b.duplicate()
        with conjugate(int_encoder)(q_a, a):
            z_phase_cuccaro(q_a, b, c_in=c_in, c_out=c_out, ctrl=ctrl)
        q_a.delete()
        return

    if not isinstance(b, QuantumVariable):
        raise ValueError("The second argument must be of type QuantumVariable.")

    dim_a = a.size
    dim_b = b.size
    max_size = jnp.maximum(dim_a, dim_b)
    effective_size_a = jnp.minimum(dim_a, dim_b)
    a = a[:effective_size_a]
    extension_size = jnp.maximum(0, dim_b - dim_a)
    extension_anc_a = QuantumVariable(extension_size)
    extended_a = a[:] + extension_anc_a[:]
    a = extended_a
    dim_a = jlen(a)
    dim_b = jlen(b)
    ancilla = QuantumFloat(max_size)

    if c_in is not None:
        if isinstance(c_in, QuantumBool):
            c_in = c_in[0]
        cx(c_in, ancilla[0])
    if c_out is not None:
        ancilla2 = c_out

    cx(a[0], b[0])
    cx(a[0], ancilla[0])
    mcx([ancilla[0], b[0]], a[0])

    phase_gate(0.3, ancilla[0])

    for i in jrange(1, dim_a):
        cx(a[i], b[i])
        cx(a[i], a[i - 1])
        mcx([a[i - 1], b[i]], a[i])

    if c_out is not None:
        cx(a[-1], ancilla2[0])

    if ctrl is None:
        for j in jrange(dim_a - 1):
            i = dim_a - j - 1
            x(b[i])
            cx(a[i - 1], b[i])
            mcx([a[i - 1], b[i]], a[i])
            x(b[i])
            cx(a[i], a[i - 1])
            cx(a[i], b[i])

        x(b[0])
        cx(ancilla[0], b[0])
        mcx([ancilla[0], b[0]], a[0])
        x(b[0])
        cx(a[0], ancilla[0])
        cx(a[0], b[0])
    else:
        for j in jrange(dim_a - 1):
            i = dim_a - j - 1
            mcx([a[i - 1], b[i]], a[i])
            mcx([ctrl, a[i - 1]], b[i])
            cx(a[i], a[i - 1])
            cx(a[i], b[i])

        mcx([ancilla[0], b[0]], a[0])
        mcx([ctrl, ancilla[0]], b[0])
        cx(a[0], ancilla[0])
        cx(a[0], b[0])

    if c_in is not None:
        cx(c_in, ancilla[0])
    ancilla.delete()
    extension_anc_a.delete()


@custom_control
def double_adjunct_cuccaro(a, b, c_in=None, c_out=None, ctrl=None):
    """Like ``cuccaro_adder`` but the full MAJ+UMA round is applied twice,
    so the result is ``b += 2*a`` instead of ``b += a``.

    Bug: the MAJ+UMA pair is applied twice (self-adjunct).
    Fails: QQ / CQ arithmetic assertions."""

    if not isinstance(a, QuantumVariable):
        q_a = b.duplicate()
        with conjugate(int_encoder)(q_a, a):
            double_adjunct_cuccaro(q_a, b, c_in=c_in, c_out=c_out,
                                   ctrl=ctrl)
        q_a.delete()
        return

    if not isinstance(b, QuantumVariable):
        raise ValueError("The second argument must be of type QuantumVariable.")

    dim_a = a.size
    dim_b = b.size
    max_size = jnp.maximum(dim_a, dim_b)
    effective_size_a = jnp.minimum(dim_a, dim_b)
    a = a[:effective_size_a]
    extension_size = jnp.maximum(0, dim_b - dim_a)
    extension_anc_a = QuantumVariable(extension_size)
    extended_a = a[:] + extension_anc_a[:]
    a = extended_a
    dim_a = jlen(a)
    dim_b = jlen(b)
    ancilla = QuantumFloat(max_size)

    if c_in is not None:
        if isinstance(c_in, QuantumBool):
            c_in = c_in[0]
        cx(c_in, ancilla[0])
    if c_out is not None:
        ancilla2 = c_out

    for _ in range(2):
        cx(a[0], b[0])
        cx(a[0], ancilla[0])
        mcx([ancilla[0], b[0]], a[0])

        for i in jrange(1, dim_a):
            cx(a[i], b[i])
            cx(a[i], a[i - 1])
            mcx([a[i - 1], b[i]], a[i])

        if c_out is not None:
            cx(a[-1], ancilla2[0])

        if ctrl is None:
            for j in jrange(dim_a - 1):
                i = dim_a - j - 1
                x(b[i])
                cx(a[i - 1], b[i])
                mcx([a[i - 1], b[i]], a[i])
                x(b[i])
                cx(a[i], a[i - 1])
                cx(a[i], b[i])

            x(b[0])
            cx(ancilla[0], b[0])
            mcx([ancilla[0], b[0]], a[0])
            x(b[0])
            cx(a[0], ancilla[0])
            cx(a[0], b[0])
        else:
            for j in jrange(dim_a - 1):
                i = dim_a - j - 1
                mcx([a[i - 1], b[i]], a[i])
                mcx([ctrl, a[i - 1]], b[i])
                cx(a[i], a[i - 1])
                cx(a[i], b[i])

            mcx([ancilla[0], b[0]], a[0])
            mcx([ctrl, ancilla[0]], b[0])
            cx(a[0], ancilla[0])
            cx(a[0], b[0])

    if c_in is not None:
        cx(c_in, ancilla[0])
    ancilla.delete()
    extension_anc_a.delete()


@custom_control
def mishandle_classical_cuccaro(a, b, c_in=None, c_out=None, ctrl=None):
    """Like ``cuccaro_adder`` but encodes the classical input in the wrong
    direction, effectively adding ``-a-1`` to ``b`` when ``a`` is an integer.
    Fails the CQ arithmetic assertion ``(b + j) % 2**i == a``.

    Bug: encodes ``-a-1`` (bitwise complement) instead of ``a``.
    Fails: CQ arithmetic assertion (step 2)."""

    if not isinstance(a, QuantumVariable):
        q_a = b.duplicate()
        with conjugate(int_encoder)(q_a, -a - 1):
            mishandle_classical_cuccaro(q_a, b, c_in=c_in, c_out=c_out,
                                        ctrl=ctrl)
        q_a.delete()
        return

    if not isinstance(b, QuantumVariable):
        raise ValueError("The second argument must be of type QuantumVariable.")

    dim_a = a.size
    dim_b = b.size
    max_size = jnp.maximum(dim_a, dim_b)
    effective_size_a = jnp.minimum(dim_a, dim_b)
    a = a[:effective_size_a]
    extension_size = jnp.maximum(0, dim_b - dim_a)
    extension_anc_a = QuantumVariable(extension_size)
    extended_a = a[:] + extension_anc_a[:]
    a = extended_a
    dim_a = jlen(a)
    dim_b = jlen(b)
    ancilla = QuantumFloat(max_size)

    if c_in is not None:
        if isinstance(c_in, QuantumBool):
            c_in = c_in[0]
        cx(c_in, ancilla[0])
    if c_out is not None:
        ancilla2 = c_out

    cx(a[0], b[0])
    cx(a[0], ancilla[0])
    mcx([ancilla[0], b[0]], a[0])

    for i in jrange(1, dim_a):
        cx(a[i], b[i])
        cx(a[i], a[i - 1])
        mcx([a[i - 1], b[i]], a[i])

    if c_out is not None:
        cx(a[-1], ancilla2[0])

    if ctrl is None:
        for j in jrange(dim_a - 1):
            i = dim_a - j - 1
            x(b[i])
            cx(a[i - 1], b[i])
            mcx([a[i - 1], b[i]], a[i])
            x(b[i])
            cx(a[i], a[i - 1])
            cx(a[i], b[i])

        x(b[0])
        cx(ancilla[0], b[0])
        mcx([ancilla[0], b[0]], a[0])
        x(b[0])
        cx(a[0], ancilla[0])
        cx(a[0], b[0])
    else:
        for j in jrange(dim_a - 1):
            i = dim_a - j - 1
            mcx([a[i - 1], b[i]], a[i])
            mcx([ctrl, a[i - 1]], b[i])
            cx(a[i], a[i - 1])
            cx(a[i], b[i])

        mcx([ancilla[0], b[0]], a[0])
        mcx([ctrl, ancilla[0]], b[0])
        cx(a[0], ancilla[0])
        cx(a[0], b[0])

    if c_in is not None:
        cx(c_in, ancilla[0])
    ancilla.delete()
    extension_anc_a.delete()
