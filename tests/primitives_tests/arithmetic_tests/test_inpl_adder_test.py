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

Tests for ``inpl_adder_test``.

Each deliberately broken Cuccaro variant from ``adder_tools`` is checked to
fail with the specific assertion that matches its bug.
"""

import pytest
from jax.errors import TracerIntegerConversionError

from qrisp import inpl_adder_test, QuantumFloat
from qrisp.alg_primitives.arithmetic.adders.adder_tools import (
    flip_first_maj_cuccaro,
    skip_uma_cuccaro,
    ctrl_ignorant_cuccaro,
    z_phase_cuccaro,
    double_adjunct_cuccaro,
    mishandle_classical_cuccaro,
)


def test_flip_first_maj_cuccaro():
    """The first MAJ gate has a swapped control/target (``cx(b[0], a[0])``
    instead of ``cx(a[0], b[0])``).  This corrupts the carry chain so that
    both the static and dynamic tests detect arithmetic failures."""
    with pytest.raises(AssertionError, match=r".*failed the static test.*"):
        inpl_adder_test(flip_first_maj_cuccaro)


def test_skip_uma_cuccaro():
    """The UMA (un-majority) section is completely omitted.  Without UMA
    the carry bits stay entangled in the output, producing garbage that
    fails both the static and dynamic arithmetic assertions."""
    with pytest.raises(AssertionError, match=r".*failed the static test.*"):
        inpl_adder_test(skip_uma_cuccaro)


def test_ctrl_ignorant_cuccaro():
    """The ``ctrl`` qubit is ignored and the addition is applied
    unconditionally.  The controlled test catches this because ``c``
    is modified even when the control qubit is ``|0>``."""
    with pytest.raises(AssertionError, match=r".*failed the static test.*"):
        inpl_adder_test(ctrl_ignorant_cuccaro)


def test_z_phase_cuccaro():
    """A spurious ``p(0.3)`` phase gate is applied to the carry ancilla.
    The arithmetic is still correct, so the dynamic test passes, but the
    static phase-angle check detects the unwanted phase shift."""
    with pytest.raises(AssertionError, match=r".*failed the static test.*"):
        inpl_adder_test(z_phase_cuccaro)


def test_double_adjunct_cuccaro():
    """The MAJ+UMA round is applied twice, producing ``b += 2*a`` instead
    of ``b += a``.  The error is caught by the arithmetic check in both
    static and dynamic modes."""
    with pytest.raises(AssertionError, match=r".*failed the static test.*"):
        inpl_adder_test(double_adjunct_cuccaro)


def test_qcla_fails_dynamic():
    """The QCLA adder uses Python ``range()``, ``len()`` on qubit
    registers, and other constructs that are incompatible with Jasp
    tracing.  The dynamic test raises ``TracerIntegerConversionError``
    (or an ``AssertionError`` / ``Exception``) as a result."""
    from qrisp import qcla

    with pytest.raises((AssertionError, Exception, TracerIntegerConversionError)):
        inpl_adder_test(qcla, mode="dynamic")


def test_thapliyal_adder_fails_static():
    """The Thapliyal procedure is a low-level routine that only works on
    qubit registers of size >= 2.  When wrapped as an ``(a, b)`` adder
    without carry propagation for size-1 inputs, the static test catches
    the resulting arithmetic error at the very first iteration
    (``i=1, j=1``): no gates are applied, so ``c`` stays equal to ``b``
    and the assertion ``(a + b) % 2**i == c`` fails for every basis state
    where ``a`` is non-zero.

    The dynamic test cannot even trace through the wrapper because
    ``len(a)`` raises ``TracerIntegerConversionError`` on a traced
    ``QuantumFloat``, so the adder is valid in neither mode."""

    from qrisp.alg_primitives.arithmetic.adders.thapliyal_adder import thapliyal_procedure

    def thap_adder(a, b):
        if not isinstance(a, QuantumFloat):
            q_a = b.duplicate()
            q_a.encode(a % 2 ** b.size)
            thap_adder(q_a, b)
            q_a.delete()
            return
        qs = a.qs
        n = len(a)
        if n > 1:
            thapliyal_procedure(qs, list(a.reg[:n-1]), list(b.reg[:n-1]), b.reg[n-1])

    with pytest.raises(AssertionError, match=r".*failed the static test.*"):
        inpl_adder_test(thap_adder, mode="static")


def test_mishandle_classical_cuccaro():
    """The classical-input path encodes ``-a-1`` (bitwise complement)
    instead of ``a``, causing the classical-quantum arithmetic check to
    fail in both modes."""
    with pytest.raises(AssertionError, match=r".*failed the static test.*"):
        inpl_adder_test(mishandle_classical_cuccaro)


# -----------------------------------------------------------------------------
# Notes on the Thapliyal adder (``thapliyal_procedure``)
# -----------------------------------------------------------------------------
# ``thapliyal_procedure`` in ``qrisp.alg_primitives.arithmetic.adders`` is
# based on https://arxiv.org/abs/1712.02630.  It accepts a ``QuantumCircuit``
# as its first argument — an old signature that predates the current
# ``QuantumSession``-based infrastructure and only works in static mode.
#
# Known issues that a future refactor should address:
#
# 1. **Signature** – ``(qc, qubit_list_1, qubit_list_2, output_qubit)``
#    is incompatible with the ``(a, b, ...)`` convention expected by
#    ``inpl_adder_test``.  It should be changed to accept
#    ``QuantumVariable`` / ``QuantumFloat`` arguments directly.
#
# 2. **Static-only** – The function uses Python ``for i in range(1, n)``,
#    plain-list slicing, and assumes concrete qubit indices.  It cannot be
#    traced by Jasp and needs to be rewritten with ``jrange``,
#    ``DynamicQubitArray``, and ``jnp`` primitives.
#
# 3. **Size-1 edge case** – When both inputs have size 1, the slicing
#    ``qubit_list[:-1]`` produces an empty list and the procedure raises
#    ``IndexError``.  A size-1 adder should fall back to a single ``cx``.
#
# 4. **Unequal-size handling** – The intended contract for the new adder:
#    only the first input's size may be adjusted (reduced for modulo
#    addition, increased via ancillas).  The second input's size must not
#    be changed because extra ancillas used during in-place addition
#    cannot be safely uncomputed and deleted afterwards.
#
# 5. **Controlled variant** – A ``@custom_control``-compatible version
#    should be provided so the adder works inside ``with control(qbl):``
#    blocks in both static and dynamic modes.
#
# 6. **Classical-quantum** – An ``int`` first argument must be converted
#    to a quantum register automatically (e.g. via ``conjugate`` /
#    ``int_encoder``) so that ``inpl_adder(j, qf)`` works in both modes.
# -----------------------------------------------------------------------------
