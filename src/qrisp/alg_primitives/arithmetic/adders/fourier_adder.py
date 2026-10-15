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

"""Implements the QFT-based (Draper) in-place adder and its QuasiRZZ helper gate.

The :func:`fourier_adder` function works in both static and dynamic (Jasp) mode.
"""

import jax
import numpy as np

from qrisp.alg_primitives import QFT
from qrisp.circuit import Operation, PGate, QuantumCircuit
from qrisp.core.gate_application_functions import cp, cz, h, p, swap
from qrisp.environments import QuantumEnvironment, conjugate, control
from qrisp.jasp import DynamicQubitArray, check_for_tracing_mode, jrange, qache
from qrisp.qtypes import QuantumFloat


@qache
def fourier_adder(a, b, perform_QFT=True):  # noqa: PLR0912
    """In-place adder function based on `this paper <https://arxiv.org/abs/quant-ph/0008033>`__.

    Performs the addition:

    ::

        b += a

    This function works in both static and dynamic (Jasp) mode. In dynamic
    mode it is Jasp-traceable and additionally accepts ``BigInteger`` inputs.

    Parameters
    ----------
    a : int or BigInteger or QuantumVariable or list[Qubit]
        The value that should be added.
    b : QuantumVariable or list[Qubit]
        The value that should be modified in the in-place addition.
    perform_QFT : bool, optional
        If set to ``False``, no QFT is executed. The default is ``True``.
        This parameter is only used in static mode.

    Examples
    --------
    We add two integers:

    >>> from qrisp import QuantumFloat, fourier_adder
    >>> a = QuantumFloat(4)
    >>> b = QuantumFloat(4)
    >>> a[:] = 4
    >>> b[:] = 5
    >>> fourier_adder(a,b)
    >>> print(b)
    {9: 1.0}

    """
    if check_for_tracing_mode():
        # Dynamic (Jasp) mode.
        #
        # BigInteger is imported lazily because importing it at module level
        # triggers a circular import:
        #
        #   fourier_adder -> BigInteger (from jasp_bigintiger)
        #     -> jasp_arithmetic/__init__ -> adders.fourier_adder (circular!)
        from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import (
            BigInteger,
        )

        if isinstance(a, BigInteger):

            @jax.jit
            def get_size(a):
                return a.bit_size()

            n_a = get_size(a)
            a = a()
        else:
            n_a = a.size

        n_b = b.size

        with conjugate(_qft)(b):
            if isinstance(a, (QuantumFloat, DynamicQubitArray)):
                for i in jrange(n_a):
                    with control(a[i]):
                        for j in jrange(n_b):
                            p(np.pi * 2.0 ** (j - n_b + 1 + i), b[j])

            else:
                for i in jrange(n_b):
                    p(a * np.pi * 2.0 ** (i - n_b + 1), b[i])

        return

    if perform_QFT:
        env = conjugate(QFT)(b, exec_swap=False)
    else:
        env = QuantumEnvironment()

    with env:
        b = list(b)
        b = b[::-1]

        if isinstance(a, (int, np.integer)):
            for i in range(len(b)):
                p(a * np.pi * 2 ** (1 + i - len(b)), b[i])

        else:
            if len(a) > len(b):
                raise Exception("Tried to add QuantumFloat of higher precision onto QuantumFloat of lower precision")

            phase_correction_a = np.zeros(len(a))
            phase_correction_b = np.zeros(len(b))
            for j in range(len(a)):
                for i in range(len(b)):
                    if 1 + j + i - len(b) >= 1:
                        continue
                    if 1 + j + i - len(b) == 0:
                        cz(a[j], b[i])
                    else:
                        b[i].qs().append(
                            QuasiRZZ(-np.pi * 2 ** (1 + j + i - len(b)) / 2),
                            [a[j], b[i]],
                        )

                        phase_correction_a[j] += np.pi * 2 ** (1 + j + i - len(b)) / 2
                        phase_correction_b[i] += np.pi * 2 ** (1 + j + i - len(b)) / 2

            for i in range(len(b)):
                if phase_correction_b[i] % (2 * np.pi) != 0:
                    p(phase_correction_b[i], b[i])
            for i in range(len(a)):
                if phase_correction_a[i] % (2 * np.pi) != 0:
                    p(phase_correction_a[i], a[i])


class QuasiRZZ(Operation):
    """A phase gate conjugated by two CX gates, used by the QFT-based adder."""

    def __init__(self, angle):
        """Create a ``QuasiRZZ`` gate rotating by ``angle`` radians."""
        qc = QuantumCircuit(2)
        qc.cx(qc.qubits[1], qc.qubits[0])
        qc.p(angle, qc.qubits[0])
        qc.cx(qc.qubits[1], qc.qubits[0])

        Operation.__init__(self, "quasi_rzz", num_qubits=2, definition=qc, params=[angle])

        self.permeability = {0: True, 1: True}
        self.is_qfree = True

    def inverse(self):
        """Return the inverse gate, which negates the rotation angle."""
        return QuasiRZZ(-self.params[0])

    def control(self, num_ctrl_qubits=1, ctrl_state=-1, method=None):
        """Return the controlled version of this gate."""
        qc = QuantumCircuit(2 + num_ctrl_qubits)
        qc.cx(qc.qubits[-1], qc.qubits[-2])
        qc.append(
            PGate(self.params[0]).control(num_ctrl_qubits, ctrl_state=ctrl_state, method=method),
            qc.qubits[:-1],
        )
        qc.cx(qc.qubits[-1], qc.qubits[-2])

        res = Operation.control(self, num_ctrl_qubits, ctrl_state=ctrl_state, method=method)

        res.definition = qc
        return res


@qache
def _qft(qv):
    """Performs qft on the first n qubits in circuit (without swaps)"""
    n = qv.size

    for i in jrange(n):
        # pass
        h(qv[n - 1 - i])
        for k in jrange(n - i - 1):
            cp(2.0 * np.pi / pow(2.0, (k + 2)), qv[n - 1 - (k + i + 1)], qv[n - 1 - i])

    for i in jrange(n // 2):
        swap(qv[i], qv[n - i - 1])
