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

import jax.numpy as jnp
from qrisp import QuantumVariable, Qubit, x, xxyy, p
from qrisp.jasp import jlen, q_fori_loop
from collections.abc import Sequence

def unbalanced_W_state(
    qv: QuantumVariable | Sequence[Qubit],
    amplitudes: list,
    reversed: bool = False
) -> None:
    r"""
    Prepare a generalized W state (unbalanced Dicke state of Hamming weight 1)
    on the given :ref:`QuantumVariable`.

    The resulting quantum state is

    .. math::

        |\psi\rangle \;=\; \sum_{i=0}^{n-1} a_i \,|e_i\rangle

    where :math:`|e_i\rangle` is the computational basis state with a single
    ``1`` at position `i`, and :math:`a_i` are the (possibly complex)
    amplitudes given by ``amplitudes``. The input array is automatically
    normalized so that :math:`\sum_i |a_i|^2 = 1`.

    Parameters
    ----------
    qv : QuantumVariable
        A freshly allocated :ref:`QuantumVariable` in the
        :math:`|0\dots0\rangle` state whose size matches ``len(amplitudes)``.
    amplitudes : array_like
        A 1-D sequence of complex (or real) target amplitudes, one per qubit.
        Its length must equal ``qv.size``.
    reversed : bool, optional
        If ``True``, reverse the order of the received amplitudes before
        preparing the state. Default is ``False``

    Raises
    ------
    ValueError
        If ``len(amplitudes) != qv.size`` or if the amplitude vector is zero.

    Notes
    -----
    **Algorithm.**
    The circuit distributes a single excitation across all qubits using a
    linear chain of ``XXYY`` gates:

    1. Apply ``X`` to qubit 0, producing :math:`|10\dots0\rangle`.
    2. For each qubit :math:`i = 0, \dots, n{-}2`:

       a. Compute :math:`\theta = 2\arccos(|a_i|\,/\,r_i)` where :math:`r_i`
          is the remaining (undistributed) amplitude magnitude.
       b. Apply ``XXYY(θ, π/2)`` on qubits :math:`(i,\, i{+}1)`.  In the
          single-excitation subspace this acts as a parametrized partial swap,
          leaving magnitude :math:`|a_i|` on qubit :math:`i` and passing the
          rest to qubit :math:`i{+}1`.
       c. Apply a phase gate :math:`P(\arg a_i)` on qubit :math:`i` to imprint
          the correct complex phase.

    3. Apply :math:`P(\arg a_{n-1})` on the last qubit.

    **Resources.**
    The circuit uses :math:`n{-}1` ``XXYY`` gates (each decomposable into
    2 CNOTs + single-qubit rotations) and :math:`n` phase gates, yielding
    :math:`\mathcal{O}(n)` depth and gate count.

    Examples
    --------
    >>> import numpy as np
    >>> from qrisp import QuantumVariable
    >>> a = np.array([1j, 2, 3, 4])
    >>> qv = QuantumVariable(4)
    >>> unbalanced_W_state(qv, a)
    >>> print(qv.qs.statevector())
    """
    n = jlen(qv)
    a = jnp.asarray(amplitudes, dtype=complex)

    if reversed:
        a = a[::-1]

    # Normalize so that <a|a> = 1
    norm = jnp.sqrt(jnp.vdot(a, a).real)
    a = a / norm
    abs_a = jnp.abs(a)

    # --- Step 1: place the single excitation on qubit 0  --->  |10...0>
    x(qv[0])

    # --- Step 2: redistribute amplitude along the qubit chain
    # `remaining` (i.e. `carry`) tracks the magnitude still carried by the "active" qubit
    # (the one that has not yet been peeled off).
    def body_for(i, carry):
        # Choose θ so that cos(θ/2) = |a_i| / remaining (i.e. carry),
        # i.e. qubit i retains exactly magnitude |a_i|.
        theta = 2 * jnp.arccos(jnp.clip(abs_a[i] / carry, -1.0, 1.0))

        # XXYY(θ, π/2) performs a parametrized partial swap in the
        # single-excitation subspace {|01>, |10>}:
        #   |10> -> cos(θ/2)|10> - sin(θ/2)|01>
        xxyy(theta, jnp.pi / 2, qv[i], qv[i + 1])

        # Update the undistributed amplitude magnitude for the next step
        carry *= jnp.sin(theta / 2)

        # Imprint the complex phase of a_i onto qubit i
        p(jnp.angle(a[i]), qv[i])

        return carry

    remaining = 1.0
    q_fori_loop(0, n-1, body_for, remaining)

    # --- Step 3: imprint the phase on the last qubit
    p(jnp.angle(a[-1]), qv[-1])
