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

import numpy as np
import jax.numpy as jnp
from qrisp.core import x, xxyy, p
from qrisp.typing import NDArrayLike
from qrisp.jasp import jrange, check_for_tracing_mode

def unbalanced_W_state(
    qv: NDArrayLike,
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

    0. Precompute all required :math:`\theta` angles using
       :math:`r_i = \sqrt{ \sum_{ j = i }^{ n - 1 }{ |a_j| ^ 2 } }`
       and :math:`\theta = 2\arccos(|a_i|\,/\,r_i)`.
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
    a = jnp.asarray(amplitudes, dtype=complex)

    if reversed:
        a = a[::-1]

    n = a.shape[0] # Use the static shape of amplitudes

    if not check_for_tracing_mode():
        if len(qv) != n:
            raise ValueError(
                f"Length of amplitudes ({n}) must match qv.size ({len(qv)})."
            )
        if np.linalg.norm(np.asarray(amplitudes, dtype=complex)) < 1e-15:
            raise ValueError("Amplitude vector must be non-zero.")

    # Normalize so that <a|a> = 1
    norm = jnp.sqrt(jnp.vdot(a, a).real)
    a = a / norm
    abs_a = jnp.abs(a)
    phases = jnp.angle(a)

    # Explicitly handle single-qubit case
    if n == 1:
        x(qv[0])
        p(phases[0], qv[0])
        return

    # --- Step 0: Precomputing angles
    # Precompute remaining values following:
    # r_i = \sqrt{ \sum_{ j = i }^{ n - 1 }{ |a_j| ^ 2 } }
    # abs_a_squared : [a0^2, a1^2, a2^2, a3^2]
    # flip          : [a3^2, a2^2, a1^2, a0^2]
    # cumsum        : [a3^2, a3^2 + a2^2, a3^2 + a2^2 + a1^2 , a3^2 + a2^2 + a1^2 + a0^2]
    # flip          : [a3^2 + a2^2 + a1^2 + a0^2, a3^2 + a2^2 + a1^2 , a3^2 + a2^2 , a3^2]
    abs_a_squared = abs_a ** 2
    remaining_arr = jnp.sqrt(jnp.flip(jnp.cumsum(jnp.flip(abs_a_squared))))
    # Calculate rations for arccos. Replace 0/0 division by 1
    # for `arccos(1) = 0` to do nothing.
    numerators = abs_a[:-1] # Strip one last fraction, as num_{n-1} / rem_{n-1} is not needed.
    denominators = remaining_arr[:-1]
    denominators_no_zeroes = jnp.where(denominators > 1e-15, denominators, 1.0)
    # remaining_arr = 0 only when abs_a = 0, so it is safe.
    ratio_arr = jnp.where(denominators > 1e-15, numerators / denominators_no_zeroes, 1.0)
    # Get precomputed angles. Choose θ so that cos(θ/2) = |a_i| / remaining
    # i.e. qubit i retains exactly magnitude |a_i|
    theta_arr = 2 * jnp.arccos(jnp.clip(ratio_arr, -1.0, 1.0))

    # --- Step 1: place the single excitation on qubit 0  --->  |10...0>
    x(qv[0])

    # --- Step 2: redistribute amplitude along the qubit chain
    for i in jrange(n - 1):
        # XXYY(θ, π/2) performs a parametrized partial swap in the
        # single-excitation subspace {|01>, |10>}:
        #   |10> -> cos(θ/2)|10> - sin(θ/2)|01>
        xxyy(theta_arr[i], jnp.pi / 2, qv[i], qv[i + 1])
        # Imprint the complex phase of a_i onto qubit i
        p(phases[i], qv[i])

    # --- Step 3: imprint the phase on the last qubit
    p(phases[n - 1], qv[n - 1])
