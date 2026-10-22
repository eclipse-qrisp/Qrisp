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

"""Implements the iterative quantum phase estimation (IQPE) algorithm using a single ancilla qubit."""

# Step j of the iterative (or semiclassical) phase estimation applies U^(2^(t-1-j))
# controlled on a fresh qubit, removes the phase of the bits measured before with a
# rotation, and measures bit j of the outcome y, from the least significant one. The
# estimated phase is y / 2**t, and the outcome has the same distribution as the
# measurement of a t-qubit phase register after the inverse QFT (Griffiths and Niu,
# Phys. Rev. Lett. 76, 3228, 1996).

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer

from qrisp import QuantumBool, control, h, measure, reset, rz
from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import DTYPE
from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import _LIMB_BITS
from qrisp.jasp import jrange, q_fori_loop

# The module is star-imported into qrisp, which should not receive its imports
__all__ = ["IQPE"]

# IQPE applies U up to 2**(precision - 1) times, counted with 64-bit integers
_MAX_IQPE_PRECISION = 63


def _semiclassical_phase_estimation(
    args: Any,
    apply_power: Callable[[Any, Array], None],
    precision: int,
    ctrl_method: str | None = None,
    outcome_limbs: bool = False,
) -> tuple[Array, Array]:
    """Estimate the phase of a unitary ``U`` with one control qubit at a time, measured after every step.

    Parameters
    ----------
    args : Any
        The arguments on which ``U`` acts, passed on to ``apply_power``.
    apply_power : Callable
        ``apply_power(args, k)`` applies ``U`` to the power ``2**k``, for an integer
        ``k`` that is traced.
    precision : int
        The number ``t`` of bits of the outcome.
    ctrl_method : str, optional
        The method of the controlled ``U``, see :meth:`.control <qrisp.Operation.control>`.
        The default is None.
    outcome_limbs : bool, optional
        If True, the outcome ``y`` is also assembled into its ``ceil(t / 32)`` 32-bit
        limbs, which requires a precision that is not traced. The default is False,
        which computes only the estimated phase.

    Returns
    -------
    tuple
        The estimated phase ``y / 2**t`` as a float, exact for up to 53 bits, and the
        limbs of the outcome ``y`` (the digits of a BigInteger), of dtype ``uint32``,
        or no limbs if ``outcome_limbs`` is False.

    Raises
    ------
    TypeError
        If ``outcome_limbs`` is True and the precision is traced or not an integer.

    """
    # The number of limbs sets the shape of an array, so it must be known while tracing
    num_limbs = -(-operator.index(precision) // _LIMB_BITS) if outcome_limbs else 0

    def step(j: Array, carry: tuple) -> tuple:
        """Measure bit ``j`` of the outcome.

        Parameters
        ----------
        j : Array
            The step, from 0 to ``precision - 1``.
        carry : tuple
            The phase of the bits measured before (as a fraction of a turn), the
            arguments of ``U`` and the limbs of the outcome.

        Returns
        -------
        tuple
            The carry after the step.

        """
        phase, args, digits = carry
        control_qubit = QuantumBool()
        h(control_qubit)
        with control(control_qubit[0], ctrl_method=ctrl_method):
            apply_power(args, precision - 1 - j)
        # For a phase y / 2**t, the power 2**(t - 1 - j) gives the phase 0.y_j y_(j-1) ... y_0
        # in binary, and ``phase`` holds 0.y_(j-1) ... y_0. The rotation removes half of it,
        # which leaves the phase y_j / 2: the Hadamard gate maps it to |y_j>. At step 0
        # the angle is 0: the rotation is kept, so that every step has the same circuit.
        rz(-np.pi * phase, control_qubit)
        h(control_qubit)
        bit = measure(control_qubit)
        # A released qubit must be in |0>, since interpreters that reuse it (such as the
        # Catalyst and static-register interpreters) do not reset it
        reset(control_qubit)
        control_qubit.delete()

        if num_limbs:
            limb = j // _LIMB_BITS
            shift = jnp.asarray(j % _LIMB_BITS, dtype=DTYPE)
            digits = digits.at[limb].set(digits[limb] | (jnp.asarray(bit, dtype=DTYPE) << shift))
        return ((phase + bit) / 2, args, digits)

    initial = (jnp.asarray(0.0), args, jnp.zeros(num_limbs, dtype=DTYPE))
    phase, _, digits = q_fori_loop(0, precision, step, initial)
    return phase, digits


# The public signature keeps its six parameters, which callers may pass positionally
def IQPE(  # noqa: PLR0913, PLR0917
    args: Any,
    U: Callable,
    precision: int,
    iter_spec: bool = False,
    ctrl_method: str | None = None,
    kwargs: dict | None = None,
) -> Array:
    r"""Estimate the phase of a unitary with the iterative quantum phase estimation algorithm.

    The algorithm is described in `this paper <https://arxiv.org/pdf/quant-ph/0610214>`_.
    The unitary to estimate is expected to be given as Python function, which is called
    on ``args``.

    Instead of a register of ``precision`` qubits, the algorithm uses a single control
    qubit at a time. Step $j$ applies $U^{2^{t - 1 - j}}$ controlled on a fresh qubit,
    where $t$ is the precision, corrects the phase of the bits measured before, and
    measures bit $j$ of the estimate, from the least significant one.

    Parameters
    ----------
    args : list
        A list of arguments (could be QuantumVariables) which represent the state,
        the quantum phase estimation is performed on.
    U : function
        A Python function, which will receive the list ``args`` as arguments in the
        course of this algorithm.
    precision : int
        The precision of the estimation.
    iter_spec : bool, optional
        If set to ``True``, ``U`` will be called with the additional keyword
        ``iter = i`` where ``i`` is the amount of iterations to perform (instead of
        simply calling ``U`` for ``i`` times). The default is False.
    ctrl_method : str, optional
        Allows to specify which method should be used to generate the
        controlled U circuit. For more information check
        :meth:`.control <qrisp.Operation.control>`. The default is None.
    kwargs : dict, optional
        A dictionary of keyword arguments to pass to ``U``. The default is None,
        which passes no keyword arguments.

    Returns
    -------
    float
        The estimated phase as a fraction of $2 \pi$.

    Raises
    ------
    TypeError
        If ``precision`` is not an integer.
    ValueError
        If ``precision`` is larger than 63. A traced precision, for example an
        argument of a jaspified function, is not checked, since its value is only
        known when the program runs.

    Notes
    -----
    Without ``iter_spec``, the first step calls ``U`` $2^{t-1}$ times and the last
    one once, $2^t - 1$ calls in total. The number of iterations $2^{t-1}$ is a
    64-bit integer, also with ``iter_spec``, so the precision is at most 63. The
    estimate is a float, exact for a precision of up to 53 bits. For the order
    finding of Shor's algorithm with thousands of bits, see
    :func:`~qrisp.shor.semiclassical_order_finding`.

    Examples
    --------
    We define a function that applies two rotations onto its input and estimate the
    applied phase. ::

        from qrisp import IQPE, h, run, x, rx, QuantumFloat
        from qrisp.jasp import make_jaspr
        import numpy as np

        def f():
            def U(qv):
                x = 1/2**3
                y = 1/2**2

                rx(x*2*np.pi, qv[0])
                rx(y*2*np.pi, qv[1])

            qv = QuantumFloat(2)

            x(qv)
            h(qv)

            return IQPE(qv, U, precision = 4)
        jaspr = make_jaspr(f)()

    Each qubit is in the state $\lvert - \rangle$, on which $R_X(2 \pi x)$ has the
    eigenvalue $e^{i \pi x}$, so the phase is $(1/8 + 1/4) / 2 = 0.1875$:

    >>> jaspr()
    Array(0.1875, dtype=float64)

    """
    # A traced precision is only known when the program runs, so it cannot be checked here
    if not isinstance(precision, Tracer) and operator.index(precision) > _MAX_IQPE_PRECISION:
        raise ValueError(f"The precision of IQPE is at most {_MAX_IQPE_PRECISION}, got {precision}")
    kwargs = {} if kwargs is None else kwargs

    def apply_power(args: Any, k: Array) -> None:
        """Apply ``U`` to the power ``2**k``.

        Parameters
        ----------
        args : Any
            The arguments of ``U``.
        k : Array
            The exponent of the power of two.

        """
        if iter_spec:
            U(args, iter=2**k, **kwargs)
        else:
            for _ in jrange(2**k):
                U(args, **kwargs)

    return _semiclassical_phase_estimation(args, apply_power, precision, ctrl_method=ctrl_method)[0]
