"""********************************************************************************
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

from collections.abc import Sequence
from typing import Literal, TypeAlias, assert_never, get_args

import jax.numpy as jnp
from jax import Array, lax
from jax.scipy.special import gammaln

from qrisp.circuit import Qubit
from qrisp.core import QuantumVariable, cx, ry, x
from qrisp.environments import control
from qrisp.jasp import jlen, jrange, q_cond

# The state preparation methods that :func:`dicke_state` knows about.
DickeStateMethod: TypeAlias = Literal["deterministic", "divide-and-conquer"]

# The same methods as a runtime tuple, derived from ``DickeStateMethod`` so that the two cannot drift apart.
_METHODS: tuple[DickeStateMethod, ...] = get_args(DickeStateMethod)


def dicke_state(
    qv: QuantumVariable | Sequence[Qubit],
    k: int | Array,
    *,
    method: DickeStateMethod = "deterministic",
) -> None:
    r"""Dicke State initialization of a QuantumVariable.

    A Dicke state is an equal positive superposition of all basis states with a given Hamming weight. We label the
    Dicke state of Hamming weight :math:`k` on :math:`n` qubits as :math:`D(n, k)`, where :math:`n` is the size of
    ``qv``. The input is a computational basis state :math:`|0\rangle^{\otimes n-l} |1\rangle^{\otimes l}` of some
    Hamming weight :math:`l`.

    Parameters
    ----------
    qv : QuantumVariable or Sequence[Qubit]
        Initial quantum variable to be prepared. Has to be in the state |00...011...1> where we label the number of 1's
        ``l``. If ``method`` is ``"divide-and-conquer"``, it is required to have ``l == k``. When ``method`` is
        `"deterministic"`, ``l`` may be less or equal to ``k``. In that case, the resulting state is :math:`D(n, l)`.
    k : int
        The Hamming weight (i.e. number of "ones") for the desired Dicke state.
    method : str, optional
        The method used to prepare the state, either ``"deterministic"`` or ``"divide-and-conquer"``. The default is
        ``"deterministic"``. The code largely follows the notation of the respective papers.

        ``"deterministic"`` applies the Dicke state unitary :math:`U_{n,k}` of
        `arXiv:1904.07358 <https://arxiv.org/abs/1904.07358>`_. Because this is a unitary rather than a state
        preparation circuit, it maps *any* input with :math:`l \leq k` to :math:`D(n, l)`, and therefore acts
        linearly on superpositions of such inputs. Use this when the Hamming weight of the input is itself in
        superposition, for instance when assembling more general symmetric states.

        ``"divide-and-conquer"`` implements the preparation of
        `arXiv:2112.12435 <https://arxiv.org/abs/2112.12435>`_. It splits ``qv`` in half and prepares both halves on
        disjoint qubits, so :math:`U_{n_1,k}` and :math:`U_{n_2,k}` are applied in parallel and the circuit depth is
        roughly halved. In exchange it is a state preparation circuit only: the input must have Hamming weight
        exactly :math:`k`.

    Raises
    ------
    ValueError
        If ``method`` is neither "deterministic" nor "divide-and-conquer".
    ValueError
        If not in tracing mode (i.e., when ``k`` and ``n = len(qv)`` are integers) and when ``k`` and ``n`` don't
        satisfy the basic inequality ``0 <= k <= n``.

    Examples
    --------
    We initiate a QuantumVariable in the "0011" state and from this create the Dicke state with Hamming weight 2.

    ::

        from qrisp import QuantumVariable, x, dicke_state

        qv = QuantumVariable(4)
        x(qv[2])
        x(qv[3])

        dicke_state(qv, 2)

    """
    if method not in _METHODS:
        raise ValueError(f"Unknown `method`: {method!r}. Possible methods are: {', '.join(map(repr, _METHODS))}.")

    n = jlen(qv)
    if isinstance(k, int) and isinstance(n, int) and not 0 <= k <= n:
        raise ValueError(f"`k` must satisfy 0 <= k <= n, got k={k} for n={n}.")


    if method == "deterministic":
        _apply_dicke_unitary(qv, n, k)
    elif method == "divide-and-conquer":

        # If k > n/2, it is easier to create D(n, n-k) instead of D(n, k), and then apply the X gate to all qubits.
        large_k = k > n // 2
        # Partially undo the initial state, reducing its Hamming weight from k to n-k (if k > n/2).
        for i in jrange(n - k, jnp.maximum(k, n - k)):
            x(qv[i])
        k = jnp.minimum(k, n - k)  # Equivalent to: `k = n - k if large_k else k`.

        n1 = n // 2  # floor(n/2)
        n2 = (n + 1) // 2  # ceil(n/2)

        _divide(qv, n1, n2, k)

        _apply_dicke_unitary(qv[:n1], n1, k)
        _apply_dicke_unitary(qv[n1:], n2, k)

        # If we were preparing D(n, n-k), now we apply the X gates to transform it into D(n, k).
        q_cond(large_k, x, lambda qv: qv, qv)  # Equivalent to: `if large_k: x(qv)`

    else:
        assert_never(method)


def _log_binom(n: int | Array, k: int | Array) -> Array:
    r"""Compute :math:`\log \binom{n}{k}` in a Jasp/Jax-traceable way.

    Staying in log space keeps the intermediate values finite: :math:`\binom{n}{k}` itself overflows a 64 bit float
    at around :math:`n = 1030`, while its logarithm does not.

    Parameters
    ----------
    n : int
        The size of the set to choose from. May be a traced value.
    k : int
        The number of elements to choose. May be a traced value.

    Returns
    -------
    Array
        A float64 scalar holding :math:`\log \binom{n}{k}`, or ``-jnp.inf`` outside of ``0 <= k <= n``, i.e. wherever
        the binomial coefficient vanishes.

    """
    n_f = jnp.asarray(n, dtype=jnp.float64)
    k_f = jnp.asarray(k, dtype=jnp.float64)
    return gammaln(n_f + 1.0) - gammaln(k_f + 1.0) - gammaln(n_f - k_f + 1.0)


def _divide(qv: QuantumVariable | Sequence[Qubit], n1: int | Array, n2: int | Array, k: int | Array) -> None:
    r"""Execute the "divide" step of the "divide-and-conquer" method of creating Dicke states.

    This takes a computational basis state consisting of ``n1 + n2 - k`` zeros followed by
    ``k`` ones and changes it into a superposition

    .. math::

        \frac{1}{\sqrt{\binom{n_1+n_2}{k}}}
        \sum_{k_1 = 0}^k
        \sqrt{\binom{n_1}{k_1} \binom{n_2}{k-k_1}}
        |0\rangle^{\otimes n_1-k_1}
        |1\rangle^{\otimes k_1}
        |0\rangle^{\otimes n_2-k+k_1}
        |1\rangle^{\otimes k-k_1}

    ready for the "conquer" step.

    Parameters
    ----------
    qv : QuantumVariable or Sequence[Qubit]
        The quantum variable to be divided. Has to be in the state |00...011...1> where the number of 1's is equal
        to ``k``.
    n1 : int
        The size of the first half of the quantum variable.
    n2 : int
        The size of the second half of the quantum variable.
    k : int
        The Hamming weight (i.e. number of "ones") of the Dicke state to be constructed.

    """

    def log_x(i: int | Array) -> Array:
        r"""Compute :math:`\log x_i`, following :math:`x_i` from page 8 of https://arxiv.org/abs/2112.12435.

        That is, :math:`x_i = \binom{n_1}{i} \binom{n_2}{k-i}`, equal to ``-jnp.inf`` in log space whenever one of
        the two binomial coefficients vanishes.
        """
        return _log_binom(n1, i) + _log_binom(n2, k - i)

    def ratio_x_s(i: int | Array) -> Array:
        r"""Compute :math:`x_i / s_i`, where :math:`s_i = \sum_{j \geq i} x_j`, without forming either quantity.

        Both :math:`x_i` and :math:`s_i` may easily overflow. Rather than computing the two and dividing, we rearrange
        into a form in which only the ratio ever appears:

        .. math::

            \frac{x_i}{s_i} = \frac{x_i}{\sum_{j \geq i} x_j}
                            = \left( \sum_{j \geq i} \frac{x_j}{x_i} \right)^{-1}
                            = \left( \sum_{j \geq i} e^{\log x_j - \log x_i} \right)^{-1}

        The sum is recomputed from scratch for every ``i`` rather than carried between iterations, so that each
        iteration of the quantum loop depends only on ``i`` and the loop stays invertible. This costs
        :math:`\mathcal{O}(k)` per step, but it is classical arithmetic with no quantum operations in it.
        """
        log_x_i = log_x(i)
        total = lax.fori_loop(
            i, k + 1,
            lambda j, acc: acc + jnp.exp(log_x(j) - log_x_i),
            jnp.asarray(0.0, dtype=jnp.float64),
        )
        return 1.0 / total

    def angle(i: int | Array) -> Array:
        r"""Compute the rotation angle :math:`2 \arccos \sqrt{x_i / s_i}` for the (controlled) RY gate at step ``i``."""
        return 2 * jnp.arccos(jnp.sqrt(ratio_x_s(i)))

    # i = 0 is the only iteration without a control qubit.
    for _ in jrange(jnp.where(k > 0, 1, 0)):
        ry(angle(0), qv[n1 - 1])

    for i in jrange(1, jnp.maximum(k, 1)):
        with control(qv[n1 - i]):
            ry(angle(i), qv[n1 - 1 - i])

    for i in jrange(k):
        cx(qv[n1 - 1 - i], qv[n1 + n2 - k + i])


def _apply_dicke_unitary(qv: QuantumVariable | Sequence[Qubit], n: int | Array, k: int | Array) -> None:
    """Apply the Dicke unitary constructed according to Lemma 2 in https://arxiv.org/abs/1904.07358.

    Parameters
    ----------
    qv : QuantumVariable or Sequence[Qubit]
        Initial quantum variable to be prepared. Has to be in target subspace.
    n : int
        The size of the quantum variable.
    k : int
        The Hamming weight (i.e. number of "ones") of the Dicke state to be constructed.

    """
    for offset in jrange(jnp.where(k > 0, n - k, 0)):  # If `k == 0`, we don't execute anything. D(n, 0) = |00 ... 0>
        index2 = n - offset
        split_cycle_shift(qv, index2, k)

    for offset in jrange(jnp.maximum(k - 1, 0)):
        index = k - offset
        split_cycle_shift(qv, index, index - 1)


def split_cycle_shift(qv: QuantumVariable | Sequence[Qubit], highIndex: int | Array, lowIndex: int | Array) -> None:
    """Apply the *Split & Cyclic Shift* unitary :math:`SCS_{n, k}` defined in https://arxiv.org/abs/1904.07358.

    Helper function for Dicke State initialization of a QuantumVariable. The unitary is applied to `qv` in place.

    Parameters
    ----------
    qv : QuantumVariable or Sequence[Qubit]
        Initial quantum variable to be prepared. Has to be in target subspace.
    highIndex : int
        Index for indication of preparation steps, as seen in original algorithm.
    lowIndex : int
        Index for indication of preparation steps, as seen in original algorithm.

    """
    # index == highIndex
    param = 2 * jnp.arccos(jnp.sqrt(1 / highIndex))
    cx(qv[highIndex - 2], qv[highIndex - 1])
    with control(qv[highIndex - 1]):
        ry(param, qv[highIndex - 2])
    cx(qv[highIndex - 2], qv[highIndex - 1])

    # index != highIndex
    for i in jrange(1, lowIndex):
        index = highIndex - i
        param = 2 * jnp.arccos(jnp.sqrt((highIndex - index + 1) / (highIndex)))

        cx(qv[index - 2], qv[highIndex - 1])
        with control([qv[highIndex - 1], qv[index - 1]]):
            ry(param, qv[index - 2])
        cx(qv[index - 2], qv[highIndex - 1])
