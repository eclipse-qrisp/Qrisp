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

#: The state preparation methods that ``dicke_state`` knows about.
DickeStateMethod: TypeAlias = Literal["deterministic", "divide-and-conquer"]

# The same methods as a runtime tuple, derived from ``DickeStateMethod`` so that the two cannot drift apart.
_METHODS: tuple[DickeStateMethod, ...] = get_args(DickeStateMethod)


def dicke_state(
    qv: QuantumVariable | Sequence[Qubit],
    k: int | Array,
    *,
    method: DickeStateMethod = "deterministic",
) -> None:
    r"""Prepare a Dicke state :math:`|D^n_k\rangle` on a QuantumVariable.

    A Dicke state is the equal superposition of all basis states of Hamming weight :math:`k` on :math:`n` qubits, where
    :math:`n` is the number of qubits of ``qv``.

    ``qv`` has to be initialized to the basis state :math:`|0\rangle^{\otimes n-l}|1\rangle^{\otimes l}` beforehand, as
    in the example below. ``"divide-and-conquer"`` requires :math:`l = k`; ``"deterministic"`` accepts any
    :math:`l \leq k` (see ``method``).

    Parameters
    ----------
    qv : QuantumVariable or Sequence[Qubit]
        The qubits to prepare, initialized as described above.
    k : int
        The Hamming weight (i.e. the number of "ones") of the desired Dicke state.
    method : {"deterministic", "divide-and-conquer"}, optional
        Either ``"deterministic"`` (`arXiv:1904.07358 <https://arxiv.org/abs/1904.07358>`_, the default) or
        ``"divide-and-conquer"`` (`arXiv:2112.12435 <https://arxiv.org/abs/2112.12435>`_). The latter prepares the two
        halves of ``qv`` on disjoint qubits and therefore has roughly half the circuit depth. The former is the full
        Dicke state unitary :math:`U_{n,k}`: it also maps an input of Hamming weight :math:`l \leq k` to
        :math:`|D^n_l\rangle`, and superpositions of such inputs to the corresponding superposition of Dicke states.

    Raises
    ------
    ValueError
        If ``method`` is unknown, or if ``k`` and :math:`n` are plain Python integers (i.e. outside of tracing) and
        violate :math:`0 \leq k \leq n`.

    Examples
    --------
    We initialize a QuantumVariable in the "0011" state and from this create the Dicke state with
    Hamming weight 2.

    ::

        from qrisp import QuantumVariable, x, dicke_state

        qv = QuantumVariable(4)
        x(qv[2])
        x(qv[3])

        dicke_state(qv, 2)

        print(qv)

    Under Jasp, the Hamming weight may be a traced value. Here we prepare the same state with the
    shallower divide-and-conquer circuit.

    ::

        from qrisp import QuantumVariable, x, dicke_state
        from qrisp.jasp import jrange, terminal_sampling

        @terminal_sampling
        def main(k):
            qv = QuantumVariable(4)
            for i in jrange(4 - k, 4):
                x(qv[i])
            dicke_state(qv, k, method="divide-and-conquer")
            return qv

        print(main(2))

    """
    if method not in _METHODS:
        raise ValueError(f"Unknown `method`: {method!r}. Possible methods are: {', '.join(map(repr, _METHODS))}.")

    n = jlen(qv)
    if isinstance(k, int) and isinstance(n, int) and not 0 <= k <= n:
        raise ValueError(f"`k` must satisfy 0 <= k <= n, got k={k} for n={n}.")

    if method == "deterministic":
        _apply_dicke_unitary(qv, n, k)
    elif method == "divide-and-conquer":
        # The divide-and-conquer method cannot handle k > n/2.
        # Instead we prepare D(n, n-k) and at the end apply the X gate to all qubits, which changes it to D(n, k).
        large_k = k > n // 2
        # Reduce the input |0^(n-k) 1^k> to |0^k 1^(n-k)>. Empty range unless k > n/2.
        for i in jrange(n - k, jnp.maximum(k, n - k)):
            x(qv[i])
        k = jnp.minimum(k, n - k)  # Equivalent to: `k = n - k if large_k else k`.

        n1 = n // 2  # floor(n/2)
        n2 = (n + 1) // 2  # ceil(n/2)

        _divide(qv, n1, n2, k)  # k <= n1 now, which `_divide` needs.

        # Disjoint qubits, so the compiler runs these in parallel — this is the depth advantage.
        _apply_dicke_unitary(qv[:n1], n1, k)
        _apply_dicke_unitary(qv[n1:], n2, k)

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
    r"""Execute the "divide" step of the divide-and-conquer method of https://arxiv.org/abs/2112.12435.

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

    that splits the target Hamming weight ``k`` over the two halves of the register with the correct
    binomial weights. Each half is then completed independently by ``_apply_dicke_unitary``, which is
    the "conquer" step. Note that each half is left in exactly the form that ``_apply_dicke_unitary``
    expects: zeros followed by ones.

    The notation follows the paper: :math:`n_1, n_2` are the sizes of the two halves and :math:`x_i`,
    :math:`s_i` are the quantities on page 8, defined in the nested helpers below.

    Parameters
    ----------
    qv : QuantumVariable or Sequence[Qubit]
        The quantum variable to be divided. Has to be in the state :math:`|0\rangle^{\otimes n-k}|1\rangle^{\otimes k}`.
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
            i,
            k + 1,
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

    # Reduce the Hamming weight of the 2nd half controlled by the state (Hamming weight) of the 1st half.
    for i in jrange(k):
        cx(qv[n1 - 1 - i], qv[n1 + n2 - k + i])


def _apply_dicke_unitary(qv: QuantumVariable | Sequence[Qubit], n: int | Array, k: int | Array) -> None:
    r"""Apply the Dicke state unitary :math:`U_{n,k}` from Lemma 2 of https://arxiv.org/abs/1904.07358.

    :math:`U_{n,k}` is built as a ladder of *Split & Cyclic Shift* blocks :math:`SCS_{n,k}`. It is
    defined by :math:`U_{n,k} |0\rangle^{\otimes n-l} |1\rangle^{\otimes l} = |D^n_l\rangle` for *every*
    :math:`l \leq k`, not just for :math:`l = k` (arXiv:2112.12435, Eq. 2), and extends linearly to
    superpositions of such inputs.

    That is why no :math:`k \to n-k` reduction may be applied here, even though it would shorten the
    circuit: it is only valid for an input of Hamming weight exactly :math:`k` and would silently
    destroy the :math:`l < k` branches. ``_divide`` does assume weight exactly :math:`k`, so the
    reduction lives in ``dicke_state`` on the divide-and-conquer path only.

    Parameters
    ----------
    qv : QuantumVariable or Sequence[Qubit]
        Initial quantum variable to be prepared. Has to be in the basis state
        :math:`|0\rangle^{\otimes n-l} |1\rangle^{\otimes l}` for some :math:`l \leq k`, or in a superposition of such
        states.
    n : int
        The size of the quantum variable.
    k : int
        The Hamming weight (i.e. number of "ones") of the Dicke state to be constructed.

    """
    for offset in jrange(jnp.where(k > 0, n - k, 0)):  # If `k == 0`, we don't execute anything. D(n, 0) = |00 ... 0>
        index2 = n - offset
        _split_cycle_shift(qv, index2, k)

    for offset in jrange(jnp.maximum(k - 1, 0)):
        index = k - offset
        _split_cycle_shift(qv, index, index - 1)


def _split_cycle_shift(qv: QuantumVariable | Sequence[Qubit], n: int | Array, k: int | Array) -> None:
    """Apply the *Split & Cyclic Shift* unitary :math:`SCS_{n, k}` defined in https://arxiv.org/abs/1904.07358.

    Helper function for Dicke State initialization of a QuantumVariable. The construction follows section 2.2. of the
    above-linked paper. The unitary is applied to ``qv`` in place.

    Parameters
    ----------
    qv : QuantumVariable or Sequence[Qubit]
        Initial quantum variable to be prepared. Has to be in target subspace.
    n : int
        Index ``n`` for indication of preparation steps, as seen in original algorithm.
    k : int
        Index ``k`` for indication of preparation steps, as seen in original algorithm.

    """
    # Qubit labels are off by one, since Qrisp labels qubits starting from 0 whereas the paper starts from 1.
    # l = 1
    param = 2 * jnp.arccos(jnp.sqrt(1 / n))
    cx(qv[n - 2], qv[n - 1])
    with control(qv[n - 1]):
        ry(param, qv[n - 2])
    cx(qv[n - 2], qv[n - 1])

    # 2 <= l <= k
    for l in jrange(2, k + 1):  # noqa: E741
        param = 2 * jnp.arccos(jnp.sqrt(l / n))

        cx(qv[n - l - 1], qv[n - 1])
        with control([qv[n - 1], qv[n - l]]):
            ry(param, qv[n - l - 1])
        cx(qv[n - l - 1], qv[n - 1])
