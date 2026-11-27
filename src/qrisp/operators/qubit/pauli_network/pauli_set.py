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

"""Symplectic binary representation of sets of Pauli strings (arXiv:2404.03280, Sec. 2.1)."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # importing at runtime would be circular: qubit_operator imports PauliSet
    from qrisp.operators import QubitOperator

BoolArray: TypeAlias = npt.NDArray[np.bool_]
IntArray: TypeAlias = npt.NDArray[np.intp]
# One axis of a PauliSet index: a qubit/string, a slice, an integer array of them, or a boolean mask.
Index: TypeAlias = int | np.integer | slice | npt.NDArray[np.integer] | BoolArray


class PauliSet:
    r"""The table $M$ of $m$ Pauli strings on $n$ qubits, with a sign per string.

    Pauli string $k$ is stored as the columns ``z[:, k]`` and ``x[:, k]``, where qubit
    $q$ carries $I = (0,0)$, $X = (0,1)$, $Z = (1,0)$ or $Y = (1,1)$ as ``(z, x)``, and
    ``signs[k]`` is ``True`` when the string carries a factor $-1$.

    The gate methods conjugate the whole set in place, $P_k \mapsto G P_k G^\dagger$,
    using the row operations of the paper plus the sign update it omits. Every string
    starts with sign $+1$; only conjugation changes it.

    Parameters
    ----------
    z : numpy.ndarray
        Boolean array of shape ``(n, m)``.
    x : numpy.ndarray
        Boolean array of shape ``(n, m)``.

    """

    z: BoolArray
    x: BoolArray
    signs: BoolArray

    def __init__(self, z: npt.ArrayLike, x: npt.ArrayLike) -> None:
        """Store ``z`` and ``x`` as boolean arrays, with every sign $+1$."""
        z = np.array(z, dtype=bool, ndmin=2)
        x = np.array(x, dtype=bool, ndmin=2)
        if z.shape != x.shape:
            raise ValueError(f"z and x must have the same shape, got {z.shape} and {x.shape}")
        self.z = z
        self.x = x
        self.signs = np.zeros(z.shape[1], dtype=bool)

    @classmethod
    def from_operator(cls, operator: QubitOperator) -> PauliSet:
        """The Pauli strings of ``operator``, one column per term.

        Parameters
        ----------
        operator : QubitOperator
            A sum of Pauli strings: every factor is ``X``, ``Y`` or ``Z``. Coefficients are ignored.

        Returns
        -------
        pauli_set : PauliSet
            Column ``k`` encodes the ``k``-th term of ``operator.coeff_by_term``, on
            ``operator.find_minimal_qubit_amount()`` qubits.
            An identity term is an all-zero column.

        Raises
        ------
        ValueError
            If a term has a ladder or projector factor (``A``, ``C``, ``P0``, ``P1``).

        """
        terms = operator.coeff_by_term.keys()
        n = operator.find_minimal_qubit_amount()
        z = np.zeros((n, len(terms)), dtype=bool)
        x = np.zeros_like(z)
        for k, term in enumerate(terms):
            if not set(term.factor_by_qubit.values()) <= {"X", "Y", "Z"}:
                raise ValueError(
                    f"{term} is not a Pauli string; convert the operator with .hermitize().to_pauli() first"
                )
            x[:, k], z[:, k] = term.binary_representation(n)
        return cls(z, x)

    @property
    def num_qubits(self) -> int:
        """The number of qubits $n$."""
        return self.z.shape[0]

    @property
    def num_paulis(self) -> int:
        """The number of strings $m$."""
        return self.z.shape[1]

    def __len__(self) -> int:
        """Number of strings, so an empty set is falsy."""
        return self.num_paulis

    def copy(self) -> PauliSet:
        """An independent copy, signs included."""
        copied = PauliSet(self.z.copy(), self.x.copy())
        copied.signs = self.signs.copy()
        return copied

    def __getitem__(self, key: Index | tuple[Index, Index]) -> PauliSet:
        """``pauli_set[qubits, strings]``: a new PauliSet of those rows and columns, as in M.

        Integer indices keep their dimension, so ``pauli_set[q]`` is a 1-qubit set and
        ``pauli_set[:, k]`` a 1-string set. Signs follow the strings.
        """
        rows, columns = key if isinstance(key, tuple) else (key, slice(None))
        rows, columns = (np.array([i]) if isinstance(i, (int, np.integer)) else i for i in (rows, columns))
        sub = PauliSet(self.z[rows][:, columns], self.x[rows][:, columns])
        sub.signs = self.signs[columns].copy()
        return sub

    def support_mask(self, k: int) -> BoolArray:
        """Boolean mask of the qubits on which string $k$ acts non-trivially."""
        return self.z[:, k] | self.x[:, k]

    def support_sizes(self) -> IntArray:
        """Number of qubits each string acts on non-trivially, shape ``(m,)``."""
        return np.count_nonzero(self.z | self.x, axis=0)

    def sorted_columns(self) -> tuple[PauliSet, IntArray]:
        """SortColumns(M): a new PauliSet with the strings in ascending support size, and the permutation used.

        Column ``k`` of the result is column ``order[k]`` of ``self``.
        """
        order = np.argsort(self.support_sizes(), kind="stable")
        return self[:, order], order

    # Clifford conjugation P -> G P G^dagger, in place

    def h(self, q: int) -> None:
        r"""Conjugate by $H$ on qubit ``q``: $P_k \mapsto H P_k H^\dagger$."""
        self.signs ^= self.z[q] & self.x[q]  # H Y H = -Y
        self.z[q], self.x[q] = self.x[q].copy(), self.z[q].copy()

    def s(self, q: int) -> None:
        r"""Conjugate by $S$ on qubit ``q``: $P_k \mapsto S P_k S^\dagger$."""
        self.signs ^= self.z[q] & self.x[q]  # S Y S^dagger = -X
        self.z[q] ^= self.x[q]

    def sx(self, q: int) -> None:
        r"""Conjugate by $\sqrt{X}$ on qubit ``q``: $P_k \mapsto \sqrt{X} P_k \sqrt{X}^\dagger$."""
        self.signs ^= self.z[q] & ~self.x[q]  # sqrt(X) Z sqrt(X)^dagger = -Y
        self.x[q] ^= self.z[q]

    def cx(self, control: int, target: int) -> None:
        r"""Conjugate by $\mathrm{CNOT}$ on ``control``, ``target``.

        $P_k \mapsto \mathrm{CNOT} \, P_k \, \mathrm{CNOT}^\dagger$.

        Raises
        ------
        ValueError
            If ``control == target``.

        """
        if control == target:
            raise ValueError("cx needs two distinct qubits")
        zc, xc, zt, xt = self.z[control], self.x[control], self.z[target], self.x[target]
        self.signs ^= xc & zt & ~(xt ^ zc)
        self.z[control] ^= self.z[target]
        self.x[target] ^= self.x[control]
