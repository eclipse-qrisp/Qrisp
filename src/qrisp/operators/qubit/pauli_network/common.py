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

r"""Greedy Pauli network synthesis.

Follows T. Goubault de Brugière and S. Martiel, "Faster and shorter synthesis of
Hamiltonian simulation circuits" (arXiv:2404.03280).

A Pauli network for strings $P_1, \dots, P_m$ is a Clifford circuit $C$ such that for
every $k$ some prefix $U_k$ of $C$ satisfies: $U_k P_k U_k^\dagger$ has support size 1
(Definition 1). Here $U_k$ is the unitary of the first $p_k$ gates, and a gate $G$ acts on
a string as $P \mapsto G P G^\dagger$ (Algorithm 1, line 9); Definition 1 writes the same
condition as $U_k^\dagger P_k U_k$.
"""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from itertools import groupby
from operator import attrgetter
from typing import Any, TypeAlias

import numpy as np

from qrisp import CXGate, HGate, Operation, QuantumVariable, SGate, SXGate, append_operation, gphase, rx, ry, rz
from qrisp.operators.qubit.pauli_network.pauli_set import PauliSet

CLIFFORD_GATES: dict[str, Operation] = {
    "h": HGate(),
    "s": SGate(),
    "sx": SXGate(),
    "cx": CXGate(),
}  # keyed by Gate.name
ROTATION_GATES: dict[str, Callable[..., None]] = {"X": rx, "Y": ry, "Z": rz}  # keyed by Rotation.axis
# A rotation angle: a float, a traced jax value or a sympy expression (symbolic time). sympy's operators
# are untyped, so a precise union would reject valid arithmetic.
Angle: TypeAlias = Any


@dataclass(frozen=True)
class Gate:
    """A Clifford gate of the network. ``name`` is a :class:`PauliSet` gate method."""

    name: str
    qubits: tuple[int, ...]

    def apply(self, qv: QuantumVariable, inverse: bool = False) -> None:
        """Apply this Clifford to qubits ``qubits`` of the QuantumVariable ``qv``."""
        gate = CLIFFORD_GATES[self.name]
        if inverse:
            gate = gate.inverse()
        append_operation(gate, [qv[q] for q in self.qubits])


class CX(Gate):
    """A CNOT gate, with named ``control`` and ``target`` qubits."""

    def __init__(self, control: int, target: int) -> None:
        """Create the CNOT with control ``control`` and target ``target``."""
        super().__init__("cx", (control, target))

    @property
    def control(self) -> int:
        """The control qubit."""
        return self.qubits[0]

    @property
    def target(self) -> int:
        """The target qubit."""
        return self.qubits[1]


Chunk = tuple[Gate, ...]  # 1 to 3 gates; the last is always a CX


def _build_chunks() -> tuple[Chunk, ...]:
    r"""The 18 chunks $\mathrm{CNOT} \cdot (U_1 \otimes U_2)$ of Sec. 3.

    $U_1 \in \{I, H, \sqrt{X}\}$ acts on the control and $U_2 \in \{I, H, S\}$ on the
    target. Qubits are local labels 0 and 1, both CNOT directions included.
    """
    chunks = []
    for control, target in ((0, 1), (1, 0)):
        for u1 in (None, "h", "sx"):
            for u2 in (None, "h", "s"):
                chunk = []
                if u1 is not None:
                    chunk.append(Gate(u1, (control,)))
                if u2 is not None:
                    chunk.append(Gate(u2, (target,)))
                chunk.append(CX(control, target))
                chunks.append(tuple(chunk))
    return tuple(chunks)


CHUNKS = _build_chunks()


@dataclass(frozen=True)
class Rotation:
    r"""A rotation $e^{-i\theta\,(\pm\sigma)}$ on ``qubit``, placed after the first ``position`` gates.

    After those gates, input string ``term_index`` has become ``-axis`` on ``qubit`` if ``negated``,
    else ``+axis``, where ``axis`` is ``"X"``, ``"Y"`` or ``"Z"``.
    """

    term_index: int
    qubit: int
    axis: str
    negated: bool
    position: int

    def apply(self, qv: QuantumVariable, theta: Angle) -> None:
        r"""Apply $e^{-i\theta\,(\pm\sigma)}$ to qubit ``qubit`` of the QuantumVariable ``qv``.

        Qrisp's $R_\sigma(\phi)$ is $e^{-i\phi\sigma/2}$, hence the angle $\pm 2\theta$.
        """
        ROTATION_GATES[self.axis](-2 * theta if self.negated else 2 * theta, qv[self.qubit])


@dataclass
class PauliNetwork:
    """The Clifford gates and the rotations between them, ordered by ``Rotation.position``.

    ``identity_indices`` are the input strings that are the identity: they need no gates,
    and their exponential is a global phase for the caller to apply.
    """

    gates: list[Gate]
    rotations: list[Rotation]
    identity_indices: list[int] = field(default_factory=list)

    def __iter__(self) -> Iterator[Gate | Rotation]:
        """Yield the gates and rotations as they appear in the circuit.

        A Rotation with ``position`` p comes after gate p - 1 and before gate p.
        """
        rotations_at = {p: list(group) for p, group in groupby(self.rotations, key=attrgetter("position"))}
        for position, gate in enumerate(self.gates):
            yield from rotations_at.get(position, ())
            yield gate
        yield from rotations_at.get(len(self.gates), ())

    def apply(self, qv: QuantumVariable, thetas: Sequence[Angle] | np.ndarray, mirror: bool = False) -> None:
        r"""Apply $e^{-i\theta_k P_k}$ for every input string $P_k$, in rotation order, to ``qv``.

        The gates are applied with the rotations between them, then undone in reverse order.
        Identity strings become a global phase.

        Parameters
        ----------
        qv : QuantumVariable
            The QuantumVariable to act on.
        thetas : array_like
            ``thetas[k]`` is the angle $\theta_k$ of input string $k$.
        mirror : bool, optional
            If True, every rotation is applied twice at half angle: once on the way in and once
            on the way back, giving the symmetric (second order) product. The default is False.

        Notes
        -----
        With gates $G_1, G_2, G_3$ and rotations $R_A, R_B, R_C$ between them, the gates leave
        the qubits in the frame $W_3 = G_3 G_2 G_1$, and the uncompute
        $W_3^\dagger = G_1^\dagger G_2^\dagger G_3^\dagger$ brings them back:

        .. math::

            U_1 = W_3^\dagger R_C G_3 R_B G_2 R_A G_1.

        With ``mirror=True``, the uncompute also carries the rotations, at half angle:

        .. math::

            S_2 = G_1^\dagger R_A G_2^\dagger R_B G_3^\dagger R_C \cdot R_C G_3 R_B G_2 R_A G_1.

        Each $R$ sits in the same frame as on the way in, so $S_2$ implements the second order
        product $e^{-i\theta_A P_A/2} e^{-i\theta_B P_B/2} e^{-i\theta_C P_C/2} \cdot
        e^{-i\theta_C P_C/2} e^{-i\theta_B P_B/2} e^{-i\theta_A P_A/2}$. It is the usual
        second order step, the first order step at half angle followed by its inverse at
        negative half angle, with the identity $W_3 W_3^\dagger$ in the middle removed. So it
        costs the same Clifford gates as $U_1$.

        """
        scale = 0.5 if mirror else 1.0
        operations = list(self)
        for operation in operations:
            if isinstance(operation, Rotation):
                operation.apply(qv, scale * thetas[operation.term_index])
            else:
                operation.apply(qv)
        for operation in reversed(operations):
            if isinstance(operation, Rotation):
                if mirror:
                    operation.apply(qv, scale * thetas[operation.term_index])
            else:
                operation.apply(qv, inverse=True)
        if self.identity_indices:
            # e^{-i theta I} = e^{-i theta}, and Qrisp's gphase(phi) multiplies by e^{i phi}.
            gphase(-sum(thetas[k] for k in self.identity_indices), qv[0])


def place(chunk: Chunk, i: int, j: int) -> list[Gate]:
    """The gates of ``chunk`` with its local qubits 0, 1 mapped to ``i``, ``j``."""
    qubit_map = (i, j)
    placed = []
    for gate in chunk:
        qubits = tuple(qubit_map[q] for q in gate.qubits)
        placed.append(CX(*qubits) if isinstance(gate, CX) else Gate(gate.name, qubits))
    return placed


def conjugate_by_chunk(pauli_set: PauliSet, chunk: Chunk, i: int, j: int) -> PauliSet:
    """Return a copy of ``pauli_set`` conjugated by ``chunk`` placed on ``i``, ``j``."""
    new_pauli_set = pauli_set.copy()
    for gate in place(chunk, i, j):
        getattr(new_pauli_set, gate.name)(*gate.qubits)
    return new_pauli_set


def leading_identities(pauli_set: PauliSet, q: int) -> int:
    """CLI(M, q): number of leading strings that are the identity on qubit ``q``."""
    acts_on_q = pauli_set[q].support_sizes() > 0
    return int(np.argmax(acts_on_q)) if acts_on_q.any() else len(acts_on_q)


def score(pauli_set: PauliSet, chunk: Chunk, i: int, j: int) -> int:
    r"""Score(M, D, i, j) $= \max_{q \in \{i, j\}} \mathrm{CLI}(D M D^\dagger, q) - \mathrm{CLI}(M, q)$."""
    conjugated_pauli_set = conjugate_by_chunk(pauli_set, chunk, i, j)

    def score_for_q(q: int) -> int:
        return leading_identities(conjugated_pauli_set, q) - leading_identities(pauli_set, q)

    return max(score_for_q(i), score_for_q(j))


AXES: dict[tuple[bool, bool], str] = {(True, False): "Z", (False, True): "X", (True, True): "Y"}  # keyed by (z, x)


def pop_rotation(pauli_set: PauliSet, k: int, term_index: int, position: int = 0) -> Rotation:
    """The Rotation for column ``k``, which must have support size 1, as input string ``term_index``."""
    (qubit,) = np.flatnonzero(pauli_set.support_mask(k))
    axis = AXES[bool(pauli_set.z[qubit, k]), bool(pauli_set.x[qubit, k])]
    return Rotation(int(term_index), int(qubit), axis, bool(pauli_set.signs[k]), position)
