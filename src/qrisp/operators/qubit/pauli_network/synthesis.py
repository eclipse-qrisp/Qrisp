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

"""Algorithm 1 (CountSynthesis) of arXiv:2404.03280: a Pauli network with few CNOTs."""

from collections.abc import Iterable
from itertools import combinations, product

import numpy as np

from qrisp.operators.qubit.pauli_network.common import (
    CHUNKS,
    Chunk,
    PauliNetwork,
    conjugate_by_chunk,
    place,
    pop_rotation,
    score,
)
from qrisp.operators.qubit.pauli_network.pauli_set import PauliSet


def _best_chunk_and_qubits(pauli_set: PauliSet, qubit_pairs: Iterable[tuple[int, int]]) -> tuple[Chunk, int, int]:
    """The (chunk, i, j), where i < j,  maximizing :func:`score` (line 8)."""
    chunk_qubit_pair_triples = [(chunk, i, j) for chunk, (i, j) in product(CHUNKS, qubit_pairs)]
    return chunk_qubit_pair_triples[
        np.argmax([score(pauli_set, chunk, i, j) for (chunk, i, j) in chunk_qubit_pair_triples])
    ]


def count_synthesis(pauli_set: PauliSet) -> PauliNetwork:
    r"""Algorithm 1 (CountSynthesis) of arXiv:2404.03280.

    Parameters
    ----------
    pauli_set : PauliSet
        The strings to implement. It is not modified.

    Returns
    -------
    PauliNetwork
        A $\{\mathrm{CNOT}, H, \sqrt{X}, S\}$-Pauli network for the strings, with one Rotation per non-identity
        string. Identity strings are listed in ``identity_indices`` instead: they never reach
        support size 1, so they are dropped before the synthesis starts.

    """
    gates, rotations = [], []
    indices = np.arange(pauli_set.num_paulis)  # input index of each current column
    # Keep track of identity indices.
    is_identity = pauli_set.support_sizes() == 0
    identity_indices = indices[is_identity].tolist()
    pauli_set, indices = pauli_set[:, ~is_identity], indices[~is_identity]
    while pauli_set:
        # Sort pauli strings by support size
        pauli_set, order = pauli_set.sorted_columns()
        indices = indices[order]
        # Pop the pauli strings with support size 1 as rotations
        is_single = pauli_set.support_sizes() == 1
        collected_rotations = (
            pop_rotation(pauli_set, k, indices[k], len(gates)) for k in np.flatnonzero(is_single).tolist()
        )
        rotations.extend(collected_rotations)
        pauli_set, indices = pauli_set[:, ~is_single], indices[~is_single]
        if not pauli_set:
            break
        # Conjugate by the best chunk on a pair of qubits that the first
        # (smallest) string non-trivially acts on.
        head_active_qubits = np.flatnonzero(pauli_set.support_mask(0)).tolist()
        active_qubit_pairs = combinations(head_active_qubits, 2)
        best_chunk, i, j = _best_chunk_and_qubits(pauli_set, active_qubit_pairs)
        pauli_set = conjugate_by_chunk(pauli_set, best_chunk, i, j)
        gates.extend(place(best_chunk, i, j))
    return PauliNetwork(gates, rotations, identity_indices)
