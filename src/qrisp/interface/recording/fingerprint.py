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

"""Circuit fingerprints that are stable across processes."""

import hashlib
import json
import numbers
from typing import Any

from sympy import Expr, srepr

from qrisp.circuit.quantum_circuit import QuantumCircuit


def circuit_fingerprint(qc: QuantumCircuit) -> str:
    """Return a SHA-256 fingerprint of ``qc`` that is stable across processes.

    Note: Qubit and clbit names are ignored; only their positions in the circuit matter.

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to fingerprint.

    Returns
    -------
    str
        ``"sha256:"`` followed by the hex digest.

    """
    canonical = json.dumps(_canonical_form(qc), separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _canonical_form(qc: QuantumCircuit) -> list:
    """Return a JSON-serializable description of ``qc`` that ignores bit names."""
    qubit_index = {qb: i for i, qb in enumerate(qc.qubits)}
    clbit_index = {cb: i for i, cb in enumerate(qc.clbits)}
    return [
        len(qc.qubits),
        len(qc.clbits),
        [
            [
                instr.op.name,
                [_canonical_param(p) for p in instr.op.params],
                [qubit_index[qb] for qb in instr.qubits],
                [clbit_index[cb] for cb in instr.clbits],
                None if instr.op.definition is None else _canonical_form(instr.op.definition),
            ]
            for instr in qc.data
        ],
    ]


def _canonical_param(param: Any) -> str:
    """Return a process-independent string for a gate parameter.

    Numbers go through ``float``/``complex`` so that ``1``, ``1.0`` and
    ``np.float64(1.0)`` give the same string.
    """
    if isinstance(param, numbers.Real):
        return repr(float(param))
    if isinstance(param, numbers.Complex):
        return repr(complex(param))
    if isinstance(param, Expr):
        return srepr(param)
    raise TypeError(f"Cannot fingerprint gate parameter of type {type(param).__name__}")
