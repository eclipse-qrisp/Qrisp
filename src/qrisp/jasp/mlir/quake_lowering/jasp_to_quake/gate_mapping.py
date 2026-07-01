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

"""
Gate-name mapping: Jasp ``gate_type`` attribute → Quake gate factory arguments.

Each entry in :data:`GATE_MAP` describes how a Jasp quantum gate should be
lowered to a Quake gate op:

``(num_controls, quake_gate_name)``

where *num_controls* is the number of qubit operands that are **control**
qubits (the remaining qubit operands are targets).

Gate-name conventions used by Qrisp / Jasp
-------------------------------------------
The ``gate_type`` string stored in ``jasp.quantum_gate`` comes from
``qrisp.Operation.name``.  Standard names include: ``h``, ``x``, ``y``, ``z``,
``cx``, ``cy``, ``cz``, ``s``, ``s_dg``, ``t``, ``t_dg``, ``rx``, ``ry``,
``rz``, ``p``, ``u3``,``sx``, ``sx_dg``, ``gphase``, ``cgphase``.  

Composite gates like ``xxyy`` and ``xxzz`` are decomposed before lowering to MLIR at the Jaspr level
via ``decompose_composite_gates`` and are not present in the map.
"""

from collections.abc import Callable, Sequence
from typing import Optional

from dataclasses import dataclass, field

from xdsl.dialects import arith
from xdsl.dialects.builtin import FloatAttr, f64
from xdsl.ir import (
    Operation,
    SSAValue,
)
from qrisp.jasp.mlir.quake_lowering.dialects.quake_dialect import make_gate_op


@dataclass(frozen=True)
class GateInfo:
    """Lowering information for a single Jasp gate.

    For simple 1:1 mappings, set ``quake_gate`` and leave ``emit`` as None.
    For multi-op decompositions, provide an ``emit`` callable.
    """

    #: Number of leading qubit operands that are *control* qubits.
    #: -1 = all-but-last (mcx family).
    num_controls: int

    #: Quake gate name (without ``quake.`` prefix).  Ignored when ``emit``
    #: is provided.
    quake_gate: str = ""

    #: Number of floating-point parameters expected.
    num_params: int = 0

    #: Optional custom emitter.  Signature:
    #:   (controls: Sequence[SSAValue], params: Sequence[SSAValue],
    #:    targets: Sequence[SSAValue]) -> list[Operation]
    #: When provided, this is called *instead* of ``make_gate_op``.
    emit: Optional[Callable] = field(default=None, repr=False, compare=False)


# ===================================================================
# Decomposition emitters
# ===================================================================


def _emit_sx(controls: Sequence[SSAValue], params: Sequence[SSAValue], targets: Sequence[SSAValue]) -> list[Operation]:
    """sx(q) = H(q) · S(q) · H(q)"""

    t = targets[0]
    return [
        make_gate_op("h", [], [], [t]),
        make_gate_op("s", controls, [], [t]),
        make_gate_op("h", [], [], [t]),
    ]


def _emit_sx_dg(
    controls: Sequence[SSAValue], params: Sequence[SSAValue], targets: Sequence[SSAValue]
) -> list[Operation]:
    """sx†(q) = H(q) · S†(q) · H(q)"""

    t = targets[0]
    return [
        make_gate_op("h", [], [], [t]),
        make_gate_op("s_dg", controls, [], [t]),
        make_gate_op("h", [], [], [t]),
    ]


def _emit_cgphase(
    controls: Sequence[SSAValue], params: Sequence[SSAValue], targets: Sequence[SSAValue]
) -> list[Operation]:
    """cgphase(ϕ) between targets q0 and q1 = p(ϕ, q0)"""

    t0, t1 = targets
    phi = params[0]
    return [
        make_gate_op("p", [], [phi], [t0]),
    ]


def _emit_gphase(
    controls: Sequence[SSAValue], params: Sequence[SSAValue], targets: Sequence[SSAValue]
) -> list[Operation]:
    """gphase(ϕ) on target q0 = p(ϕ, q0)"""

    t0 = targets[0]
    phi = params[0]
    two = arith.ConstantOp(FloatAttr(2.0, f64))
    two_phi = arith.MulfOp(two.result, phi)
    neg_two_phi = arith.NegfOp(two_phi.result)
    return [
        two,
        two_phi,
        neg_two_phi,
        make_gate_op("rz", [], [neg_two_phi], [t0]),
        make_gate_op("p", [], [two_phi], [t0]),
    ]


# ===================================================================
# Gate map
# ===================================================================

GATE_MAP: dict[str, GateInfo] = {
    # ---- 1-qubit, 0-param ------------------------------------------------
    "h": GateInfo(num_controls=0, quake_gate="h"),
    "x": GateInfo(num_controls=0, quake_gate="x"),
    "y": GateInfo(num_controls=0, quake_gate="y"),
    "z": GateInfo(num_controls=0, quake_gate="z"),
    "s": GateInfo(num_controls=0, quake_gate="s"),
    "t": GateInfo(num_controls=0, quake_gate="t"),
    "s_dg": GateInfo(num_controls=0, quake_gate="s<adj>"),
    "t_dg": GateInfo(num_controls=0, quake_gate="t<adj>"),
    # ---- 1-qubit, 0-param, DECOMPOSED ------------------------------------
    "sx": GateInfo(num_controls=0, quake_gate="", emit=_emit_sx),
    "sx_dg": GateInfo(num_controls=0, quake_gate="", emit=_emit_sx_dg),
    # ---- 1-qubit, 1-param ------------------------------------------------
    "rx": GateInfo(num_controls=0, quake_gate="rx", num_params=1),
    "ry": GateInfo(num_controls=0, quake_gate="ry", num_params=1),
    "rz": GateInfo(num_controls=0, quake_gate="rz", num_params=1),
    "p": GateInfo(num_controls=0, quake_gate="r1", num_params=1),
    "r1": GateInfo(num_controls=0, quake_gate="r1", num_params=1),
    "gphase": GateInfo(num_controls=0, quake_gate="", emit=_emit_gphase, num_params=1),
    # ---- 1-qubit, 3-param ------------------------------------------------
    "u3": GateInfo(num_controls=0, quake_gate="u3", num_params=3),
    # ---- 2-qubit, 1-control (= CX / CNOT family) -------------------------
    "cx": GateInfo(num_controls=1, quake_gate="x"),
    "cy": GateInfo(num_controls=1, quake_gate="y"),
    "cz": GateInfo(num_controls=1, quake_gate="z"),
    # ---- 2-qubit, 1-param, DECOMPOSED ------------------------------------
    "cgphase": GateInfo(num_controls=0, quake_gate="", emit=_emit_cgphase, num_params=1),
}


def get_gate_info(gate_name: str) -> GateInfo | None:
    """Look up lowering info for a Jasp gate by name.

    Parameters
    ----------
    gate_name:
        The ``gate_type`` attribute value from a ``jasp.quantum_gate`` op.

    Returns
    -------
    GateInfo | None
        The :class:`GateInfo` entry if the gate is supported, or *None* if it
        is not in the map.
    """
    return GATE_MAP.get(gate_name)
