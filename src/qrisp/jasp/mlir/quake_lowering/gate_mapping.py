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

"""
Gate-name mapping: Jasp ``gate_type`` attribute → Quake gate factory arguments.

Each entry in :data:`GATE_MAP` describes how a Jasp quantum gate should be
lowered to a Quake gate op:

``(num_controls, quake_gate_name)``

where *num_controls* is the number of qubit operands that are **control**
qubits (the remaining qubit operands are targets).

Unsupported gates (``parity`` and exotic gates not in Quake's native set) are
intentionally absent from the map; callers should check for ``None`` returns
and handle them accordingly (e.g. raise an error or skip).

Gate-name conventions used by Qrisp / Jasp
-------------------------------------------
The ``gate_type`` string stored in ``jasp.quantum_gate`` comes from
``qrisp.Operation.name``.  Standard names include: ``h``, ``x``, ``y``, ``z``,
``cx``, ``cy``, ``cz``, ``s``, ``s_dg``, ``t``, ``t_dg``, ``rx``, ``ry``,
``rz``, ``p``, ``u3``, ``swap``, ``mcx``, ``mcp``, etc.
"""


from dataclasses import dataclass, field
from typing import Callable, Optional


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
    #:   (controls: list[SSAValue], params: list[SSAValue],
    #:    targets: list[SSAValue]) -> list[Operation]
    #: When provided, this is called *instead* of ``make_gate_op``.
    emit: Optional[Callable] = field(default=None, repr=False, compare=False)


# ===================================================================
# Decomposition emitters
# ===================================================================


def _emit_sx(controls, params, targets):
    """sx(q) = H(q) · S(q) · H(q)"""
    from qrisp.jasp.mlir.quake_lowering.quake_dialect import make_gate_op
    t = targets[0]
    return [
        make_gate_op("h", [], [], [t]),
        make_gate_op("s", controls, [], [t]),
        make_gate_op("h", [], [], [t]),
    ]


def _emit_sx_dg(controls, params, targets):
    """sx†(q) = H(q) · S†(q) · H(q)"""
    from qrisp.jasp.mlir.quake_lowering.quake_dialect import make_gate_op
    t = targets[0]
    return [
        make_gate_op("h", [], [], [t]),
        make_gate_op("s_dg", controls, [], [t]),
        make_gate_op("h", [], [], [t]),
    ]


# ===================================================================
# Gate map
# ===================================================================

# Jasp gate name → GateInfo
GATE_MAP: dict[str, GateInfo] = {
    # ---- 1-qubit, 0-param ------------------------------------------------
    "h":     GateInfo(num_controls=0, quake_gate="h"),
    "x":     GateInfo(num_controls=0, quake_gate="x"),
    "y":     GateInfo(num_controls=0, quake_gate="y"),
    "z":     GateInfo(num_controls=0, quake_gate="z"),
    "s":     GateInfo(num_controls=0, quake_gate="s"),
    "t":     GateInfo(num_controls=0, quake_gate="t"),
    "s_dg":  GateInfo(num_controls=0, quake_gate="s<adj>"),
    "t_dg":  GateInfo(num_controls=0, quake_gate="t<adj>"),
    # ---- 1-qubit, 0-param, DECOMPOSED -----------------------------------
    "sx":     GateInfo(num_controls=0, quake_gate="", emit=_emit_sx),
    "sx_dg":  GateInfo(num_controls=0, quake_gate="", emit=_emit_sx_dg),
    # ---- 1-qubit, 1-param ------------------------------------------------
    "rx":    GateInfo(num_controls=0, quake_gate="rx",  num_params=1),
    "ry":    GateInfo(num_controls=0, quake_gate="ry",  num_params=1),
    "rz":    GateInfo(num_controls=0, quake_gate="rz",  num_params=1),
    "p":     GateInfo(num_controls=0, quake_gate="r1",  num_params=1),
    "r1":    GateInfo(num_controls=0, quake_gate="r1",  num_params=1),
    # ---- 1-qubit, 3-param ------------------------------------------------
    "u3":    GateInfo(num_controls=0, quake_gate="u3",  num_params=3),
    # ---- 2-qubit, 0-param ------------------------------------------------
    "swap":  GateInfo(num_controls=0, quake_gate="swap"),
    # ---- 2-qubit, 1-control (= CX / CNOT family) -------------------------
    "cx":    GateInfo(num_controls=1, quake_gate="x"),
    "cy":    GateInfo(num_controls=1, quake_gate="y"),
    "cz":    GateInfo(num_controls=1, quake_gate="z"),
    "ch":    GateInfo(num_controls=1, quake_gate="h"),
    "cs":    GateInfo(num_controls=1, quake_gate="s"),
    "ct":    GateInfo(num_controls=1, quake_gate="t"),
    # ---- multi-controlled-X (Toffoli-family) -----------------------------
    # For mcx with N controls, all qubits except the last are controls.
    # This is handled specially in pass1; we include a sentinel entry here
    # so presence in the map indicates "supported".
    "2cx":   GateInfo(num_controls=2, quake_gate="x"),
    "pt2cx": GateInfo(num_controls=2, quake_gate="x"),
    "ccx":   GateInfo(num_controls=2, quake_gate="x"),
    "mcx":   GateInfo(num_controls=-1, quake_gate="x"),   # -1 = all-but-last
    "mcp":   GateInfo(num_controls=-1, quake_gate="r1",  num_params=1),
    # ---- 2-qubit, 1-param, controlled ------------------------------------
    "crx":   GateInfo(num_controls=1, quake_gate="rx",   num_params=1),
    "cry":   GateInfo(num_controls=1, quake_gate="ry",   num_params=1),
    "crz":   GateInfo(num_controls=1, quake_gate="rz",   num_params=1),
    "cp":    GateInfo(num_controls=1, quake_gate="r1",   num_params=1),
    # ---- explicitly unsupported (will be flagged) -------------------------
    # "rxx", "rzz", "xxyy", "gphase": not in Quake's native gate set.
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
        is not in the map (unsupported / exotic).
    """
    return GATE_MAP.get(gate_name)
