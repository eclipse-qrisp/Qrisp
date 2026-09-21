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
Round-trip tests for the custom assembly format of the JASP dialect.

The textual syntax of the dialect exists in two places: ``assembly_format`` in
``qrisp.jasp.mlir.xdsl_dialect`` (the printer Qrisp actually uses) and
``assemblyFormat`` in ``dialect_definition/JaspOps.td`` (what an MLIR-based
consumer builds its parser from).  Nothing in Qrisp's own pipeline exercises the
TableGen side -- MLIR emission goes through xDSL, and the TableGen files are only
consumed by ``mlir-tblgen -gen-python-op-bindings``, which does not embed the
assembly format.  The two therefore drifted apart once already, see
https://github.com/eclipse-qrisp/Qrisp/issues/783.

These tests pin down the part that can be checked without an LLVM installation:

* what Qrisp prints can be parsed back, and printing the result is a fixed point
  (this fails outright for any op that has a printer but no matching parser),
* ``jasp_dialect_syntax.mlir`` -- a hand-written module in the syntax both
  definitions are supposed to describe -- parses, verifies and round-trips, and
  covers every op of the dialect.

Keeping that file in sync with the dialect is what keeps it usable as the
reference an MLIR-based consumer can check its own parser against.
"""

import pathlib

import pytest

from qrisp import QuantumFloat, QuantumVariable, control, cx, h, jrange, make_jaspr, measure, rz, x
from qrisp.jasp.mlir.jaxpr_lowering import MLIR_str_to_xdsl
from qrisp.jasp.mlir.xdsl_dialect import JaspDialect

SYNTAX_REFERENCE = pathlib.Path(__file__).parent / "jasp_dialect_syntax.mlir"
TABLEGEN_OPS = (
    pathlib.Path(__file__).parents[2] / "src" / "qrisp" / "jasp" / "mlir" / "dialect_definition" / "JaspOps.td"
)


def assert_roundtrips(mlir_str):
    """Parse *mlir_str*, print it again and assert that printing is a fixed point.

    Returns the re-printed module string.
    """
    _, module = MLIR_str_to_xdsl(mlir_str)
    printed = str(module)

    _, reparsed = MLIR_str_to_xdsl(printed)
    assert str(reparsed) == printed, "printing a parsed module is not a fixed point"

    return printed


def test_syntax_reference_roundtrips():
    """The hand-written reference module parses, verifies and round-trips."""
    reference = SYNTAX_REFERENCE.read_text()

    _, module = MLIR_str_to_xdsl(reference)
    module.verify()

    assert_roundtrips(reference)


def test_syntax_reference_covers_every_op():
    """Every op of the dialect appears in the reference module.

    A new op has to be added to ``jasp_dialect_syntax.mlir`` as well, otherwise
    its custom syntax is never parsed by anything.
    """
    reference = SYNTAX_REFERENCE.read_text()

    missing = [op.name for op in JaspDialect.operations if op.name not in reference]
    assert not missing, f"ops missing from {SYNTAX_REFERENCE.name}: {missing}"


def test_tablegen_declares_the_same_assembly_format():
    """Every xDSL assembly format appears verbatim in the TableGen definition.

    This is a containment check, deliberately not a TableGen parser: the format
    string Qrisp prints with has to occur literally in ``JaspOps.td``, modulo
    whitespace.  That is enough to catch the two sides describing a different
    syntax -- which is what happened in issue #783, where the TableGen kept the
    operand types of ``create_qubits`` in the wrong order -- without making any
    assumption about TableGen grammar beyond string literals being written on
    one line.

    Type constraints are not compared.  They cannot be, exactly: ``_TensorInt``
    and the unconstrained ``gate_operands`` are looser on the xDSL side than
    their TableGen counterparts on purpose (see ``xdsl_dialect``), so a
    comparison would need a table of exceptions per op, which is where the next
    drift would hide.
    """
    tablegen = " ".join(TABLEGEN_OPS.read_text().split())

    undeclared = []
    for op in JaspDialect.operations:
        assembly_format = getattr(op, "assembly_format", None)
        assert assembly_format is not None, (
            f"{op.name} has no declarative assembly_format, so xDSL can print it but not parse it back"
        )

        if " ".join(assembly_format.split()) not in tablegen:
            undeclared.append((op.name, assembly_format))

    assert not undeclared, (
        f"assembly format of {[name for name, _ in undeclared]} not found in {TABLEGEN_OPS.name}. "
        f"xDSL prints:\n" + "\n".join(f"  {name}: {fmt}" for name, fmt in undeclared)
    )


@pytest.mark.parametrize("lower_stablehlo", [False, True])
def test_emitted_mlir_roundtrips(lower_stablehlo):
    """What ``to_mlir`` prints can be parsed back.

    Covers create_qubits, get_qubit, slice, get_size, quantum_gate (both with and
    without float parameters), measure, delete_qubits and the classical control
    flow around them.
    """

    def main(i):
        qv = QuantumVariable(i)
        qf = QuantumFloat(2)

        h(qv[0])
        cx(qv[0], qv[1])
        rz(0.5, qv[0])

        meas_res = measure(qv[0])

        for j in jrange(qf.size):
            rz(1 / (j + 1), qf[j])
            cx(qv[1], qf[j])

        with control(meas_res == 0):
            x(qf[0])

        return measure(qf)

    jaspr = make_jaspr(main)(3)
    printed = str(jaspr.to_mlir(lower_stablehlo=lower_stablehlo))

    assert "jasp.quantum_gate" in printed, "test program should emit gates"
    assert_roundtrips(printed)
