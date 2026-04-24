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

from cudaq.mlir.ir import Context, Module
from cudaq.mlir.dialects import quake, cc


def validate_quake_mlir(mlir_str: str) -> Module:
    """Validate a Quake MLIR string using CUDA-Q's MLIR parser and verifier.

    Parses the provided MLIR string within a fresh MLIR context that has
    the ``quake`` and ``cc`` dialects registered. Parsing implicitly runs
    MLIR's built-in verification, which checks syntax, type correctness,
    and dialect operation invariants.

    Parameters
    ----------
    mlir_str : str
        A string containing a complete MLIR module that uses the
        ``quake`` and/or ``cc`` dialects.

    Returns
    -------
    Module
        The parsed and verified ``cudaq.mlir.ir.Module`` object, ready
        for further processing (e.g. lowering, optimization, execution).

    Raises
    ------
    ValueError
        If the MLIR string cannot be parsed or fails verification,
        with the original parser/verifier diagnostic attached.

    Examples
    --------

    >>> from qrisp import QuantumVariable, h, cx, measure
    >>> from qrisp.jasp import make_jaspr
    >>> from qrisp.jasp.mlir.quake_lowering import jaspr_to_quake
    >>>
    >>> def bell():
    ...     qv = QuantumVariable(2)
    ...     h(qv[0])
    ...     cx(qv[0], qv[1])
    ...     return measure(qv)
    ...
    >>> mlir_str = str(jaspr_to_quake(make_jaspr(bell)()))
    >>> module = validate_quake_mlir(mlir_str)
    """
    with Context() as ctx:
        quake.register_dialect(context=ctx)
        cc.register_dialect(context=ctx)

        try:
            module = Module.parse(mlir_str, ctx)
        except Exception as exc:
            raise ValueError(
                f"Invalid Quake MLIR — parsing/verification failed:\n{exc}"
            ) from exc

    return module
