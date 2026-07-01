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
Jasp → Quake (memory-semantics) lowering backend.

This package adds a Quake emission backend to the Qrisp MLIR pipeline.
The main entry point is :func:`jaspr_to_quake`.

Sub-modules
-----------
quake_dialect
    xDSL type and op definitions for the CUDA-Q Quake dialect.
cc_dialect
    xDSL type and op definitions for the CUDA-Q CC (classical control) dialect.
gate_mapping
    Mapping from Jasp gate names to Quake gate descriptors.
pass1_jasp_to_quake
    PASS 1: QuantumState elimination + Jasp→Quake op rewriting.
pass2_scf_to_cc
    PASS 2: SCF → CC dialect lowering.
pass3_tensor_unwrap
    PASS 3: Tensor unwrapping and function-signature rewrite.
jaspr_to_quake_mlir
    Pipeline entry-point (:func:`jaspr_to_quake_mlir`).
"""

from qrisp.jasp.mlir.quake_lowering.jaspr_to_quake import jaspr_to_quake_mlir
from qrisp.jasp.mlir.quake_lowering.validation_tools import validate_quake_mlir

try:
    from qrisp.jasp.cudaq_interface import (
        run_quake_mlir,
        qrisp_cudaq_kernel,
        cudaq_kernel_from_mlir,
        FixedShapeNDArray,
    )
except ImportError:
    pass
from qrisp.jasp.mlir.quake_lowering.dialects.quake_dialect import (
    QuakeDialect,
    QuakeMeasureType,
    QuakeRefType,
    QuakeVeqType,
)
from qrisp.jasp.mlir.quake_lowering.dialects.cc_dialect import CcDialect, CcStdVecType

__all__ = [
    "jaspr_to_quake_mlir",
    "validate_quake_mlir",
    "run_quake_mlir",
    "qrisp_cudaq_kernel",
    "cudaq_kernel_from_mlir",
    "FixedShapeNDArray",
    "QuakeDialect",
    "QuakeRefType",
    "QuakeVeqType",
    "QuakeMeasureType",
    "CcDialect",
    "CcStdVecType",
]
