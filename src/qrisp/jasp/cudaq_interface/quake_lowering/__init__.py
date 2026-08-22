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

# Jasp → Quake (memory-semantics) lowering backend.
# ====================================================
#
# This package adds a Quake emission backend to the Qrisp MLIR pipeline.
# The main entry point is jaspr_to_quake_mlir, which lowers a Jaspr all the
# way from the generic Jasp xDSL dialect down to the CUDA-Q Quake and CC
# dialects.
#
# Sub-modules
# -----------
# dialects.quake_dialect
#     xDSL type and op definitions for the CUDA-Q Quake dialect.
# dialects.cc_dialect
#     xDSL type and op definitions for the CUDA-Q CC (classical control) dialect.
# jasp_to_quake
#     PASS 1: QuantumState elimination + Jasp→Quake op rewriting (split into
#     pass1a_lower_jasp_to_quake and pass1b_strip_qst, with shared
#     gate_mapping/helper_functions modules).
# pass2_scf_to_cc
#     PASS 2: SCF → CC dialect lowering.
# pass3_scalar_tensor_unwrap
#     PASS 3: Scalar/tensor unwrapping and function-signature rewrite.
# pass4_ranked_tensor_to_array
#     PASS 4: Ranked-tensor → array rewriting.
# pass5_array_to_stdvec
#     PASS 5: Array → !cc.stdvec rewriting.
# safeguard_no_ranked_tensor_linalg
#     Early-fail verifier that raises a helpful error when unsupported
#     ranked-tensor/linalg operations survive to the end of the pipeline.
# validation_tools
#     Structural validation helpers for the emitted Quake/CC module.
# jaspr_to_quake
#     Pipeline entry-point (jaspr_to_quake_mlir), orchestrating passes 1-5.

from qrisp.jasp.cudaq_interface.quake_lowering.jaspr_to_quake import jaspr_to_quake_mlir
from qrisp.jasp.cudaq_interface.quake_lowering.validation_tools import validate_quake_mlir
from qrisp.jasp.cudaq_interface.quake_lowering.dialects.quake_dialect import (
    QuakeDialect,
    QuakeMeasureType,
    QuakeRefType,
    QuakeVeqType,
)
from qrisp.jasp.cudaq_interface.quake_lowering.dialects.cc_dialect import CcDialect, CcMeasureHandleType, CcStdVecType

__all__ = [
    "jaspr_to_quake_mlir",
    "validate_quake_mlir",
    "QuakeDialect",
    "QuakeRefType",
    "QuakeVeqType",
    "QuakeMeasureType",
    "CcDialect",
    "CcMeasureHandleType",
    "CcStdVecType",
]
