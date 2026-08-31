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

"""Provide lowering passes from Jasp IR to CUDA-Q Quake IR."""

from qrisp.jasp.cudaq_interface.quake_lowering.jaspr_to_quake import _jaspr_to_quake_mlir
from qrisp.jasp.cudaq_interface.quake_lowering.validation_tools import _validate_quake_mlir
from qrisp.jasp.cudaq_interface.quake_lowering.dialects.quake_dialect import (
    QuakeDialect,
    QuakeMeasureType,
    QuakeRefType,
    QuakeVeqType,
)
from qrisp.jasp.cudaq_interface.quake_lowering.dialects.cc_dialect import CcDialect, CcMeasureHandleType, CcStdVecType

__all__ = [
    "QuakeDialect",
    "QuakeRefType",
    "QuakeVeqType",
    "QuakeMeasureType",
    "CcDialect",
    "CcMeasureHandleType",
    "CcStdVecType",
]
