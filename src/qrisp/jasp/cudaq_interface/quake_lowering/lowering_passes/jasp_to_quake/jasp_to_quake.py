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

"""Run the complete Jasp-to-Quake lowering pipeline."""

from qrisp.jasp.cudaq_interface.quake_lowering.lowering_passes.jasp_to_quake.lower_jasp_to_quake import (
    _lower_jasp_to_quake,
)
from qrisp.jasp.cudaq_interface.quake_lowering.lowering_passes.jasp_to_quake.strip_qst import _strip_qst


def _jasp_to_quake(module, execution_mode="run"):
    """Full Jasp→Quake lowering pipeline."""
    _lower_jasp_to_quake(module, execution_mode)  # Lower operations
    _strip_qst(module, execution_mode)  # Remove QuantumState structure
