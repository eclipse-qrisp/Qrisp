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

from qrisp.circuit.pass_management.passes.arrange_swaps import arrange_swaps
from qrisp.circuit.pass_management.passes.fuse_adjacents import fuse_adjacents
from qrisp.circuit.pass_management.passes.cancel_zero_controls import cancel_zero_controls
from qrisp.circuit.pass_management.passes.combine_single_qubit_gates import combine_single_qubit_gates
from qrisp.circuit.pass_management.passes.commute_swaps import commute_swaps
from qrisp.circuit.pass_management.passes.convert_to_cx import convert_to_cx
from qrisp.circuit.pass_management.passes.convert_to_cz import convert_to_cz
from qrisp.circuit.pass_management.passes.decompose import decompose
from qrisp.circuit.pass_management.passes.gray_synth_toffoli import gray_synth_toffoli, is_toffoli
from qrisp.circuit.pass_management.passes.manual_layout import manual_layout
from qrisp.circuit.pass_management.passes.remove_barriers import remove_barriers
from qrisp.circuit.pass_management.passes.resolve_swaps import resolve_swaps
from qrisp.circuit.pass_management.passes.reverse_parallelize import reverse_parallelize
from qrisp.circuit.pass_management.passes.visualize import visualize
from qrisp.circuit.pass_management.passes.convert_to_prx import PRXGate, convert_to_prx

__all__ = [
    "arrange_swaps",
    "fuse_adjacents",
    "cancel_zero_controls",
    "combine_single_qubit_gates",
    "commute_swaps",
    "convert_to_cx",
    "convert_to_cz",
    "decompose",
    "gray_synth_toffoli",
    "is_toffoli",
    "manual_layout",
    "remove_barriers",
    "resolve_swaps",
    "reverse_parallelize",
    "visualize",
    "convert_to_prx",
    "PRXGate"
]
