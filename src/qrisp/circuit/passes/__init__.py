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

from qrisp.circuit.passes.pass_manager import PassManager
from qrisp.circuit.passes.arange_swaps import arange_swaps
from qrisp.circuit.passes.cancel_inverses import cancel_inverses
from qrisp.circuit.passes.cancel_zero_controls import cancel_zero_controls
from qrisp.circuit.passes.combine_single_qubit_gates import combine_single_qubit_gates
from qrisp.circuit.passes.commute_swaps import commute_swaps
from qrisp.circuit.passes.convert_to_cz import convert_to_cz
from qrisp.circuit.passes.gray_synth_toffoli import gray_synth_toffoli, is_toffoli
from qrisp.circuit.passes.remove_barriers import remove_barriers
from qrisp.circuit.passes.resolve_swaps import resolve_swaps
from qrisp.circuit.passes.reverse_parallelize import reverse_parallelize
from qrisp.circuit.passes.manual_layout import manual_layout

__all__ = ["PassManager", "arange_swaps", "cancel_inverses", "cancel_zero_controls", "combine_single_qubit_gates", "commute_swaps", "convert_to_cz", "gray_synth_toffoli", "is_toffoli", "manual_layout", "remove_barriers", "resolve_swaps", "reverse_parallelize"]
