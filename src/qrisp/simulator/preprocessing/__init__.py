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

from qrisp.simulator.preprocessing.circuit_preprocessing import circuit_preprocessor
from qrisp.simulator.preprocessing.circuit_reordering import reorder_circuit
from qrisp.simulator.preprocessing.disentangling import Disentangler, insert_disentangling
from qrisp.simulator.preprocessing.gate_grouping import (
    GroupedInstruction,
    IntegerCircuit,
    group_qc,
    optimal_grouping_recursion_parameter,
)
from qrisp.simulator.preprocessing.measurement_handling import (
    count_measurements_and_treat_alloc,
    extract_measurements,
    insert_multiverse_measurements,
)
