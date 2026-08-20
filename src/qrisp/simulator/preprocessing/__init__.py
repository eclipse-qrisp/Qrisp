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

================================================================================
Qrisp Simulator Circuit Preprocessor
================================================================================

This package is responsible for optimizing and preprocessing quantum circuits
before they are dispatched to the Qrisp statevector simulator backend.
Because simulating large statevectors is exponentially expensive in both time
and memory, it applies several advanced heuristic transformations to
significantly reduce computational overhead.

The preprocessor acts as a compiler pass, modifying the circuit's structure
without altering its mathematical outcome. It is split into the following
submodules, see each for details:

1. :mod:`~qrisp.simulator.preprocessing.gate_grouping` -- Gate Grouping
2. :mod:`~qrisp.simulator.preprocessing.disentangling` -- State Disentangling
3. :mod:`~qrisp.simulator.preprocessing.measurement_handling` -- Measurement
   and Allocation Management
4. :mod:`~qrisp.simulator.preprocessing.circuit_reordering` -- Circuit
   Reordering
5. :mod:`~qrisp.simulator.preprocessing.circuit_preprocessing` -- The main
   wrapper `circuit_preprocessor(qc)`, which combines all of the above into
   a single preprocessing pipeline.
================================================================================
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
