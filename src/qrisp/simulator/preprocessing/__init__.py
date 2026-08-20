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
submodules:

1. Gate Grouping (:mod:`~qrisp.simulator.preprocessing.gate_grouping`)
   Applying many small unitary matrices (e.g., 1-qubit or 2-qubit gates) to a
   massive 2^n statevector is inefficient due to high memory bandwidth usage.
   - The `GroupedInstruction` class and `group_qc` function recursively search
     the circuit for sets of small, adjacent, or commuting gates.
   - These gates are grouped together so their combined "medium-sized" unitary
     can be pre-calculated. Applying one medium unitary saves millions of
     floating-point operations (FLOPs) compared to applying many small ones.
   - To make this search fast, `IntegerCircuit` translates the circuit into a
     bitwise representation, allowing the Numba-jitted search functions
     (`binary_get_circuit_block_jitted`, `binary_get_circuit_block_jitted_chunked`)
     to evaluate gate commutativity using ultra-fast bitwise logic.
     It features a dual-path that vectorizes qubit bitmasks into chunks to bypass
     64-bit memory limitations on massive statevector simulations.

2. State Disentangling (:mod:`~qrisp.simulator.preprocessing.disentangling`)
   Simulating a 50+ qubit statevector is practically impossible if fully entangled.
   However, many algorithms naturally "disentangle" certain qubits during execution
   (e.g., via measurements, resets, or specific uncomputations).
   - `insert_disentangling` identifies points in the circuit where wave-function
     branches no longer interact.
   - It inserts a custom `disentangle` instruction. The simulator catches this and
     splits the massive simulation into smaller, separate, parallelizable wave-functions,
     effectively turning an intractable problem into a solvable one.

3. Measurement and Allocation Management
   (:mod:`~qrisp.simulator.preprocessing.measurement_handling`)
   - `extract_measurements` and `count_measurements_and_treat_alloc` optimize
     how classical measurements and temporary qubit allocations are handled.
   - `insert_multiverse_measurements` handles deferred measurement patterns by
     introducing ancilla qubits and CNOT gates, ensuring probability distributions
     are correctly captured without breaking coherence prematurely.

4. The Main Wrapper: `circuit_preprocessor(qc)`
   (:mod:`~qrisp.simulator.preprocessing.circuit_preprocessing`)
   The main entry point for this package. It evaluates the incoming circuit,
   applies disentangling (if the circuit is dangerously wide, e.g., >45 qubits),
   groups the gates for performance, and finally reorders the circuit
   (:mod:`~qrisp.simulator.preprocessing.circuit_reordering`) to safely push
   measurements, resets, and disentanglers to the end of execution blocks.
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
