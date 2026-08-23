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

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np

from qrisp.circuit import Operation, QuantumCircuit, Qubit, XGate
from qrisp.simulator import QuantumState, advance_quantum_state, gen_res_dict

if TYPE_CHECKING:
    import stim

# Maps Qrisp gate names to the identically-behaving stim.TableauSimulator method name.
# Only "s_dg" differs from its Qrisp name (stim calls it s_dag).
_STIM_GATE_METHODS = {
    "x": "x",
    "y": "y",
    "z": "z",
    "h": "h",
    "cx": "cx",
    "cy": "cy",
    "cz": "cz",
    "s": "s",
    "s_dg": "s_dag",
}


class BufferedQuantumState:
    """Duck-typed quantum state carrier consumed by simulate_jaspr's interpreter.

    Quantum primitives' impl rules (e.g. append_impl, measure_implementation in
    qrisp.jasp.primitives) are written generically against any object exposing
    .append/.measure/.reset -- a real QuantumCircuit for the circuit-extraction
    interpreters, or this class for simulate_jaspr. Appended gates are only
    recorded into buffer_qc; they are not applied to the backend quantum_state
    (a qrisp.simulator.QuantumState or a stim.TableauSimulator) until a
    measurement, reset, or explicit apply_buffer() forces a flush -- letting
    many gates fuse before the comparatively expensive state update runs.
    """

    def __init__(self, simulator: Literal["qrisp", "stim"] = "qrisp") -> None:

        self.quantum_state: "QuantumState | stim.TableauSimulator"
        if simulator == "qrisp":
            self.quantum_state = QuantumState(n=0)
        elif simulator == "stim":
            # stim is an optional dependency (see the `stimulate` decorator's
            # docstring), so it can only be imported lazily, on actual use.
            import stim

            self.quantum_state = stim.TableauSimulator()
        else:
            raise Exception(f"Don't know simulator {simulator}")
        self.buffer_qc = QuantumCircuit(0)
        self.deallocated_qubits: list[Qubit] = []
        self.simulator: Literal["qrisp", "stim"] = simulator
        self.qubit_to_index_dict: dict[Qubit, int] = {}
        self.qubit_counter = 0
        self.gate_counts: dict[str, int] = {}

    def add_qubit(self) -> Qubit:
        """Allocate a fresh qubit on both the backend state and the buffer circuit."""
        if self.simulator == "qrisp":
            assert isinstance(self.quantum_state, QuantumState)
            self.quantum_state.add_qubit()
        qb = self.buffer_qc.add_qubit()
        self.qubit_to_index_dict[qb] = self.qubit_counter
        self.qubit_counter += 1
        return qb

    def _bump_gate_count(self, key: str, amount: int = 1) -> None:
        """Increment gate_counts[key] by amount, initializing it to amount if absent."""
        try:
            self.gate_counts[key] += amount
        except KeyError:
            self.gate_counts[key] = amount

    def append(self, op: Operation, qubits: Sequence[Qubit]) -> None:
        """Buffer a gate application without touching the backend state yet."""
        self.buffer_qc.append(op, qubits)
        if op.name not in ("qb_alloc", "qb_dealloc"):
            self._bump_gate_count(op.name)

    def apply_buffer(self) -> None:
        """Flush every buffered gate into the backend quantum state.
    
        For the "qrisp" backend, the buffered circuit is handed to
        qrisp.simulator.advance_quantum_state, which preprocesses
        (e.g. gate-grouping via ``group_qc``) and executes it, advancing
        ``self.quantum_state`` in place. For the "stim" backend, each
        buffered instruction is dispatched to the corresponding
        stim.TableauSimulator method.

        Afterwards, qubits marked for deallocation (``qb_dealloc``) are
        removed from ``buffer_qc`` and ``qubit_to_index_dict``, and the
        buffer is cleared.
        """

        if self.simulator == "qrisp":
            assert isinstance(self.quantum_state, QuantumState)
            self.quantum_state = advance_quantum_state(
                self.buffer_qc.copy(),
                self.quantum_state,
                self.deallocated_qubits,
                self.qubit_to_index_dict,
            )
        else:
            for instr in self.buffer_qc.data:
                qubit_indices = [self.qubit_to_index_dict[qb] for qb in instr.qubits]

                method_name = _STIM_GATE_METHODS.get(instr.op.name)
                if method_name is not None:
                    getattr(self.quantum_state, method_name)(*qubit_indices)
                elif instr.op.name not in ("qb_alloc", "qb_dealloc"):
                    raise Exception(f"Don't know how to simulate quantum gate {instr.op.name} with stim")

        for instr in self.buffer_qc.data:
            if instr.op.name == "qb_dealloc":
                self.buffer_qc.qubits.remove(instr.qubits[0])
                del self.qubit_to_index_dict[instr.qubits[0]]

        self.buffer_qc = self.buffer_qc.clearcopy()

    def measure(self, qubit: Sequence[Qubit], track_measurement: bool = True) -> bool:
        """Measure a qubit, flushing the buffer first."""
        if track_measurement:
            self._bump_gate_count("measure")

        self.apply_buffer()
        if self.simulator == "qrisp":
            assert isinstance(self.quantum_state, QuantumState)
            meas_res, self.quantum_state = self.quantum_state.measure(self.qubit_to_index_dict[qubit[0]], keep_res=True)
            return meas_res

        # stim is only imported lazily (see __init__); already loaded by this
        # point since self.simulator == "stim" is only reachable after it was.
        import stim

        assert isinstance(self.quantum_state, stim.TableauSimulator)
        return self.quantum_state.measure(self.qubit_to_index_dict[qubit[0]])

    def reset(self, qubit: Sequence[Qubit]) -> None:
        """Reset a qubit to the |0> state via measurement and a conditional flip."""

        if qubit[0] not in self.qubit_to_index_dict:
            return

        meas_res = self.measure(qubit, track_measurement=False)
        if meas_res:
            self.buffer_qc.append(XGate(), qubit)

    def copy(self) -> "BufferedQuantumState":
        """Return an independent copy of this buffered quantum state."""
        res = BufferedQuantumState(self.simulator)
        res.buffer_qc = self.buffer_qc.copy()
        res.deallocated_qubits = list(self.deallocated_qubits)
        res.quantum_state = self.quantum_state.copy()
        res.qubit_to_index_dict = dict(self.qubit_to_index_dict)
        res.qubit_counter = self.qubit_counter
        res.gate_counts = dict(self.gate_counts)
        return res

    def multi_measure(self, qubits: Sequence[Qubit], shots: int | None) -> dict[int, int] | dict[int, float]:
        """Measure several qubits at once and tabulate outcomes over the given shots.

        Only supported against the "qrisp" backend. The only caller of this method
        is the terminal-sampling evaluator, and simulate_jaspr already refuses to
        combine terminal sampling with simulator="stim", so this restriction is
        never actually hit in practice.
        """
        self._bump_gate_count("measure", len(qubits))

        self.apply_buffer()
        assert isinstance(self.quantum_state, QuantumState)
        qubit_indices = [self.qubit_to_index_dict[qb] for qb in qubits]
        mes_ints, probs = self.quantum_state.multi_measure(qubit_indices)

        if shots is not None and shots != 0:
            samples = np.random.choice(len(mes_ints), int(shots), p=probs)

            res: dict[int, int] = {}
            for k, v in gen_res_dict(samples).items():
                res[int(mes_ints[k])] = v
            return res

        res_probs: dict[int, float] = {}
        for i, mes_int in enumerate(mes_ints):
            res_probs[int(mes_int)] = probs[i]
        return res_probs
