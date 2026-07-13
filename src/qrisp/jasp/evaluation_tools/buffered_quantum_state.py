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

import numpy as np

from qrisp.circuit import QuantumCircuit, XGate
from qrisp.simulator import QuantumState, advance_quantum_state, gen_res_dict


class BufferedQuantumState:
    """Incremental quantum state used to simulate Jasp-traced programs.

    Jasp interprets a jaxpr equation by equation instead of building the full
    quantum circuit ahead of time. Feeding the simulator backend one gate at
    a time would be extremely slow, so this class instead buffers the gates
    appended between two measurements/resets into a plain
    :class:`~qrisp.circuit.QuantumCircuit` (``self.buffer_qc``). Once a
    measurement, reset, or multi-measure is requested, the buffered circuit
    is flushed onto the actual quantum state (see :meth:`apply_buffer`),
    which is where the buffered circuit is grouped/preprocessed and executed.

    Two backends are supported:

    - ``"qrisp"``: Uses Qrisp's own statevector simulator
      (:class:`~qrisp.simulator.QuantumState`) via
      :func:`~qrisp.simulator.advance_quantum_state`.
    - ``"stim"``: Uses the `stim <https://github.com/quantumlib/Stim>`_
      Clifford simulator (``stim.TableauSimulator``) for circuits containing
      only Clifford gates.

    Parameters
    ----------
    simulator : str, optional
        Which simulator backend to use, either ``"qrisp"`` or ``"stim"``.
        The default is ``"qrisp"``.

    Attributes
    ----------
    quantum_state : QuantumState or stim.TableauSimulator
        The underlying (already-executed) quantum state.
    buffer_qc : QuantumCircuit
        Circuit collecting gates that have not yet been applied to
        ``quantum_state``.
    deallocated_qubits : list
        Qubits that have been deallocated (via ``qb_dealloc``) and are
        therefore no longer tracked.
    qubit_to_index_dict : dict
        Maps qubit objects to their integer index within the simulator
        backend.
    qubit_counter : int
        Number of qubits that have been allocated so far. Used to assign
        fresh indices in :meth:`add_qubit`.
    gate_counts : dict
        Running tally of how many times each gate/measurement has been
        applied, used for resource estimation.
    """

    def __init__(self, simulator="qrisp"):

        if simulator == "qrisp":
            self.quantum_state = QuantumState(n=0)
        elif simulator == "stim":
            import stim

            self.quantum_state = stim.TableauSimulator()
        else:
            raise Exception("Don't know simulator {simulator}")
        self.buffer_qc = QuantumCircuit(0)
        self.deallocated_qubits = []
        self.simulator = simulator
        self.qubit_to_index_dict = {}
        self.qubit_counter = 0
        self.gate_counts = {}

    def add_qubit(self):
        """Allocate a new qubit on both the backend state and the buffer circuit.

        Returns
        -------
        Qubit
            The newly allocated qubit, as tracked by ``buffer_qc``.
        """
        if self.simulator == "qrisp":
            self.quantum_state.add_qubit()
        qb = self.buffer_qc.add_qubit()
        self.qubit_to_index_dict[qb] = self.qubit_counter
        self.qubit_counter += 1
        return qb

    def append(self, op, qubits):
        """Append an operation to the instruction buffer.

        The operation is not immediately simulated; it is only added to
        ``buffer_qc`` and executed later, once :meth:`apply_buffer` is
        triggered (e.g. by a measurement or reset).

        Parameters
        ----------
        op : Operation
            The quantum gate/operation to append.
        qubits : list[Qubit]
            The qubits the operation acts on.
        """
        self.buffer_qc.append(op, qubits)
        try:
            if op.name != "qb_alloc" and op.name != "qb_dealloc":
                self.gate_counts[op.name] += 1
        except KeyError:
            self.gate_counts[op.name] = 1

    def apply_buffer(self):
        """Flush the buffered circuit onto the underlying quantum state.

        For the ``"qrisp"`` backend, the buffered circuit is handed to
        :func:`~qrisp.simulator.advance_quantum_state`, which preprocesses
        (e.g. gate-grouping via ``group_qc``) and executes it, advancing
        ``self.quantum_state`` in place. For the ``"stim"`` backend, each
        buffered instruction is dispatched to the corresponding
        ``stim.TableauSimulator`` method.

        Afterwards, qubits marked for deallocation (``qb_dealloc``) are
        removed from ``buffer_qc`` and ``qubit_to_index_dict``, and the
        buffer is cleared.
        """

        if self.simulator == "qrisp":
            self.quantum_state = advance_quantum_state(
                self.buffer_qc.copy(),
                self.quantum_state,
                self.deallocated_qubits,
                self.qubit_to_index_dict,
            )
        else:
            for instr in self.buffer_qc.data:
                qubit_indices = [self.qubit_to_index_dict[qb] for qb in instr.qubits]

                if instr.op.name == "x":
                    self.quantum_state.x(*qubit_indices)
                elif instr.op.name == "y":
                    self.quantum_state.y(*qubit_indices)
                elif instr.op.name == "z":
                    self.quantum_state.z(*qubit_indices)
                elif instr.op.name == "h":
                    self.quantum_state.h(*qubit_indices)
                elif instr.op.name == "cx":
                    self.quantum_state.cx(*qubit_indices)
                elif instr.op.name == "cy":
                    self.quantum_state.cy(*qubit_indices)
                elif instr.op.name == "cz":
                    self.quantum_state.cz(*qubit_indices)
                elif instr.op.name == "s":
                    self.quantum_state.s(*qubit_indices)
                elif instr.op.name == "s_dg":
                    self.quantum_state.s_dag(*qubit_indices)
                elif instr.op.name not in ["qb_alloc", "qb_dealloc"]:
                    raise Exception(f"Don't know how to simulate quantum gate {instr.op.name} with stim")

        for instr in self.buffer_qc.data:
            if instr.op.name == "qb_dealloc":
                self.buffer_qc.qubits.remove(instr.qubits[0])
                del self.qubit_to_index_dict[instr.qubits[0]]

        self.buffer_qc = self.buffer_qc.clearcopy()

    def measure(self, qubit, track_measurement=True):
        """Measure a single qubit, flushing the buffer beforehand.

        Parameters
        ----------
        qubit : list[Qubit]
            A single-element list containing the qubit to measure.
        track_measurement : bool, optional
            Whether to include this measurement in ``gate_counts``. The
            default is True.

        Returns
        -------
        bool or int
            The measurement outcome.
        """
        if track_measurement:
            try:
                self.gate_counts["measure"] += 1
            except KeyError:
                self.gate_counts["measure"] = 1

        self.apply_buffer()
        if self.simulator == "qrisp":
            meas_res, self.quantum_state = self.quantum_state.measure(self.qubit_to_index_dict[qubit[0]], keep_res=True)
            return meas_res
        elif self.simulator == "stim":
            return self.quantum_state.measure(self.qubit_to_index_dict[qubit[0]])

    def reset(self, qubit):
        """Reset a qubit to the |0> state.

        Implemented by measuring the qubit and, if the outcome was 1,
        appending an ``XGate`` to flip it back to |0>. If the qubit is
        already deallocated (not in ``qubit_to_index_dict``), this is a
        no-op.

        Parameters
        ----------
        qubit : list[Qubit]
            A single-element list containing the qubit to reset.
        """

        if qubit[0] not in self.qubit_to_index_dict:
            return

        meas_res = self.measure(qubit, track_measurement=False)
        if meas_res:
            self.buffer_qc.append(XGate(), qubit)

    def copy(self):
        """Create an independent copy of this buffered quantum state.

        Returns
        -------
        BufferedQuantumState
            A deep-enough copy where mutating the buffer circuit, quantum
            state, or bookkeeping dictionaries of the copy does not affect
            the original.
        """
        res = BufferedQuantumState()
        res.buffer_qc = self.buffer_qc.copy()
        res.deallocated_qubits = list(self.deallocated_qubits)
        res.quantum_state = self.quantum_state.copy()
        res.qubit_to_index_dict = dict(self.qubit_to_index_dict)
        res.qubit_counter = self.qubit_counter
        return res

    def multi_measure(self, qubits, shots):
        """Measure several qubits at once and optionally sample shots from the result.

        This is more efficient than measuring qubits one at a time because
        it samples directly from the terminal probability distribution
        instead of re-simulating the state for each shot (used by the
        ``terminal_sampling`` feature).

        Parameters
        ----------
        qubits : list[Qubit]
            The qubits to measure.
        shots : int or None
            Number of samples to draw. If ``None`` or ``0``, the full
            probability distribution is returned instead of samples.

        Returns
        -------
        dict
            If ``shots`` is given, a mapping from measured integer outcomes
            to observed counts. Otherwise, a mapping from measured integer
            outcomes to probabilities.
        """
        try:
            self.gate_counts["measure"] += len(qubits)
        except KeyError:
            self.gate_counts["measure"] = len(qubits)

        self.apply_buffer()
        qubit_indices = [self.qubit_to_index_dict[qb] for qb in qubits]
        mes_ints, probs = self.quantum_state.multi_measure(qubit_indices)

        if shots is not None and shots != 0:
            samples = np.random.choice(len(mes_ints), int(shots), p=probs)

            samples = gen_res_dict(samples)
            res = {}
            for k, v in samples.items():
                res[mes_ints[k]] = v
            return res
        else:
            res = {}
            for i in range(len(mes_ints)):
                res[mes_ints[i]] = probs[i]
            return res
