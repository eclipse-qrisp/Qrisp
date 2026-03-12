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

import weakref
from collections import deque
from typing import Deque, List, cast

import jax
from sympy import symbols

from qrisp.core.quantum_variable import QuantumVariable
from qrisp.jasp.primitives import create_qubits, delete_qubits_p, quantum_gate_p
from qrisp.jasp.primitives.abstract_quantum_state import AbstractQuantumState
from qrisp.jasp.tracing_logic.dynamic_qubit_array import DynamicQubitArray

greek_letters = symbols(
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega"
)


class TracingQuantumSession:
    """
    Manage tracing-time state for building quantum circuits in Jasp mode.

    This class acts as the central recording context while JAX traces a Python
    function that constructs a quantum program. In particular, it maintains:

    - a reference to the currently active :class:`~qrisp.jasp.AbstractQuantumState`
      being built (``self.abs_qst``),

    - a cache for qubits allocated during tracing (``self.qubit_cache``),

    - a list of "live" quantum variables currently registered in this session
      (``self.qv_list``),

    - a garbage-collection policy controlling what happens when a quantum variable
      goes out of scope during tracing (``self.gc_mode``).

    The session supports nested tracing. Each call to :meth:`start_tracing`
    pushes the previous tracing state onto internal stacks; :meth:`conclude_tracing`
    restores the previous state and returns the circuit traced in the nested scope.
    """

    tr_qs_container: Deque = deque([None])
    abs_qst_stack: List[AbstractQuantumState | None] = []
    qubit_cache_stack: List = []

    def __init__(self):
        """
        Construct a new tracing quantum session and make it the current session.

        The session starts with no active circuit.
        Use :meth:`start_tracing` to begin recording into a provided abstract circuit object.
        """

        self.abs_qst = None
        self.qv_list = []
        self.deleted_qv_list = []
        self.qubit_cache = None
        TracingQuantumSession.tr_qs_container.appendleft(self)
        self.qv_stack = []

    def start_tracing(self, abs_qst: AbstractQuantumState) -> None:
        """Start a new tracing context with the provided abstract quantum state."""

        # Save current state to stacks
        self.abs_qst_stack.append(self.abs_qst)
        self.qubit_cache_stack.append(self.qubit_cache)
        self.qv_stack.append((self.qv_list, self.deleted_qv_list))

        # Initialize new tracing state
        self.abs_qst = abs_qst
        self.qubit_cache = {}
        self.qv_list = []
        self.deleted_qv_list = []

    def conclude_tracing(self) -> AbstractQuantumState | None:
        """Conclude the current tracing context and return the traced circuit."""

        temp = self.abs_qst

        # Restore previous tracing state
        self.abs_qst = self.abs_qst_stack.pop()
        self.qubit_cache = self.qubit_cache_stack.pop()
        self.qv_list, self.deleted_qv_list = self.qv_stack.pop()

        return temp

    def append(self, operation, qubits=[], clbits=[], param_tracers=[]):

        if not self.abs_qst._trace is jax.core.trace_ctx.trace:
            raise Exception(
                """Lost track of QuantumState during tracing. This might have been caused by a missing quantum_kernel decorator or not using quantum prefix control (like q_fori_loop, q_cond). Please visit https://www.qrisp.eu/reference/Jasp/Quantum%20Kernel.html for more details"""
            )

        if len(clbits):
            raise Exception(
                "Tried to append Operation with non-zero classical bits in JAX mode."
            )

        from qrisp.core import QuantumArray, QuantumVariable
        from qrisp.jasp import jrange

        if isinstance(qubits[0], (QuantumVariable, DynamicQubitArray)):

            for i in jrange(qubits[0].size):
                self.append(
                    operation,
                    [qubits[j][i] for j in range(operation.num_qubits)],
                    param_tracers=param_tracers,
                )
            return

        elif isinstance(qubits[0], list):

            for i in range(len(qubits[0])):
                self.append(
                    operation,
                    [qubits[j][i] for j in range(operation.num_qubits)],
                    param_tracers=param_tracers,
                )
            return

        elif isinstance(qubits[0], QuantumArray):

            for i in range(1, len(qubits)):
                if not isinstance(qubits[i], QuantumArray):
                    raise Exception(
                        f"Tried to apply multi-qubit gate to mixed qubit argument types (QuantumArray + {type(qubits[i])})"
                    )

                if qubits[i].shape != qubits[0].shape:
                    raise Exception(
                        "Tried to apply multi-qubit quantum gate to QuantumArrays of differing shape."
                    )

            flattened_qubits = [qubits[i].flatten() for i in range(len(qubits))]

            for i in jrange(flattened_qubits[0].size):
                self.append(
                    operation,
                    [flattened_qubits[j][i] for j in range(operation.num_qubits)],
                    param_tracers=param_tracers,
                )
            return

        temp_op = operation.copy()

        temp_op.params = list(temp_op.params)
        for i in range(len(temp_op.params)):
            temp_op.params[i] = greek_letters[i]

        self.abs_qst = quantum_gate_p.bind(
            *([b for b in qubits] + param_tracers + [self.abs_qst]), gate=operation
        )

    def register_qv(self, qv, size):

        if not self.abs_qst._trace is jax.core.trace_ctx.trace:
            raise Exception(
                """Lost track of QuantumState during tracing. This might have been caused by a missing quantum_kernel decorator or not using quantum prefix control (like q_fori_loop, q_cond). Please visit https://www.qrisp.eu/reference/Jasp/Quantum%20Kernel.html for more details"""
            )

        # Determine amount of required qubits
        if size is not None:
            qv.reg = self.request_qubits(size)

        # Register in the list of active quantum variable
        self.qv_list.append(qv)
        qv.qs = self

        QuantumVariable.live_qvs.append(weakref.ref(qv))
        qv.creation_time = int(QuantumVariable.creation_counter[0])
        QuantumVariable.creation_counter += 1

    def request_qubits(self, amount) -> DynamicQubitArray:
        """Request and allocate the specified number of qubits."""
        qb_array_tracer, self.abs_qst = create_qubits(amount, self.abs_qst)
        return DynamicQubitArray(qb_array_tracer)

    def delete_qv(self, qv: QuantumVariable, verify: bool = False) -> None:
        """Delete the specified quantum variable and free its qubits."""

        if not self.abs_qst._trace is jax.core.trace_ctx.trace:
            raise Exception(
                """Lost track of QuantumState during tracing. This might have been caused by a missing quantum_kernel decorator or not using quantum prefix control (like q_fori_loop, q_cond). Please visit https://www.qrisp.eu/reference/Jasp/Quantum%20Kernel.html for more details"""
            )

        if verify:
            raise ValueError("Tried to verify deletion in tracing mode.")

        # Check if quantum variable appears in this session
        if qv.name not in [qv.name for qv in self.qv_list]:
            raise ValueError(
                "Tried to remove a non-existent quantum variable from quantum session"
            )

        self.clear_qubits(qv.reg)

        # Remove quantum variable from list
        self.qv_list = [temp_qv for temp_qv in self.qv_list if temp_qv.name != qv.name]

        self.deleted_qv_list.append(qv)

    def clear_qubits(self, qubits) -> None:
        """Free the specified qubits."""
        self.abs_qst = delete_qubits_p.bind(qubits.tracer, self.abs_qst)

    @classmethod
    def release(cls) -> None:
        cls.tr_qs_container.popleft()

    @classmethod
    def get_instance(cls) -> "TracingQuantumSession":
        return cast("TracingQuantumSession", cls.tr_qs_container[0])


tracing_qs_singleton = TracingQuantumSession()


def check_for_tracing_mode() -> bool:
    """Check if the current execution context is within a JAX tracing session."""
    return hasattr(jax._src.core.trace_ctx.trace, "frame")


def get_last_equation(i=-1):
    return jax._src.core.trace_ctx.trace.frame.tracing_eqns[i]()


def check_live(tracer):
    if tracer is None:
        return True
    return not not tracer._trace.main.jaxpr_stack
