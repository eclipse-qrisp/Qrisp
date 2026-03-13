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


"""This module defines the TracingQuantumSession class, which manages the state of quantum circuit construction during JAX tracing in Jasp mode."""

import weakref
from collections import deque
from typing import Deque, Dict, List, Tuple


import jax
from jax.extend.core import JaxprEqn
from sympy import symbols

from qrisp.core.quantum_variable import QuantumVariable
from qrisp.jasp.primitives import create_qubits, delete_qubits_p, quantum_gate_p
from qrisp.jasp.primitives.abstract_quantum_state import AbstractQuantumState
from qrisp.jasp.tracing_logic.dynamic_qubit_array import DynamicQubitArray

greek_letters = symbols(
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega"
)


class SingletonMeta(type):
    """Metaclass for implementing the singleton pattern."""

    _instances: Dict["SingletonMeta", "TracingQuantumSession"] = {}

    def __call__(cls, *args, **kwargs):
        """Return the singleton instance of the class."""
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


# Question 1: This class is a singleton by construction. Therefore, there is only one instance
# of TracingQuantumSession that can exist at any time.
# Therefore, what is the purpose of the trc_qs_stack attribute,
# which is a deque that can hold multiple instances of TracingQuantumSession or None?


# Question 2: For the same reason as above, what is the point of distinguishing
# between class attributes (like trc_qs_stack) and instance attributes (like abs_qst, qubit_cache, qv_list, deleted_qv_list)?


# Question 3: `qubit_cache` is always initialized as an empty dictionary in the constructor,
# and is also reset to an empty dictionary in the `start_tracing` method. Looks like that this always initialized externally,
# and is never modified internally in the class. If it is an attribute of the class, it should be initialized inside the class.

# Question 4: Why do we store self.abs_qst instead of just using the abs_qst passed to start_tracing?


class TracingQuantumSession(metaclass=SingletonMeta):
    """
    Central context for managing quantum circuit construction during JAX tracing in Jasp mode.
    """

    trc_qs_stack: Deque["TracingQuantumSession | None"] = deque()

    abs_qst_stack: Deque[AbstractQuantumState | None] = deque()
    qubit_cache_stack: Deque[Dict] = deque()
    qv_stack: Deque[Tuple[List[QuantumVariable], List[QuantumVariable]]] = deque()

    def __init__(self):
        """
        Construct a new tracing quantum session and make it the current session.
        """

        TracingQuantumSession.trc_qs_stack.appendleft(self)

        self.abs_qst: AbstractQuantumState | None = None
        self.qubit_cache: Dict = {}

        self.qv_list: List[QuantumVariable] = []
        self.deleted_qv_list: List[QuantumVariable] = []

    def _ensure_active_trace(self) -> None:
        """Ensure that there is an active tracing context and that the quantum state is being tracked."""
        if self.abs_qst is None or self.abs_qst._trace is not jax.core.trace_ctx.trace:
            raise RuntimeError(
                """Lost track of QuantumState during tracing. This might have been caused by a missing quantum_kernel decorator or not using quantum prefix control (like q_fori_loop, q_cond). Please visit https://www.qrisp.eu/reference/Jasp/Quantum%20Kernel.html for more details"""
            )

    def start_tracing(self, abs_qst: AbstractQuantumState) -> None:
        """Start a new tracing context with the provided abstract quantum state."""

        # Save current state to stacks
        TracingQuantumSession.abs_qst_stack.append(self.abs_qst)
        TracingQuantumSession.qubit_cache_stack.append(self.qubit_cache)
        TracingQuantumSession.qv_stack.append((self.qv_list, self.deleted_qv_list))

        # Initialize new tracing state
        self.abs_qst = abs_qst
        self.qubit_cache = {}

        self.qv_list = []
        self.deleted_qv_list = []

    def conclude_tracing(self) -> AbstractQuantumState | None:
        """
        Conclude the current tracing context and return the traced circuit.

        Returns
        -------
        AbstractQuantumState or None
            The abstract quantum state representing the traced circuit, or None if there was no active trace.
        """

        current_abs_qst = self.abs_qst

        # Restore previous tracing state
        self.abs_qst = TracingQuantumSession.abs_qst_stack.pop()
        self.qubit_cache = TracingQuantumSession.qubit_cache_stack.pop()
        self.qv_list, self.deleted_qv_list = TracingQuantumSession.qv_stack.pop()

        return current_abs_qst

    def append(
        self,
        operation,
        qubits: List | None = None,
        clbits: List | None = None,
        param_tracers: List | None = None,
    ) -> None:
        """
        Append an operation to the currently traced circuit.

        Parameters
        ----------
        operation : qrisp.circuit.operation.Operation
            The quantum gate or instruction to append.

        qubits : list, optional
            Qubit arguments. May be plain qubit tracers, lists, QuantumVariables,
            DynamicQubitArrays, or QuantumArrays; dispatch is handled automatically.

        clbits : list, optional
            Classical bit arguments. Must be empty in Jasp mode.

        param_tracers : list, optional
            JAX tracers for any parametric angles in the operation.

        """

        self._ensure_active_trace()

        if clbits is not None and len(clbits) > 0:
            raise ValueError(
                "Tried to append Operation with non-zero classical bits in JAX mode."
            )

        qubits = qubits if qubits is not None else []
        param_tracers = param_tracers if param_tracers is not None else []

        from qrisp.core import QuantumArray
        from qrisp.jasp import jrange

        if isinstance(qubits[0], (QuantumVariable, DynamicQubitArray)):
            for i in jrange(qubits[0].size):
                self.append(
                    operation,
                    [qubits[j][i] for j in range(operation.num_qubits)],
                    param_tracers=param_tracers,
                )
            return

        if isinstance(qubits[0], list):
            for i in range(len(qubits[0])):
                self.append(
                    operation,
                    [qubits[j][i] for j in range(operation.num_qubits)],
                    param_tracers=param_tracers,
                )
            return

        if isinstance(qubits[0], QuantumArray):
            for i in range(1, len(qubits)):
                if not isinstance(qubits[i], QuantumArray):
                    raise ValueError(
                        f"Tried to apply multi-qubit gate to mixed qubit argument types (QuantumArray + {type(qubits[i])})"
                    )

                if qubits[i].shape != qubits[0].shape:
                    raise ValueError(
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

        args = list(qubits) + list(param_tracers) + [self.abs_qst]
        self.abs_qst = quantum_gate_p.bind(*args, gate=operation)

    def register_qv(self, qv: QuantumVariable, size: int | jax.Array) -> None:
        """Register a quantum variable in the session and allocate its qubits."""

        self._ensure_active_trace()

        if size is not None:
            qv.reg = self._request_qubits(size)

        # Register in the list of active quantum variables
        self.qv_list.append(qv)
        qv.qs = self

        QuantumVariable.live_qvs.append(weakref.ref(qv))
        qv.creation_time = int(QuantumVariable.creation_counter[0])
        QuantumVariable.creation_counter += 1

    def _request_qubits(self, amount: int | jax.Array) -> DynamicQubitArray:
        """Request and allocate the specified number of qubits."""

        qb_array_tracer, self.abs_qst = create_qubits(amount, self.abs_qst)
        return DynamicQubitArray(qb_array_tracer)

    def delete_qv(self, qv: QuantumVariable, verify: bool = False) -> None:
        """Delete the specified quantum variable and free its qubits."""

        self._ensure_active_trace()

        if verify:
            raise ValueError("Tried to verify deletion in tracing mode.")

        # Check if quantum variable appears in this session
        if qv.name not in [qv.name for qv in self.qv_list]:
            raise ValueError(
                "Tried to remove a non-existent quantum variable from quantum session"
            )

        self._clear_qubits(qv.reg)

        # Remove quantum variable from list
        for idx, temp_qv in enumerate(self.qv_list):
            if temp_qv.name == qv.name:
                del self.qv_list[idx]
                break

        self.deleted_qv_list.append(qv)

    def _clear_qubits(self, qubits: DynamicQubitArray) -> None:
        """Free the specified qubits."""

        self.abs_qst = delete_qubits_p.bind(qubits.tracer, self.abs_qst)

    @classmethod
    def release(cls) -> None:
        """Release the current tracing quantum session, allowing a new one to be created."""
        cls.trc_qs_stack.popleft()

    @classmethod
    def get_instance(cls) -> "TracingQuantumSession":
        """Return the current active TracingQuantumSession instance."""
        for instance in cls.trc_qs_stack:
            if instance is not None:
                return instance
        raise RuntimeError("No active TracingQuantumSession.")


tracing_qs_singleton = TracingQuantumSession()


def check_for_tracing_mode() -> bool:
    """Check if the current execution context is within a JAX tracing session."""
    return hasattr(jax._src.core.trace_ctx.trace, "frame")


def get_last_equation(i: int = -1) -> JaxprEqn:
    """Get the last equation in the current JAX trace."""
    return jax._src.core.trace_ctx.trace.frame.tracing_eqns[i]()


def check_live(tracer):
    """Check if the provided tracer is live (i.e., part of the current JAX trace)."""
    if tracer is None:
        return True
    return tracer._trace.main.jaxpr_stack
