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
from typing import Deque, Dict, Sequence

import jax
from jax.extend.core import JaxprEqn
from sympy import symbols

from qrisp.circuit.clbit import Clbit
from qrisp.circuit.operation import Operation
from qrisp.circuit.qubit import Qubit
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


class TracingQuantumSession(metaclass=SingletonMeta):
    """Manage tracing-time state for building quantum circuits in Jasp mode."""

    tr_qs_instance: "TracingQuantumSession | None" = None
    abs_qst_stack: Deque[AbstractQuantumState | None] = deque()
    qubit_cache_stack: Deque[Dict | None] = deque()

    def __init__(self):
        """
        Construct a new tracing quantum session and make it the current session.

        The session starts with no active circuit.
        Use :meth:`start_tracing` to begin recording into a provided abstract circuit object.
        """

        TracingQuantumSession.tr_qs_instance = self

        self.abs_qst = None
        self.qubit_cache = None

        # TODO: remove these (they should no longer be needed with the new garbage collection policy)
        self.qv_list = []
        self.deleted_qv_list = []
        self.qv_stack = []

    def _check_trace(self):
        """Raise if current JAX trace no longer matches the session's quantum state."""

        # NOTE: We use base Exception class (not a subclass) because JAX's tracing machinery
        # uses broad `except Exception` guards internally. Subclasses can escape
        # those guards, causing performance regressions as every call retrace.
        if not self.abs_qst._trace is jax.core.trace_ctx.trace:
            raise Exception(
                """Lost track of QuantumState during tracing. This might have been caused by a missing quantum_kernel decorator or not using quantum prefix control (like q_fori_loop, q_cond). Please visit https://www.qrisp.eu/reference/Jasp/Quantum%20Kernel.html for more details"""
            )

    def start_tracing(self, abs_qst: AbstractQuantumState) -> None:
        """Start a new tracing context with the provided abstract quantum state."""

        # Save current state to stacks
        TracingQuantumSession.abs_qst_stack.append(self.abs_qst)
        TracingQuantumSession.qubit_cache_stack.append(self.qubit_cache)

        self.qv_stack.append((self.qv_list, self.deleted_qv_list))

        # Initialize new tracing state
        self.abs_qst = abs_qst
        self.qubit_cache = {}
        self.qv_list = []
        self.deleted_qv_list = []

    def conclude_tracing(self) -> AbstractQuantumState | None:
        """Conclude the current tracing context and return the traced circuit."""

        tmp_abs_qst = self.abs_qst

        # Restore previous tracing state
        self.abs_qst = TracingQuantumSession.abs_qst_stack.pop()
        self.qubit_cache = TracingQuantumSession.qubit_cache_stack.pop()

        self.qv_list, self.deleted_qv_list = self.qv_stack.pop()

        return tmp_abs_qst

    def append(
        self,
        operation: Operation,
        qubits: Sequence[Qubit] | None = None,
        clbits: Sequence[Clbit] | None = None,
        param_tracers: Sequence | None = None,
    ):
        """
        Append an operation to the currently traced circuit, with the provided qubits, classical bits, and parameter tracers.


        Parameters
        ----------
        operation : Operation
            The operation to append to the circuit.

        qubits : Sequence[Qubit] | None, optional
            The qubits to which the operation should be applied. If None, defaults to an empty list.

        clbits : Sequence[Clbit] | None, optional
            The classical bits associated with the operation. If None, defaults to an empty list.

        param_tracers : Sequence | None, optional
            The parameter tracers for the operation, used for parameterized gates. If None, defaults to an empty list.

        """

        self._check_trace()

        qubits = qubits if qubits is not None else []
        clbits = clbits if clbits is not None else []
        param_tracers = param_tracers if param_tracers is not None else []

        if len(clbits):
            raise Exception(
                "Tried to append Operation with non-zero classical bits in JAX mode."
            )

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

        args = list(qubits) + list(param_tracers) + [self.abs_qst]
        self.abs_qst = quantum_gate_p.bind(*args, gate=operation)

    def register_qv(self, qv: QuantumVariable, size: int | jax.Array) -> None:
        """
        Register a quantum variable in the session and allocate its qubits.

        Parameters
        ----------
        qv : QuantumVariable
            The quantum variable to register in the session.

        size : int | jax.Array
            The number of qubits to allocate for the quantum variable.
            Can be a static integer or a JAX Tracer representing a dynamic size.

        """

        self._check_trace()

        # Determine amount of required qubits
        if size is not None:
            qv.reg = self.request_qubits(size)

        # Register in the list of active quantum variables
        self.qv_list.append(qv)
        qv.qs = self

        QuantumVariable.live_qvs.append(weakref.ref(qv))
        qv.creation_time = int(QuantumVariable.creation_counter[0])
        QuantumVariable.creation_counter += 1

    def request_qubits(self, amount: int | jax.Array) -> DynamicQubitArray:
        """Request and allocate the specified number of qubits."""
        qb_array_tracer, self.abs_qst = create_qubits(amount, self.abs_qst)
        return DynamicQubitArray(qb_array_tracer)

    def delete_qv(self, qv: QuantumVariable, verify: bool = False) -> None:
        """
        Delete the specified quantum variable and free its qubits.

        Parameters
        ----------
        qv : QuantumVariable
            The quantum variable to delete from the session.

        verify : bool, optional
            If True, perform additional checks to verify that the quantum variable can be safely deleted.
            This is intended for use in non-tracing contexts and will raise an error if attempted during tracing.

        """

        self._check_trace()

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
        """
        Free the specified qubits.

        Parameters
        ----------
        qubits : DynamicQubitArray
            The qubits to free from the session.

        """
        self.abs_qst = delete_qubits_p.bind(qubits.tracer, self.abs_qst)

    @classmethod
    def release(cls) -> None:
        """Release the current tracing quantum session, allowing a new one to be created."""
        cls.tr_qs_instance = None

    @classmethod
    def get_instance(cls) -> "TracingQuantumSession":
        """Get the current instance of the tracing quantum session."""
        if cls.tr_qs_instance is None:
            raise RuntimeError("No active TracingQuantumSession instance found.")
        return cls.tr_qs_instance


tracing_qs_singleton = TracingQuantumSession()


def check_for_tracing_mode() -> bool:
    """Check if the current execution context is within a JAX tracing session."""
    return hasattr(jax._src.core.trace_ctx.trace, "frame")


def get_last_equation(i: int = -1) -> JaxprEqn:
    """Return the *i*-th equation recorded in the current JAX trace frame.

    Parameters
    ----------
    i:
        Index into the tracing equation list. Defaults to ``-1`` (the most
        recent equation).
    """
    return jax._src.core.trace_ctx.trace.frame.tracing_eqns[i]()
