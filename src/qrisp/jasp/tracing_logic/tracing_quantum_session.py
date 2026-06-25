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

import weakref
from collections.abc import Sequence
from typing import Any

import jax
from sympy import symbols

from qrisp.core.quantum_variable import QuantumVariable
from qrisp.jasp.primitives import create_qubits, delete_qubits_p, quantum_gate_p
from qrisp.jasp.tracing_logic.dynamic_qubit_array import DynamicQubitArray
from qrisp.typing import ClbitLike, FloatLike

greek_letters = symbols(
    "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    " lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega"
)

_LOST_TRACK_MSG = (
    "Lost track of QuantumState during tracing. This might have been caused by a"
    " missing quantum_kernel decorator or not using quantum prefix control"
    " (like q_fori_loop, q_cond). Please visit"
    " https://www.qrisp.eu/reference/Jasp/Quantum%20Kernel.html for more details"
)


class TracingQuantumSession:
    """Manage tracing-time state for building quantum circuits in Jasp mode.

    A single shared instance is created at module import time (``_instance``)
    and returned by :meth:`get_instance`.

    The session acts as the central recording context while JAX is tracing a
    Python function that constructs a quantum program. In particular, it maintains:

    - a reference to the currently active :class:`~qrisp.jasp.AbstractQuantumState`
      being built (``self.abs_qst``),

    - a cache for qubits allocated during tracing (``self.qubit_cache``),

    - a list of "live" quantum variables currently registered in this session
      (``self.qv_list``).

    The session supports nested tracing. Each call to :meth:`start_tracing`
    pushes the previous tracing state onto internal stacks; :meth:`conclude_tracing`
    restores the previous state and returns the circuit that was traced in the
    nested scope.
    """

    def __init__(self) -> None:
        """Initialise the session state.

        The session starts with no active circuit.
        Use :meth:`start_tracing` to begin recording into a provided abstract
        quantum state object.
        """
        self.abs_qst: Any = None
        self.qv_list: list = []
        self.deleted_qv_list: list = []
        self.qubit_cache: dict | None = None
        self.abs_qst_stack: list = []
        self.qubit_cache_stack: list = []
        self.qv_stack: list = []

    def start_tracing(self, abs_qst: Any) -> None:
        """Push the current tracing state and begin recording into *abs_qst*.

        Parameters
        ----------
        abs_qst : AbstractQuantumState
            The abstract quantum state to record into.
        """
        self.abs_qst_stack.append(self.abs_qst)
        self.qubit_cache_stack.append(self.qubit_cache)

        self.abs_qst = abs_qst
        self.qubit_cache = {}

        self.qv_stack.append((self.qv_list, self.deleted_qv_list))
        self.qv_list = []
        self.deleted_qv_list = []

    def conclude_tracing(self) -> Any:
        """Pop the current tracing state and return the recorded abstract quantum state.

        Returns
        -------
        Any
            The abstract quantum state that was active in the concluded scope.
        """
        temp = self.abs_qst
        self.abs_qst = self.abs_qst_stack.pop()
        self.qubit_cache = self.qubit_cache_stack.pop()
        self.qv_list, self.deleted_qv_list = self.qv_stack.pop()
        return temp

    def append(
        self,
        operation: Any,
        qubits: Any = None,
        clbits: Sequence[ClbitLike] = (),
        param_tracers: Sequence[FloatLike] = (),
    ) -> None:
        """Record a gate application into the current abstract quantum state.

        Dispatches based on the type of the first qubit argument to handle
        :class:`~qrisp.QuantumVariable`, :class:`~qrisp.QuantumArray`,
        :class:`~qrisp.jasp.DynamicQubitArray`, and individual-qubit cases.

        Parameters
        ----------
        operation : Operation
            The gate or operation to apply.
        qubits : sequence or None, optional
            The qubits to apply the operation to. May be a heterogeneous mix of
            :class:`~qrisp.QuantumVariable`, :class:`~qrisp.jasp.DynamicQubitArray`,
            :class:`~qrisp.QuantumArray`, or individual :class:`~qrisp.circuit.Qubit`
            objects. Defaults to ``None`` (no qubits).
        clbits : Sequence[ClbitLike], optional
            Classical bits (must be empty in JAX mode). Defaults to ``()``.
        param_tracers : Sequence[FloatLike], optional
            JAX tracers for the parametric gate arguments (e.g. rotation angles).
            Defaults to ``()``.

        Raises
        ------
        Exception
            If the abstract quantum state has gone out of scope, or if classical
            bits are provided, or if mixed qubit types or incompatible shapes are used.
        """
        if self.abs_qst._trace is not jax.core.trace_ctx.trace:
            raise Exception(_LOST_TRACK_MSG)

        if clbits:
            raise Exception("Tried to append Operation with non-zero classical bits in JAX mode.")

        if qubits is None:
            qubits = ()

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
            for other in qubits[1:]:
                if not isinstance(other, QuantumArray):
                    raise Exception(
                        f"Tried to apply multi-qubit gate to mixed qubit argument"
                        f" types (QuantumArray + {type(other).__name__})"
                    )
                if other.shape != qubits[0].shape:
                    raise Exception("Tried to apply multi-qubit quantum gate to QuantumArrays of differing shape.")

            flattened_qubits = [q.flatten() for q in qubits]
            for i in jrange(flattened_qubits[0].size):
                self.append(
                    operation,
                    [flattened_qubits[j][i] for j in range(operation.num_qubits)],
                    param_tracers=param_tracers,
                )
            return

        self.abs_qst = quantum_gate_p.bind(*qubits, *param_tracers, self.abs_qst, gate=operation)

    def register_qv(self, qv: QuantumVariable, size: Any) -> None:
        """Register a quantum variable in this session and optionally allocate qubits.

        Parameters
        ----------
        qv : QuantumVariable
            The quantum variable to register.
        size : int or None
            Number of qubits to allocate. If ``None``, no allocation is performed
            (the variable already has qubits assigned).
        """
        if self.abs_qst._trace is not jax.core.trace_ctx.trace:
            raise Exception(_LOST_TRACK_MSG)

        if size is not None:
            qv.reg = self.request_qubits(size)

        self.qv_list.append(qv)
        qv.qs = self

        QuantumVariable.live_qvs.append(weakref.ref(qv))
        qv.creation_time = int(QuantumVariable.creation_counter[0])
        QuantumVariable.creation_counter += 1

    def request_qubits(self, amount: Any) -> DynamicQubitArray:
        """Allocate *amount* qubits and return a :class:`~qrisp.jasp.DynamicQubitArray`.

        Parameters
        ----------
        amount : int or JAX tracer
            Number of qubits to allocate.

        Returns
        -------
        DynamicQubitArray
            A dynamic array backed by the newly allocated qubits.
        """
        qb_array_tracer, self.abs_qst = create_qubits(amount, self.abs_qst)
        return DynamicQubitArray(qb_array_tracer)

    def delete_qv(self, qv: QuantumVariable, verify: bool = False) -> None:
        """Remove a quantum variable from this session and free its qubits.

        Parameters
        ----------
        qv : QuantumVariable
            The quantum variable to remove.
        verify : bool, optional
            Verification is not supported in tracing mode. Passing ``True`` raises
            immediately. The default is ``False``.

        Raises
        ------
        Exception
            If *verify* is ``True``, if the abstract quantum state is out of scope,
            or if *qv* is not registered in this session.
        """
        if self.abs_qst._trace is not jax.core.trace_ctx.trace:
            raise Exception(_LOST_TRACK_MSG)

        if verify:
            raise Exception("Tried to verify deletion in tracing mode.")

        try:
            idx = next(i for i, existing_qv in enumerate(self.qv_list) if existing_qv.name == qv.name)
        except StopIteration:
            raise Exception("Tried to remove a non existent quantum variable from quantum session") from None

        self.clear_qubits(qv.reg, verify)
        self.qv_list.pop(idx)
        self.deleted_qv_list.append(qv)

    def clear_qubits(self, qubits: Any, verify: bool = False) -> None:
        """Free *qubits* from the abstract quantum state.

        Parameters
        ----------
        qubits : DynamicQubitArray
            The qubit array to deallocate.
        verify : bool, optional
            Unused in tracing mode; present for interface compatibility. Default is ``False``.
        """
        self.abs_qst = delete_qubits_p.bind(qubits.tracer, self.abs_qst)

    @classmethod
    def release(cls) -> None:
        """No-op retained for backwards compatibility."""

    @classmethod
    def get_instance(cls) -> "TracingQuantumSession":
        """Return the module-level singleton :class:`TracingQuantumSession`.

        The instance is created at import time and is always available.
        """
        return _instance


_instance = TracingQuantumSession()


def check_for_tracing_mode() -> bool:
    """Return ``True`` if JAX is currently in tracing (JIT) mode."""
    return hasattr(jax._src.core.trace_ctx.trace, "frame")


def get_last_equation(i: int = -1) -> Any:
    """Return the *i*-th equation recorded in the current JAX tracing frame.

    Parameters
    ----------
    i : int, optional
        Index into the current frame's equation list. The default is ``-1``
        (the most recently recorded equation).
    """
    return jax._src.core.trace_ctx.trace.frame.tracing_eqns[i]()


def check_live(tracer: Any) -> bool:
    """Return ``True`` if *tracer* is still live in the current JAX tracing context.

    Parameters
    ----------
    tracer : JAX tracer or None
        The tracer to check. ``None`` is treated as always live.
    """
    if tracer is None:
        return True
    return bool(tracer._trace.main.jaxpr_stack)
