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

"""Gate application functions (cx, h, mcx, rz, measure, ...) that append operations to circuits."""

from __future__ import annotations

# Private alias, so that star imports of this module do not export it.
from functools import partial as _partial
from typing import TYPE_CHECKING, Any

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
import sympy

import qrisp.circuit.standard_operations as std_ops
from qrisp.circuit import Qubit, convert_to_qb_list
from qrisp.core.quantum_array import QuantumArray
from qrisp.core.quantum_variable import QuantumVariable
from qrisp.jasp import (
    AbstractQubit,
    AbstractQubitArray,
    DynamicQubitArray,
    Measurement_p,
    TracingQuantumSession,
    check_for_tracing_mode,
    jlen,
    jrange,
    q_fori_loop,
    reset_p,
)
from qrisp.misc.utility import bin_rep, find_qs, gate_wrap

# Private alias, so that star imports of this module do not export it.
from qrisp.simulator.unitary_management import u3matrix as _u3matrix

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence, Sized

    from jax.core import Tracer

    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger
    from qrisp.circuit import Clbit, Operation
    from qrisp.qtypes import QuantumBool
    from qrisp.typing import FloatLike, NDArrayLike, QuantumOperand

# The gidney, gidney_inv, amy and jones mcx methods are Toffoli constructions.
_TOFFOLI_NUM_CONTROLS = 2

# From this many controls on, the maslov mcx method uses an ancilla.
_MASLOV_ANCILLA_MIN_CONTROLS = 3

# With method "auto", mcp uses the balauca method from this many qubits on.
_MCP_AUTO_BALAUCA_MIN_QUBITS = 4

# In Jasp mode, QuantumVariables and qubit arrays have a traced size, and a single
# traced qubit has no length.
_UNSIZED_JASP_OPERANDS = (QuantumVariable, DynamicQubitArray, jax.core.Tracer)

_LOST_TRACK_MESSAGE = (
    "Lost track of QuantumCircuit during tracing. This might have been caused by a missing "
    "quantum_kernel decorator or not using quantum prefix control (like q_fori_loop, q_cond). "
    "Please visit https://www.qrisp.eu/reference/Jasp/Quantum%20Kernel.html for more details"
)


def append_operation(
    operation: Operation,
    qubits: Sequence[QuantumOperand] | QuantumVariable | DynamicQubitArray | None = None,
    clbits: Sequence[Clbit | Sequence[Clbit]] | None = None,
    param_tracers: Sequence[FloatLike] | None = None,
) -> None:
    """Append an operation to the QuantumSession of its qubits.

    The QuantumSession is found from ``qubits``. Appending to empty qubit lists, as in
    ``cx([], [])``, does nothing.

    Parameters
    ----------
    operation : Operation
        The operation to append.
    qubits : list or QuantumVariable or DynamicQubitArray, optional
        The qubit arguments of the operation, one per qubit of the operation. The default
        is an empty list.
    clbits : list, optional
        The classical bit arguments of the operation. The default is an empty list.
    param_tracers : list, optional
        In Jasp mode, the traced values of the parameters of ``operation``. The default
        is an empty list.

    """
    qubits = [] if qubits is None else qubits
    clbits = [] if clbits is None else clbits
    param_tracers = [] if param_tracers is None else param_tracers
    try:
        qs = find_qs(qubits)
        qs.append(operation, qubits, clbits, param_tracers=param_tracers)
    except Exception as e:
        # Appending to empty qubit lists, as in cx([], []), does nothing.
        no_session = "Couldn't find QuantumSession" in str(e)
        if no_session and all(isinstance(q, list) and len(q) == 0 for q in qubits):
            return
        raise


# Placeholder parameters of the gates appended in Jasp mode, where the traced values are
# passed separately as param_tracers.
_PARAM_SYMBOLS = (sympy.Symbol("alpha"), sympy.Symbol("beta"), sympy.Symbol("gamma"))


def _append_parametrized_gate(
    make_gate: Callable[..., Operation],
    params: list[FloatLike],
    qubits: list[QuantumOperand],
) -> None:
    """Append a gate with parameters, using placeholder symbols in Jasp mode.

    Parameters
    ----------
    make_gate : callable
        Builds the gate from its parameters.
    params : list[FloatLike]
        The parameters of the gate.
    qubits : list[QuantumOperand]
        The qubit arguments of the gate.

    """
    if check_for_tracing_mode():
        gate = make_gate(*_PARAM_SYMBOLS[: len(params)])
        append_operation(gate, qubits, param_tracers=params)
    else:
        append_operation(make_gate(*params), qubits)


def _ctrl_state_string(ctrl_state: int | str | jax.Array, n: int) -> str:
    """Return a static control state as a string, whose character i is the state of qubit i.

    Parameters
    ----------
    ctrl_state : int or str
        The control state. An integer is read in binary, -1 meaning all qubits in state 1.
        A string is returned unchanged.
    n : int
        The number of control qubits.

    Returns
    -------
    str
        The control state as a string.

    """
    if isinstance(ctrl_state, str):
        return ctrl_state
    if ctrl_state == -1:
        ctrl_state += 2**n
    return bin_rep(ctrl_state, n)[::-1]


def _check_toffoli_controls(name: str, controls: Sized) -> None:
    """Raise an error unless a Toffoli construction gets exactly two controls.

    Parameters
    ----------
    name : str
        The name of the construction, used in the error message.
    controls : list[Qubit]
        The control qubits.

    Raises
    ------
    ValueError
        If there are not exactly two controls.

    """
    if len(controls) != _TOFFOLI_NUM_CONTROLS:
        raise ValueError(f"Tried to call {name} with {len(controls)} controls instead of two")


def cx(control: QuantumOperand, target: QuantumOperand) -> None:
    """Applies a CX gate.

    Parameters
    ----------
    control : Qubit or list[Qubit] or QuantumVariable
        The Qubit to control on.
    target : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the X gate on.

    """
    append_operation(std_ops.CXGate(), [control, target])


def cy(control: QuantumOperand, target: QuantumOperand) -> None:
    """Applies a CY gate.

    Parameters
    ----------
    control : Qubit or list[Qubit] or QuantumVariable
        The Qubit to control on.
    target : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the Y gate on.

    """
    append_operation(std_ops.CYGate(), [control, target])


def cz(control: QuantumOperand, target: QuantumOperand) -> None:
    """Applies a CZ gate.

    Parameters
    ----------
    control : Qubit or list[Qubit] or QuantumVariable
        The Qubit to control on.
    target : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the Z gate on.

    """
    append_operation(std_ops.CZGate(), [control, target])


def h(qubits: QuantumOperand) -> None:
    """Applies an H gate.

    Parameters
    ----------
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the H gate on.

    """
    append_operation(std_ops.HGate(), [qubits])


def x(qubits: QuantumOperand) -> None:
    """Applies an X gate.

    Parameters
    ----------
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the X gate on.

    """
    append_operation(std_ops.XGate(), [qubits])


def y(qubits: QuantumOperand) -> None:
    """Applies a Y gate.

    Parameters
    ----------
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the Y gate on.

    """
    append_operation(std_ops.YGate(), [qubits])


def z(qubits: QuantumOperand) -> None:
    """Applies a Z gate.

    Parameters
    ----------
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the Z gate on.

    """
    append_operation(std_ops.ZGate(), [qubits])


# The mcx methods. Each function applies an mcx gate with one method and takes the controls,
# the target, the control state and the number of ancillas, in this order. In static mode,
# the controls are a list and the control state is a string with one character per control.
# In Jasp mode, the controls can also be a QuantumVariable or a DynamicQubitArray, and the
# control state an integer, a string or a traced integer. The algorithm modules import this
# module, so the functions import them at call time.


def _mcx_gate(
    controls: list[Qubit | Tracer],
    target: Qubit | Tracer,
    ctrl_state: int | str,
    _num_ancilla: int,
    method: str,
) -> None:
    """Append an MCXGate, which implements the gray-code methods and ``auto``."""
    append_operation(std_ops.MCXGate(len(controls), ctrl_state, method=method), controls + [target])


def _gidney_mcx(
    controls: list[Qubit],
    target: Qubit,
    ctrl_state: str,
    _num_ancilla: int,
    inv: bool,
) -> None:
    """Append the Gidney logical AND, or its uncomputation if ``inv``, in static mode."""
    from qrisp.alg_primitives.mcx_algs import GidneyLogicalAND

    _check_toffoli_controls("Gidney logical AND", controls)
    append_operation(GidneyLogicalAND(ctrl_state=ctrl_state, inv=inv), controls + [target])


def _jasp_gidney_mcx(
    controls: list[Tracer],
    target: Tracer,
    _ctrl_state: int | str | jax.Array,
    _num_ancilla: int,
    inv: bool,
) -> None:
    """Apply the Gidney logical AND, or its uncomputation if ``inv``, in Jasp mode.

    The control state is not used.
    """
    from qrisp.alg_primitives.mcx_algs import jasp_gidney_mcx
    from qrisp.environments import invert

    _check_toffoli_controls("Gidney logical AND", controls)
    if inv:
        with invert():
            jasp_gidney_mcx(controls[0], controls[1], target)
    else:
        jasp_gidney_mcx(controls[0], controls[1], target)


def _maslov_mcx(controls: list[Qubit], target: Qubit, ctrl_state: str, _num_ancilla: int) -> None:
    """Append the maslov mcx gate, with an ancilla from three controls on."""
    from qrisp.alg_primitives.mcx_algs import maslov_mcx
    from qrisp.qtypes import QuantumBool

    n = len(controls)
    if n >= _MASLOV_ANCILLA_MIN_CONTROLS:
        ancilla = [QuantumBool(name="maslov_anc_")]
    else:
        ancilla = []
    append_operation(maslov_mcx(n, ctrl_state), controls + ancilla + [target])

    for qv in ancilla:
        qv.delete()


def _balauca_mcx(controls: list[Qubit], target: Qubit, ctrl_state: str, _num_ancilla: int) -> None:
    """Apply the balauca mcx gate in static mode."""
    from qrisp.alg_primitives.mcx_algs import balauca_mcx

    balauca_mcx(controls, [target], ctrl_state=ctrl_state)


def _jasp_balauca_mcx(
    controls: Sequence[Tracer] | QuantumVariable | DynamicQubitArray,
    target: Tracer,
    ctrl_state: int | str | jax.Array,
    _num_ancilla: int,
) -> None:
    """Apply the balauca mcx gate in Jasp mode."""
    from qrisp.alg_primitives.mcx_algs import jasp_balauca_mcx

    jasp_balauca_mcx(controls, [target], ctrl_state)


def _balauca_dirty_mcx(controls, target, ctrl_state, num_ancilla: int) -> None:
    """Apply the balauca mcx gate with ``num_ancilla`` dirty ancillas."""
    from qrisp.alg_primitives.mcx_algs import balauca_dirty

    balauca_dirty(controls, [target], k=num_ancilla, ctrl_state=ctrl_state)


def _yong_mcx(controls, target, ctrl_state, _num_ancilla: int) -> None:
    """Apply the yong mcx gate."""
    from qrisp.alg_primitives.mcx_algs import yong_mcx

    yong_mcx(controls, [target], ctrl_state=ctrl_state)


def _hybrid_mcx(controls, target, ctrl_state, num_ancilla: int) -> None:
    """Apply the hybrid mcx gate with ``num_ancilla`` ancillas."""
    from qrisp.alg_primitives.mcx_algs import hybrid_mcx

    hybrid_mcx(controls, [target], ctrl_state=ctrl_state, num_ancilla=num_ancilla)


def _amy_mcx(controls, target, ctrl_state, _num_ancilla: int) -> None:
    """Apply the Toffoli construction of Amy et al."""
    from qrisp.alg_primitives.mcx_algs import amy_toffoli

    _check_toffoli_controls("Amy MCX", controls)
    amy_toffoli(controls, [target], ctrl_state=ctrl_state)


def _jones_mcx(controls, target, ctrl_state, _num_ancilla: int) -> None:
    """Apply the Toffoli construction of Jones."""
    from qrisp.alg_primitives.mcx_algs import jones_toffoli

    _check_toffoli_controls("Jones MCX", controls)
    jones_toffoli(controls, [target], ctrl_state=ctrl_state)


def _khattar_mcx(controls, target, ctrl_state, _num_ancilla: int) -> None:
    """Apply the khattar mcx gate."""
    from qrisp.alg_primitives.mcx_algs import khattar_mcx

    khattar_mcx(controls, [target], ctrl_state)


# The methods that are called in the same way in both modes.
_SHARED_MCX_METHODS: dict[str, Callable[..., None]] = {
    "gray": _partial(_mcx_gate, method="gray"),
    "gray_pt": _partial(_mcx_gate, method="gray_pt"),
    "gray_pt_inv": _partial(_mcx_gate, method="gray_pt_inv"),
    "auto": _partial(_mcx_gate, method="auto"),
    "balauca_dirty": _balauca_dirty_mcx,
    "yong": _yong_mcx,
    "hybrid": _hybrid_mcx,
    "amy": _amy_mcx,
    "jones": _jones_mcx,
    "khattar": _khattar_mcx,
}

_STATIC_MCX_METHODS: dict[str, Callable[..., None]] = {
    **_SHARED_MCX_METHODS,
    "gidney": _partial(_gidney_mcx, inv=False),
    "gidney_inv": _partial(_gidney_mcx, inv=True),
    "maslov": _maslov_mcx,
    "balauca": _balauca_mcx,
}

_JASP_MCX_METHODS: dict[str, Callable[..., None]] = {
    **_SHARED_MCX_METHODS,
    "gidney": _partial(_jasp_gidney_mcx, inv=False),
    "gidney_inv": _partial(_jasp_gidney_mcx, inv=True),
    "balauca": _jasp_balauca_mcx,
}


def _mcx_method(methods: dict[str, Callable[..., None]], method: str) -> Callable[..., None]:
    """Return the function of an mcx method.

    Parameters
    ----------
    methods : dict[str, callable]
        The available methods, ``_STATIC_MCX_METHODS`` or ``_JASP_MCX_METHODS``.
    method : str
        The name of the method.

    Returns
    -------
    callable
        The function applying the mcx gate with this method.

    Raises
    ------
    ValueError
        If the method is unknown.

    """
    synthesize = methods.get(method) if isinstance(method, str) else None
    if synthesize is None:
        raise ValueError(f"Don't know mcx method {method}")
    return synthesize


def _mcx_operands(
    controls: Sequence[Qubit | Tracer | QuantumBool] | QuantumVariable | DynamicQubitArray,
    target: Qubit | Tracer | QuantumBool,
) -> tuple[
    Sequence[Qubit | Tracer | QuantumBool] | QuantumVariable | DynamicQubitArray,
    Qubit | Tracer,
]:
    """Return the controls and target of mcx, with QuantumBools replaced by their qubit.

    Parameters
    ----------
    controls : list[Qubit or QuantumBool] or QuantumVariable or DynamicQubitArray
        The controls.
    target : Qubit or QuantumBool
        The target.

    Returns
    -------
    tuple
        The controls and the target qubit.

    Raises
    ------
    TypeError
        If the target is a list or a QuantumVariable other than a QuantumBool.

    """
    # qrisp.qtypes imports this module, so it can only be imported at call time.
    from qrisp.qtypes import QuantumBool

    if isinstance(controls, list):
        controls = [qbl[0] if isinstance(qbl, QuantumBool) else qbl for qbl in controls]

    if isinstance(target, (list, QuantumVariable)):
        if isinstance(target, QuantumBool):
            return controls, target[0]
        raise TypeError("mcx target is not of type Qubit or QuantumBool")

    return controls, target


def _static_mcx(
    controls: Sequence[Qubit | Tracer | QuantumBool] | QuantumVariable | DynamicQubitArray,
    target: Qubit | Tracer,
    method: str,
    ctrl_state: int | str | jax.Array,
    num_ancilla: int,
) -> None:
    """Apply an mcx gate in static mode.

    Without controls, nothing is applied. With one control, a CX gate on the control
    state is applied, whatever the method.

    Parameters
    ----------
    controls : list[Qubit] or QuantumVariable
        The controls.
    target : Qubit
        The target.
    method : str
        The synthesis method.
    ctrl_state : int or str
        The control state.
    num_ancilla : int
        The number of ancillas of the ``hybrid`` and ``balauca_dirty`` methods.

    Raises
    ------
    ValueError
        If the control state does not have one character per control, or the method
        is unknown.

    """
    controls = list(controls)
    n = len(controls)

    if n == 0:
        return
    if n == 1:
        append_operation(std_ops.MCXGate(n, ctrl_state, method="gray"), controls + [target])
        return

    ctrl_state = _ctrl_state_string(ctrl_state, n)
    if len(ctrl_state) != n:
        msg = f"Given control state {ctrl_state} does not match control qubit amount {n}"
        raise ValueError(msg)

    _mcx_method(_STATIC_MCX_METHODS, method)(controls, target, ctrl_state, num_ancilla)


def _jasp_mcx(
    controls: Sequence[Qubit | Tracer | QuantumBool] | QuantumVariable | DynamicQubitArray,
    target: Qubit | Tracer,
    method: str,
    ctrl_state: int | str | jax.Array,
    num_ancilla: int,
) -> None:
    """Apply an mcx gate in Jasp mode.

    Controls given as a QuantumVariable or a DynamicQubitArray use the ``balauca`` method,
    unless the method is ``khattar``.

    Parameters
    ----------
    controls : list[Qubit] or QuantumVariable or DynamicQubitArray
        The controls.
    target : Qubit
        The target.
    method : str
        The synthesis method.
    ctrl_state : int or str or jax.Array
        The control state.
    num_ancilla : int
        The number of ancillas of the ``hybrid`` and ``balauca_dirty`` methods.

    Raises
    ------
    ValueError
        If the method is unknown or not available in Jasp mode.

    """
    qubit_arrays = (QuantumVariable, DynamicQubitArray)
    if isinstance(controls, qubit_arrays) and method not in ("balauca", "khattar"):
        method = "balauca"

    if method == "maslov":
        raise ValueError("The maslov method of mcx is not available in Jasp mode")

    _mcx_method(_JASP_MCX_METHODS, method)(controls, target, ctrl_state, num_ancilla)


def mcx(
    controls: Sequence[Qubit | Tracer | QuantumBool] | QuantumVariable | DynamicQubitArray,
    target: Qubit | Tracer | QuantumBool,
    method: str = "auto",
    ctrl_state: int | str | jax.Array = -1,
    num_ancilla: int = 1,
) -> None:
    r""".. _mcx:

    Applies a multi-controlled X gate.

    The following methods are available:


    .. list-table::
        :widths: 20 80
        :header-rows: 1

        *   - Method
            - Description
        *   - ``gray``
            - Performs a gray code traversal which requires no ancillae but is rather
              inefficient for large numbers of control qubits.
        *   - ``gray_pt``/``gray_pt_inv``
            - More efficient but introduce extra phases that need to be uncomputed by
              performing the inverse of this gate on the same inputs. For more information
              on phase tolerance, check `this paper
              <https://iopscience.iop.org/article/10.1088/2058-9565/acaf9d/meta>`__.
        *   - ``balauca``
            - Method based on this `paper
              <https://www.iccs-meeting.org/archive/iccs2022/papers/133530169.pdf>`__ with
              logarithmic depth but requires many ancilla qubits.
        *   - ``balauca_dirty``
            - A variant of ``balauca`` based on dirty ancilla qubits, whose number is set
              by ``num_ancilla``.
        *   - ``khattar``
            - Method based on this `paper <https://arxiv.org/abs/2407.17966>`_, implements
              the n-controlled x with 2n − 3 Toffoli and linear depth using 1 clean
              ancilla. The case n=3 implements this `paper
              <https://arxiv.org/abs/2106.11513>`__.
        *   - ``maslov``
            - Documented `here <https://arxiv.org/abs/1508.03273>`_, requires less ancilla
              qubits but is only available for 4 or less control qubits.
        *   - ``yong``
            - Can be found in this `article
              <https://link.springer.com/article/10.1007/s10773-017-3389-4>`__. This method
              requires only a single ancilla and has moderate scaling in depth and gate
              count.
        *   - ``amy``
            - A Toffoli-circuit (i.e. only two control qubits are possible), which
              (temporarily) requires one ancilla qubit. However, instead of the no-ancilla
              T-depth 4, this circuit achieves a T-depth of 2. Find the implementation
              details in `this paper <https://arxiv.org/pdf/1206.0758.pdf>`__.
        *   - ``jones``
            - Similar to ``amy`` but uses two ancilla qubits, and has a T-depth of 1. Read
              about it `here <https://arxiv.org/abs/1212.5069>`__.
        *   - ``gidney``
            - A very unique way for synthesizing a logical AND. The Gidney Logical AND
              performs a circuit with T-depth 1 to compute the truth value and performs
              another circuit involving a measurement and a classically controlled CZ gate
              for uncomputation. The uncomputation circuit has T-depth 0, such that the
              combined T-depth is 1. Requires no ancillae. More details `here
              <https://arxiv.org/abs/1709.06648>`__. Works only for two control qubits.
        *   - ``gidney_inv``
            - The uncomputation of ``gidney``, see the examples below.
        *   - ``hybrid``
            - A flexible method which combines the other available methods, such that the
              amount of used ancillae is customizable. After several ``balauca``-layers,
              the recursion is canceled by either a ``yong``, ``maslov`` or ``gray`` mcx,
              depending on what fits the most.
        *   - ``auto``
            - Recompiles the mcx gate at compile time using the hybrid algorithm together
              with the information about how many clean/dirty ancillae qubits are
              available. For more information check :meth:`qrisp.QuantumSession.compile`.

    .. note::
        Due to Qrisp's automatic qubit management, clean ancilla qubits are not as much
        of a sparse resource as one might think. Even though the ``balauca`` method
        requires a considerable amount of ancillae, many other functions also do,
        implying there is a lot of recycling potential. The net effect is that in more
        complex programs, the amount of qubits of the circuit returned by the
        :meth:`compile method <qrisp.QuantumSession.compile>` increases only slightly.

    Parameters
    ----------
    controls : list[Qubit] or QuantumVariable
        The Qubits to control on.
    target : Qubit or QuantumBool
        The Qubit to perform the X gate on.
    method : str, optional
        The synthesis method, one of the methods in the table above. The default is
        ``auto``.
    ctrl_state : int or str, optional
        The state on which to activate the X gate. The default is "1111..".
    num_ancilla : int, optional
        Specifies the amount of ancilla qubits to use. This parameter is used only by
        the ``hybrid`` and ``balauca_dirty`` methods. The default is 1.

    Examples
    --------
    We apply a 3-controlled X gate:

    >>> from qrisp import QuantumBool, QuantumVariable, mcx
    >>> control = QuantumVariable(3)
    >>> target = QuantumBool()
    >>> mcx(control, target, method="gray")
    >>> print(control.qs)  # doctest: +SKIP
    QuantumCircuit:
    ---------------
    control.0: ──■──
                 │
    control.1: ──■──
                 │
    control.2: ──■──
               ┌─┴─┐
     target.0: ┤ X ├
               └───┘
    Live QuantumVariables:
    ----------------------
    QuantumVariable control
    QuantumBool target

    We compare different performance indicators:

    >>> def benchmark_mcx(n, methods):
    ...     for method in methods:
    ...         controls = QuantumVariable(n)
    ...         target = QuantumBool()
    ...         mcx(controls, target, method=method)
    ...         qc = controls.qs.compile()
    ...         depth, cnots, qubits = qc.depth(), qc.cnot_count(), qc.num_qubits()
    ...         print(f"{method:8} depth: {depth:4}  CNOT count: {cnots:4}  qubits: {qubits}")
    >>> benchmark_mcx(4, methods=["gray", "gray_pt", "maslov", "balauca", "yong"])
    gray     depth:   38  CNOT count:   30  qubits: 5
    gray_pt  depth:   38  CNOT count:   30  qubits: 5
    maslov   depth:   44  CNOT count:   18  qubits: 6
    balauca  depth:   26  CNOT count:   18  qubits: 7
    yong     depth:   91  CNOT count:   60  qubits: 6
    >>> benchmark_mcx(10, methods=["gray", "gray_pt", "balauca", "yong"])
    gray     depth: 4069  CNOT count: 2046  qubits: 11
    gray_pt  depth: 4069  CNOT count: 2046  qubits: 11
    balauca  depth:   50  CNOT count:   56  qubits: 18
    yong     depth:  746  CNOT count:  528  qubits: 12

    **Mid circuit measurement based methods**

    The ``gidney`` and ``jones`` method are unique in the way that they require
    mid circuit measurements. The measurements are inserted retroactively by the
    :meth:`.compile <qrisp.QuantumSession.compile>` method, because immediate compilation
    would prevent evaluation of the statevector (since a measurement is involved).

    Instead a tentative (measurement free) representative is inserted and replaced
    at compile time.

    To get a better understanding consider the following script:

    >>> control = QuantumVariable(2)
    >>> target = QuantumBool()
    >>> mcx(control, target, method="jones")
    >>> print(control.qs)  # doctest: +SKIP
    QuantumCircuit:
    ---------------
                        ┌───────────────────────────┐
             control.0: ┤0                          ├
                        │                           │
             control.1: ┤1                          ├
                        │                           │
              target.0: ┤2 uncompiled_jones_toffoli ├
                        │                           │
    jones_ancilla_40.0: ┤3                          ├
                        │                           │
    jones_ancilla_40.1: ┤4                          ├
                        └───────────────────────────┘
    Live QuantumVariables:
    ----------------------
    QuantumVariable control
    QuantumBool target

    We see that there is no classical bit and therefore also no measurement.
    The statevector can still be accessed:

    >>> print(control.qs.statevector())
    |00>*|False>

    To introduce the measurement we simply call the
    :meth:`.compile <qrisp.QuantumSession.compile>` method with the keyword
    ``compile_mcm = True``. The compiled circuit has a classical bit for the measurement:

    >>> qc = control.qs.compile(compile_mcm=True)
    >>> len(qc.clbits)
    1

    Because there is a measurement now, the statevector can no longer be accessed.

    >>> qc.statevector_array()
    Traceback (most recent call last):
    ...
    Exception: Tried to determine the statevector of a circuit containing a measurement

    However the T-depth went down by 50%:

    >>> print(qc.t_depth())
    1
    >>> print(control.qs.compile(compile_mcm=False).t_depth())
    2

    A similar construction holds for the
    `Gidney's temporary logical AND <https://arxiv.org/abs/1709.06648>`_.
    However there are additional details: This technique always comes in pairs. A computation
    and an uncomputation. The computation circuit has a T-depth of 1 and the uncomputation
    circuit has a T-depth of 0. The uncomputation circuit however contains a measurement.

    As you can imagine, this measurement is also inserted at compile time.

    Even though both circuits are not the inverses of each other, Qrisp will use
    the respective partner if called to invert:

    >>> control = QuantumVariable(2)
    >>> target = QuantumBool()
    >>> mcx(control, target, method="gidney")
    >>> print(control.qs)  # doctest: +SKIP
    QuantumCircuit:
    ---------------
               ┌────────────────────────┐
    control.0: ┤0                       ├
               │                        │
    control.1: ┤1 uncompiled_gidney_mcx ├
               │                        │
     target.0: ┤2                       ├
               └────────────────────────┘
    Live QuantumVariables:
    ----------------------
    QuantumVariable control
    QuantumBool target

    This even works in conjunction with the :ref:`uncomputation module <Uncomputation>`:

    >>> target.uncompute()
    >>> print(target.qs)  # doctest: +SKIP
    QuantumCircuit:
    ---------------
               ┌────────────────────────┐┌────────────────────────────┐
    control.0: ┤0                       ├┤0                           ├
               │                        ││                            │
    control.1: ┤1 uncompiled_gidney_mcx ├┤1 uncompiled_gidney_mcx_inv ├
               │                        ││                            │
     target.0: ┤2                       ├┤2                           ├
               └────────────────────────┘└────────────────────────────┘
    Live QuantumVariables:
    ----------------------
    QuantumVariable control

    The compile method inserts the measurement of the uncomputation. Here the pair
    surrounds a CX gate, since a computation directly followed by its uncomputation
    is removed at compile time:

    >>> from qrisp import cx
    >>> control = QuantumVariable(2)
    >>> target = QuantumBool()
    >>> result = QuantumBool()
    >>> mcx(control, target, method="gidney")
    >>> cx(target, result)
    >>> target.uncompute()
    >>> qc = result.qs.compile(compile_mcm=True)
    >>> [instr.op.name for instr in qc.data if instr.op.name.startswith("compiled")]
    ['compiled_gidney_mcx', 'compiled_gidney_mcx_inv']
    >>> len(qc.clbits)
    1

    Apart from uncomputation, the inverted Gidney mcx can also be accessed via
    the :ref:`InversionEnvironment`:

    >>> from qrisp import invert
    >>> control = QuantumVariable(2)
    >>> target = QuantumBool()
    >>> with invert():
    ...     mcx(control, target, method="gidney")
    >>> print(control.qs)  # doctest: +SKIP
    QuantumCircuit:
    ---------------
               ┌────────────────────────────┐
    control.0: ┤0                           ├
               │                            │
    control.1: ┤1 uncompiled_gidney_mcx_inv ├
               │                            │
     target.0: ┤2                           ├
               └────────────────────────────┘
    Live QuantumVariables:
    ----------------------
    QuantumVariable control
    QuantumBool target

    """
    controls, target = _mcx_operands(controls, target)
    if check_for_tracing_mode():
        _jasp_mcx(controls, target, method, ctrl_state, num_ancilla)
    else:
        _static_mcx(controls, target, method, ctrl_state, num_ancilla)


@gate_wrap(permeability="full", is_qfree=True, name="anc supported mcz")
def _mcz_inner(qubits, method: str, ctrl_state: str) -> None:
    """Apply a static multi-controlled Z gate as an mcx gate conjugated by H gates.

    Parameters
    ----------
    qubits : list[Qubit] or QuantumVariable
        The qubits of the gate. The last one is the target of the mcx gate.
    method : str
        The mcx method.
    ctrl_state : str
        The control state, with one character per qubit.

    Raises
    ------
    ValueError
        If the control state does not have one character per qubit.

    """
    n = jlen(qubits)
    if len(ctrl_state) != n:
        msg = f"Given control state {ctrl_state} does not matchcontrol qubit amount {n}"
        raise ValueError(msg)

    if ctrl_state[-1] == "0":
        x(qubits[-1])

    h(qubits[-1])
    mcx(qubits[:-1], qubits[-1], method=method, ctrl_state=ctrl_state[:-1])
    h(qubits[-1])

    if ctrl_state[-1] == "0":
        x(qubits[-1])


@gate_wrap(permeability="full", is_qfree=True, name="anc supported mcz")
def _jasp_mcz_inner(qubits, method: str, ctrl_state: int | jax.Array) -> None:
    """Apply a multi-controlled Z gate in Jasp mode as an mcx gate conjugated by H gates.

    Parameters
    ----------
    qubits : QuantumVariable or DynamicQubitArray or list[Qubit]
        The qubits of the gate. The last one is the target of the mcx gate.
    method : str
        The mcx method.
    ctrl_state : int or jax.Array
        The control state, -1 meaning all qubits in state 1.

    """
    # qrisp.environments imports this module, so it can only be imported at call time.
    from qrisp.environments import control

    n = jlen(qubits)
    ctrl_state = jnp.int64(ctrl_state)
    ctrl_state = jax.lax.cond(ctrl_state == -1, lambda x: x + (1 << n), lambda x: x, ctrl_state)

    with control((ctrl_state >> (n - 1)) & 1 == 0):
        x(qubits[-1])

    h(qubits[-1])
    mcx(
        qubits[:-1],
        qubits[-1],
        method=method,
        ctrl_state=ctrl_state & ((1 << (n - 1)) - 1),
    )
    h(qubits[-1])

    with control((ctrl_state >> (n - 1)) & 1 == 0):
        x(qubits[-1])


def _static_mcz(
    qubits: Sequence[Qubit | Tracer] | QuantumVariable | DynamicQubitArray,
    n: int,
    method: str,
    ctrl_state: int | str | jax.Array,
) -> None:
    """Apply a multi-controlled Z gate in static mode.

    The ``gray`` and ``auto`` methods append a controlled Z gate, the other methods
    conjugate an mcx gate with H gates.

    Parameters
    ----------
    qubits : list[Qubit] or QuantumVariable
        The qubits of the gate.
    n : int
        The number of qubits.
    method : str
        The synthesis method.
    ctrl_state : int or str
        The control state.

    """
    ctrl_state = _ctrl_state_string(ctrl_state, n)

    if method in ("gray", "auto"):
        if ctrl_state[-1] == "0":
            x(qubits[-1])

        if len(qubits) > 1:
            append_operation(
                std_ops.ZGate().control(len(qubits) - 1, method=method, ctrl_state=ctrl_state[:-1]),
                qubits,
            )
        else:
            z(qubits[0])

        if ctrl_state[-1] == "0":
            x(qubits[-1])

        return

    _mcz_inner(qubits, method, ctrl_state)


def _jasp_mcz(
    qubits: Sequence[Qubit | Tracer] | QuantumVariable | DynamicQubitArray,
    method: str,
    ctrl_state: int | str | jax.Array,
) -> None:
    """Apply a multi-controlled Z gate in Jasp mode.

    The ``auto`` method uses ``balauca``, and a string control state is converted to an
    integer.

    Parameters
    ----------
    qubits : list[Qubit] or QuantumVariable or DynamicQubitArray
        The qubits of the gate.
    method : str
        The synthesis method.
    ctrl_state : int or str or jax.Array
        The control state.

    """
    if method == "auto":
        method = "balauca"
    if isinstance(ctrl_state, str):
        ctrl_state = int(ctrl_state, 2)
    _jasp_mcz_inner(qubits, method=method, ctrl_state=ctrl_state)


def mcz(
    qubits: Sequence[Qubit | Tracer] | QuantumVariable | DynamicQubitArray,
    method: str = "auto",
    ctrl_state: int | str | jax.Array = -1,
    num_ancilla: int = 1,
) -> None:
    """Applies a multi-controlled Z gate.

    For more information on the available methods, check
    :func:`the mcx documentation page <qrisp.mcx>`.

    Parameters
    ----------
    qubits : QuantumVariable or list[Qubit]
        The Qubits to control on.
    method : str, optional
        The synthesis method. Available are ``auto``, ``gray``, ``gray_pt``,
        ``gray_pt_inv``, ``maslov``, ``balauca``, ``yong``, ``hybrid`` and ``khattar``.
        The default is ``auto``.
    ctrl_state : int or str, optional
        The state on which to activate the Z gate. The default is "1111...".
    num_ancilla : int, optional
        Specifies the amount of ancilla qubits to use. This parameter is used only if
        the method is set to ``hybrid``. The default is 1.

    """
    n = jlen(qubits)
    if check_for_tracing_mode():
        _jasp_mcz(qubits, method, ctrl_state)
    else:
        _static_mcz(qubits, n, method, ctrl_state)


@gate_wrap(permeability="full", is_qfree=True, name="anc supported mcp")
def _balauca_mcp(phi: FloatLike, qubits, ctrl_state: str) -> None:
    """Apply a static multi-controlled phase gate with the balauca method.

    Parameters
    ----------
    phi : FloatLike
        The phase to apply.
    qubits : list[Qubit] or QuantumVariable
        The qubits of the gate.
    ctrl_state : str
        The control state, with one character per qubit.

    """
    # These packages import this module, so they can only be imported at call time.
    from qrisp.alg_primitives.mcx_algs import hybrid_mcx
    from qrisp.qtypes import QuantumBool

    qubits = convert_to_qb_list(qubits)

    temp = QuantumBool()
    hybrid_mcx(
        qubits,
        temp[0],
        ctrl_state=ctrl_state,
        phase=phi,
        num_ancilla=np.inf,
        use_mcm=True,
    )

    temp.delete()


def _static_mcp(
    phi: FloatLike,
    qubits: Sequence[Qubit | Tracer] | QuantumVariable | DynamicQubitArray,
    n: int,
    method: str,
    ctrl_state: int | str | jax.Array,
) -> None:
    """Apply a multi-controlled phase gate in static mode.

    Parameters
    ----------
    phi : FloatLike
        The phase to apply.
    qubits : list[Qubit] or QuantumVariable
        The qubits of the gate.
    n : int
        The number of qubits.
    method : str
        The synthesis method.
    ctrl_state : int or str
        The control state.

    Raises
    ------
    ValueError
        If the method is unknown.

    """
    # These packages import this module, so they can only be imported at call time.
    from qrisp.alg_primitives.mcx_algs import khattar_mcp
    from qrisp.environments import control

    ctrl_state = _ctrl_state_string(ctrl_state, n)

    if method in ("gray", "gray_pt"):
        if ctrl_state[-1] == "0":
            x(qubits[-1])

        mcp_gate = std_ops.PGate(phi).control(n - 1, ctrl_state=ctrl_state[:-1], method=method)
        append_operation(mcp_gate, qubits)

        if ctrl_state[-1] == "0":
            x(qubits[-1])

    elif method == "balauca":
        _balauca_mcp(phi, qubits, ctrl_state=ctrl_state)

    elif method == "khattar":
        khattar_mcp(phi, qubits, ctrl_state=ctrl_state)

    elif method == "auto":
        with control(n < _MCP_AUTO_BALAUCA_MIN_QUBITS):
            mcp(phi, qubits, method="gray", ctrl_state=ctrl_state)
        with control(n >= _MCP_AUTO_BALAUCA_MIN_QUBITS):
            mcp(phi, qubits, method="balauca", ctrl_state=ctrl_state)
    else:
        raise ValueError(f"Don't know method {method}")


def _jasp_mcp(
    phi: FloatLike,
    qubits: Sequence[Qubit | Tracer] | QuantumVariable | DynamicQubitArray,
    n: int | jax.Array,
    method: str,
    ctrl_state: int | str | jax.Array,
) -> None:
    """Apply a multi-controlled phase gate in Jasp mode.

    The ``auto`` method uses ``balauca``, and a string control state is converted to an
    integer.

    Parameters
    ----------
    phi : FloatLike
        The phase to apply.
    qubits : list[Qubit] or QuantumVariable or DynamicQubitArray
        The qubits of the gate.
    n : int or jax.Array
        The number of qubits.
    method : str
        The synthesis method.
    ctrl_state : int or str or jax.Array
        The control state.

    Raises
    ------
    ValueError
        If the method is unknown.

    """
    # These packages import this module, so they can only be imported at call time.
    from qrisp.alg_primitives.mcx_algs import jasp_balauca_mcp, khattar_mcp

    if method == "auto":
        method = "balauca"
    if isinstance(ctrl_state, str):
        ctrl_state = int(ctrl_state, 2)

    if method in ("gray", "gray_pt"):
        if ctrl_state[-1] == "0":
            x(qubits[-1])

        p_gate = std_ops.PGate(_PARAM_SYMBOLS[0])
        mcp_gate = p_gate.control(n - 1, ctrl_state=ctrl_state[:-1], method=method)
        append_operation(mcp_gate, qubits, param_tracers=[phi])

        if ctrl_state[-1] == "0":
            x(qubits[-1])

    elif method == "balauca":
        jasp_balauca_mcp(phi, qubits, ctrl_state)

    elif method == "khattar":
        khattar_mcp(phi, qubits, ctrl_state=ctrl_state)

    else:
        raise ValueError(f"Don't know method {method}")


def mcp(
    phi: FloatLike,
    qubits: Sequence[Qubit | Tracer] | QuantumVariable | DynamicQubitArray,
    method: str = "auto",
    ctrl_state: int | str | jax.Array = -1,
) -> None:
    """Applies a multi-controlled phase gate.

    The available methods are:

    * ``gray``, which performs a traversal of the gray code.

    * ``gray_pt``, a phase-tolerant version of ``gray``, see the
      :func:`mcx documentation <qrisp.mcx>`.

    * ``balauca``, which is a modified version of the algorithm presented `here
      <https://www.iccs-meeting.org/archive/iccs2022/papers/133530169.pdf>`_.

    * ``khattar``, which is based on this `paper <https://arxiv.org/abs/2407.17966>`_.

    * ``auto``, which picks ``gray`` for any qubit count less than 4 and ``balauca``
      otherwise.

    Parameters
    ----------
    phi : float or sympy.Symbol
        The phase to apply.
    qubits : list[Qubit] or QuantumVariable
        The qubits to apply the multi-controlled phase gate on.
    method : str, optional
        The method to deploy. The default is "auto".
    ctrl_state : str or int, optional
        The control state on which to apply the phase. The default is "111...".

    """
    n = jlen(qubits)
    if check_for_tracing_mode():
        _jasp_mcp(phi, qubits, n, method, ctrl_state)
    else:
        _static_mcp(phi, qubits, n, method, ctrl_state)


def p(phi: FloatLike, qubits: QuantumOperand) -> None:
    """Applies a phase gate.

    Parameters
    ----------
    phi : float or sympy.Symbol
        The phase to apply.
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit on which to apply the phase gate.

    """
    _append_parametrized_gate(std_ops.PGate, [phi], [qubits])


def cp(phi: FloatLike, qubits_0: QuantumOperand, qubits_1: QuantumOperand) -> None:
    """Applies a controlled phase gate.

    Parameters
    ----------
    phi : float or sympy.Symbol
        The phase to apply.
    qubits_0 : Qubit or list[Qubit] or QuantumVariable
        The first Qubit.
    qubits_1 : Qubit or list[Qubit] or QuantumVariable
        The second Qubit.

    """
    _append_parametrized_gate(std_ops.CPGate, [phi], [qubits_0, qubits_1])


def rx(phi: FloatLike, qubits: QuantumOperand) -> None:
    """Applies an RX gate.

    Parameters
    ----------
    phi : float or sympy.Symbol
        The angle parameter.
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the RX gate on.

    """
    _append_parametrized_gate(std_ops.RXGate, [phi], [qubits])


def ry(phi: FloatLike, qubits: QuantumOperand) -> None:
    """Applies an RY gate.

    Parameters
    ----------
    phi : float or sympy.Symbol
        The angle parameter.
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the RY gate on.

    """
    _append_parametrized_gate(std_ops.RYGate, [phi], [qubits])


def rz(phi: FloatLike, qubits: QuantumOperand) -> None:
    """Applies an RZ gate.

    Parameters
    ----------
    phi : float or sympy.Symbol
        The angle parameter.
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the RZ gate on.

    """
    _append_parametrized_gate(std_ops.RZGate, [phi], [qubits])


def crz(phi: FloatLike, qubits_0: QuantumOperand, qubits_1: QuantumOperand) -> None:
    """Applies a controlled RZ gate.

    Parameters
    ----------
    phi : float or sympy.Symbol
        The angle parameter.
    qubits_0 : Qubit or list[Qubit] or QuantumVariable
        The first Qubit.
    qubits_1 : Qubit or list[Qubit] or QuantumVariable
        The second Qubit.

    """
    _append_parametrized_gate(
        lambda angle: std_ops.RZGate(angle).control(1),
        [phi],
        [qubits_0, qubits_1],
    )


def s(qubits: QuantumOperand) -> None:
    """Applies an S gate.

    Parameters
    ----------
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the S gate on.

    """
    append_operation(std_ops.SGate(), [qubits])


def t(qubits: QuantumOperand) -> None:
    """Applies a T gate.

    Parameters
    ----------
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the T gate on.

    """
    append_operation(std_ops.TGate(), [qubits])


def s_dg(qubits: QuantumOperand) -> None:
    """Applies a daggered S gate.

    Parameters
    ----------
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the daggered S gate on.

    """
    append_operation(std_ops.SGate().inverse(), [qubits])


def t_dg(qubits: QuantumOperand) -> None:
    """Applies a daggered T gate.

    Parameters
    ----------
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the daggered T gate on.

    """
    append_operation(std_ops.TGate().inverse(), [qubits])


def sx(qubits: QuantumOperand) -> None:
    """Applies an SX gate.

    Parameters
    ----------
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the SX gate on.

    """
    append_operation(std_ops.SXGate(), [qubits])


def sx_dg(qubits: QuantumOperand) -> None:
    """Applies a daggered SX gate.

    Parameters
    ----------
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the daggered SX gate on.

    """
    append_operation(std_ops.SXDGGate().inverse(), [qubits])


def gphase(phi: FloatLike, qubits: QuantumOperand) -> None:
    """Applies a global phase. This gate turns into a phase gate when controlled.

    Parameters
    ----------
    phi : float or sympy.Symbol
        The global phase to apply.
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the global phase gate on.

    """
    _append_parametrized_gate(std_ops.GPhaseGate, [phi], [qubits])


def xxyy(
    phi: FloatLike,
    beta: FloatLike,
    qubits_0: QuantumOperand,
    qubits_1: QuantumOperand,
) -> None:
    """Applies an XXYY interaction gate.

    Parameters
    ----------
    phi : float or sympy.Symbol
        The global phase to apply.
    beta : float or sympy.Symbol
        The other angle parameter.
    qubits_0 : Qubit or list[Qubit] or QuantumVariable
        The first Qubit to perform the XXYY gate on.
    qubits_1 : Qubit or list[Qubit] or QuantumVariable
        The second Qubit to perform the XXYY gate on.

    """
    _append_parametrized_gate(std_ops.XXYYGate, [phi, beta], [qubits_0, qubits_1])


def rzz(phi: FloatLike, qubits_0: QuantumOperand, qubits_1: QuantumOperand) -> None:
    """Applies an RZZ gate.

    Parameters
    ----------
    phi : float or sympy.Symbol
        The phase to apply.
    qubits_0 : Qubit or list[Qubit] or QuantumVariable
        The first argument to perform the RZZ gate on.
    qubits_1 : Qubit or list[Qubit] or QuantumVariable
        The second argument to perform the RZZ gate on.

    """
    _append_parametrized_gate(std_ops.RZZGate, [phi], [qubits_0, qubits_1])


def rxx(phi: FloatLike, qubits_0: QuantumOperand, qubits_1: QuantumOperand) -> None:
    """Applies an RXX gate.

    Parameters
    ----------
    phi : float or sympy.Symbol
        The phase to apply.
    qubits_0 : Qubit or list[Qubit] or QuantumVariable
        The first argument to perform the RXX gate on.
    qubits_1 : Qubit or list[Qubit] or QuantumVariable
        The second argument to perform the RXX gate on.

    """
    _append_parametrized_gate(std_ops.RXXGate, [phi], [qubits_0, qubits_1])


def ryy(phi: FloatLike, qubits_0: QuantumOperand, qubits_1: QuantumOperand) -> None:
    """Applies an RYY gate.

    Parameters
    ----------
    phi : float or sympy.Symbol
        The phase to apply.
    qubits_0 : Qubit or list[Qubit] or QuantumVariable
        The first argument to perform the RYY gate on.
    qubits_1 : Qubit or list[Qubit] or QuantumVariable
        The second argument to perform the RYY gate on.

    """
    _append_parametrized_gate(std_ops.RYYGate, [phi], [qubits_0, qubits_1])


def u3(
    theta: FloatLike,
    phi: FloatLike,
    lam: FloatLike,
    qubits: QuantumOperand,
) -> None:
    """Applies a U3 gate.

    Parameters
    ----------
    theta : float or sympy.Symbol
        The first angle parameter.
    phi : float or sympy.Symbol
        The second angle parameter.
    lam : float or sympy.Symbol
        The third angle parameter.
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to perform the U3 gate on.

    """
    _append_parametrized_gate(std_ops.U3Gate, [theta, phi, lam], [qubits])


def _u3_matrix_jax(theta: jax.Array, phi: jax.Array, lam: jax.Array) -> jax.Array:
    """Return the matrix of a U3 gate as a JAX array, so that the angles can be traced.

    Parameters
    ----------
    theta : jax.Array
        The first angle of the U3 gate.
    phi : jax.Array
        The second angle of the U3 gate.
    lam : jax.Array
        The third angle of the U3 gate.

    Returns
    -------
    jax.Array
        The 2x2 matrix, with the same entries as ``qrisp.simulator.unitary_management.u3matrix``.

    """
    cos = jnp.cos(theta / 2)
    sin = jnp.sin(theta / 2)
    return jnp.array(
        [
            [cos, -jnp.exp(1j * lam) * sin],
            [jnp.exp(1j * phi) * sin, jnp.exp(1j * (phi + lam)) * cos],
        ]
    )


def unitary(unitary_array: NDArrayLike, qubits: Qubit | Tracer) -> None:
    """Applies a single-qubit gate given by its unitary matrix.

    The matrix is decomposed into a U3 gate and a global phase.

    Parameters
    ----------
    unitary_array : numpy.ndarray
        The 2x2 unitary matrix to apply.
    qubits : Qubit
        The Qubit to apply the gate on.

    Examples
    --------
    >>> import numpy as np
    >>> from qrisp import QuantumVariable, unitary
    >>> qv = QuantumVariable(1)
    >>> unitary(np.array([[0, 1], [1, 0]], dtype=complex), qv[0])
    >>> print(qv)
    {'1': 1.0}

    """
    mat = unitary_array
    coeff = 1 / jnp.sqrt(jnp.linalg.det(mat))
    tmp_10 = jnp.abs((coeff * mat[1][0]))
    tmp_00 = jnp.abs((coeff * mat[0][0]))
    theta = 2 * jnp.arctan2(tmp_10, tmp_00)
    phiplambda2 = jnp.angle(coeff * mat[1][1]) % (2 * jnp.pi)
    phimlambda2 = jnp.angle(coeff * mat[1][0]) % (2 * jnp.pi)
    phi = phiplambda2 + phimlambda2
    lam = phiplambda2 - phimlambda2

    if check_for_tracing_mode():
        # The angles are traced, so the U3 matrix is built with jax.numpy, and the
        # matrix is indexed with a traced index.
        flat_mat = jnp.ravel(mat)
        temp_u3 = _u3_matrix_jax(theta, phi, lam).ravel()
    else:
        flat_mat = mat.flatten()
        temp_u3 = _u3matrix(theta, phi, lam, 0).flatten()
    arg_max = jnp.argmax(jnp.abs(flat_mat))

    gphase_angle = (-jnp.angle(temp_u3[arg_max] / flat_mat[arg_max])) % (2 * jnp.pi)

    u3(theta, phi, lam, qubits)
    gphase(gphase_angle, qubits)


def measure(qubits: QuantumOperand) -> Any:
    """Measure qubits in the computational basis.

    In static mode, the measurement is appended to the circuit and the classical
    bits holding the outcome are returned. In :ref:`Jasp <jasp>` mode, the outcome
    itself is returned as a traced value.

    Parameters
    ----------
    qubits : Qubit or QuantumVariable or QuantumArray or DynamicQubitArray or list[Qubit]
        The qubits to measure. In Jasp mode, a list of qubits is not accepted:
        measure a QuantumVariable or a slice of it instead.

    Returns
    -------
    Clbit or list[Clbit] or jax.Array
        In static mode, a :ref:`Clbit` for a single qubit, and otherwise a list
        with one Clbit per qubit. In Jasp mode, a boolean for a
        qubit, an integer for an array of qubits, and the decoded value for a
        :ref:`QuantumVariable` or a :ref:`QuantumArray`.

    Raises
    ------
    TypeError
        In Jasp mode, if ``qubits`` has a type that cannot be measured.
    RuntimeError
        In Jasp mode, if the quantum state was lost, for example because
        ``measure`` was called inside ``jax.lax.fori_loop`` instead of
        :func:`qrisp.jasp.q_fori_loop`.

    See Also
    --------
    QuantumVariable.get_measurement : Simulate a QuantumVariable and return its measurement results.

    Examples
    --------
    >>> from qrisp import QuantumFloat, QuantumVariable, measure, x
    >>> from qrisp.jasp import boolean_simulation
    >>> qv = QuantumVariable(2)
    >>> clbits = measure(qv)
    >>> len(clbits), qv.qs.data[-1].op.name
    (2, 'measure')

    In Jasp mode, the outcome is a traced value:

    >>> @boolean_simulation
    ... def main():
    ...     a = QuantumFloat(3)
    ...     a[:] = 5
    ...     x(a[1])
    ...     return measure(a), measure(a[0])
    >>> value, first_qubit = main()
    >>> float(value), bool(first_qubit)
    (7.0, True)

    """
    qs = find_qs(qubits)

    if not isinstance(qs, TracingQuantumSession):
        # One new Clbit per qubit, or a single Clbit for a single qubit. The qubits of
        # a QuantumArray are counted with convert_to_qb_list, like append_operation does.
        if isinstance(qubits, QuantumArray):
            clbits = [qs.add_clbit() for _ in convert_to_qb_list(qubits)]
        elif hasattr(qubits, "__len__"):
            clbits = [qs.add_clbit() for _ in qubits]
        else:
            clbits = qs.add_clbit()
        append_operation(std_ops.Measurement(), [qubits], [clbits])
        return clbits

    # The quantum state belongs to the trace of the enclosing Jasp function. Inside a
    # JAX control-flow primitive, such as jax.lax.fori_loop, the trace is different.
    if qs.abs_qst._trace is not jax.core.trace_ctx.trace:
        raise RuntimeError(_LOST_TRACK_MESSAGE)

    if isinstance(qubits, (DynamicQubitArray, QuantumVariable, QuantumArray)):
        return qubits.measure()
    qubit_avals = (AbstractQubitArray, AbstractQubit)
    if isinstance(qubits, jax.core.Tracer) and isinstance(qubits.aval, qubit_avals):
        result, qs.abs_qst = Measurement_p.bind(qubits, qs.abs_qst)
        return result

    raise TypeError(f"Tried to measure type {type(qubits)}")


def measure_to_big_integer(qv: QuantumVariable, size: int) -> BigInteger:
    """Measure a QuantumVariable in Jasp mode and return the outcome as a BigInteger.

    The qubits are measured in groups of 32, each giving one 32-bit digit of the
    result, so the variable can have more qubits than a 64-bit integer can hold.

    Parameters
    ----------
    qv : QuantumVariable
        The variable to measure.
    size : int
        The number of 32-bit digits of the returned BigInteger.

    Returns
    -------
    BigInteger
        The measured value.

    """
    # qrisp.alg_primitives imports this module, so BigInteger can only be imported at
    # call time.
    from qrisp import BigInteger

    def body_fun(i: int | jax.Array, val: jax.Array) -> jax.Array:
        return val.at[i].set(jnp.uint32(measure(qv[32 * i : 32 * (i + 1)])))

    last_digit = (qv.size - 1) // 32
    digits = q_fori_loop(0, last_digit, body_fun, jnp.zeros(size, jnp.uint32))
    digits = digits.at[last_digit].set(jnp.uint32(measure(qv[32 * last_digit :])))
    return BigInteger(digits)


def reset(qubits: QuantumOperand) -> None:
    """Performs a reset of the specified qubits.

    Parameters
    ----------
    qubits : Qubit or list[Qubit] or QuantumVariable or QuantumArray
        The qubits to reset.

    """
    qs = find_qs(qubits)

    if not isinstance(qs, TracingQuantumSession):
        append_operation(std_ops.Reset(), [qubits])
        return

    if isinstance(qubits, QuantumVariable):
        abs_qst = reset_p.bind(qubits.reg.tracer, qs.abs_qst)
    elif isinstance(qubits, QuantumArray):
        flattened_qa = qubits.flatten()
        for i in jrange(qubits.size):
            reset(flattened_qa[i])
        return
    elif isinstance(qubits.aval, AbstractQubitArray):
        abs_qst = reset_p.bind(qubits.tracer, qs.abs_qst)
    elif isinstance(qubits.aval, AbstractQubit):
        abs_qst = reset_p.bind(qubits, qs.abs_qst)
    else:
        raise TypeError(f"Tried to reset type {type(qubits.aval)}")

    qs.abs_qst = abs_qst


def barrier(qubits: QuantumOperand) -> None:
    """A visual marker for structuring the QuantumCircuit.

    In :ref:`Jasp <jasp>` mode, a barrier on a list of qubits or on a QuantumArray is
    applied as in static mode. A barrier on a QuantumVariable, whose size is traced, or
    on a single qubit does nothing.

    Parameters
    ----------
    qubits : Qubit or list[Qubit] or QuantumVariable
        The Qubit to apply the barrier to.

    Examples
    --------
    >>> from qrisp import QuantumVariable, x, y, barrier
    >>> qv = QuantumVariable(5)
    >>> x(qv)
    >>> barrier(qv)
    >>> y(qv)
    >>> print(qv.qs)  # doctest: +SKIP
    QuantumCircuit:
    ---------------
          ┌───┐ ░ ┌───┐
    qv.0: ┤ X ├─░─┤ Y ├
          ├───┤ ░ ├───┤
    qv.1: ┤ X ├─░─┤ Y ├
          ├───┤ ░ ├───┤
    qv.2: ┤ X ├─░─┤ Y ├
          ├───┤ ░ ├───┤
    qv.3: ┤ X ├─░─┤ Y ├
          ├───┤ ░ ├───┤
    qv.4: ┤ X ├─░─┤ Y ├
          └───┘ ░ └───┘
    Live QuantumVariables:
    ----------------------
    QuantumVariable qv

    """
    if isinstance(qubits, Qubit):
        qubits = [qubits]
    elif check_for_tracing_mode() and isinstance(qubits, _UNSIZED_JASP_OPERANDS):
        # These have no static number of qubits, so no Barrier operation can be built.
        return

    append_operation(std_ops.Barrier(len(qubits)), qubits)


def swap(qubits_0: QuantumOperand, qubits_1: QuantumOperand) -> None:
    """Applies a SWAP gate.

    Parameters
    ----------
    qubits_0 : Qubit or list[Qubit] or QuantumVariable
        The first Qubit.
    qubits_1 : Qubit or list[Qubit] or QuantumVariable
        The second Qubit.

    """
    append_operation(std_ops.SwapGate(), [qubits_0, qubits_1])
