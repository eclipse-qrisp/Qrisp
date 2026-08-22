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

Static Register Interpreter
============================

Transforms a jaspr into another jaspr that pre-allocates a fixed-size qubit
register and manages qubit create/delete through index bookkeeping, mirroring
the strategy used in the Catalyst interpreter.

State representation inside the context_dic
--------------------------------------------
While the original jaspr uses:

    AbstractQuantumState  -->  a single abstract quantum-state tracer

this interpreter replaces every quantum-state value with a 3-tuple:

    (pre_alloc_array, free_indices, abs_qst)

where
    pre_alloc_array  -- AbstractQubitArray tracer for the single up-front
                        allocation of `size` qubits
    free_indices     -- ScalarList whose elements are qubit *positions*
                        (integers) inside pre_alloc_array that are currently
                        unused
    abs_qst          -- AbstractQuantumState tracer carrying the actual
                        quantum state through jasp primitives

Every AbstractQubitArray value in the context_dic is replaced by a
ScalarList of integer positions into pre_alloc_array.

ScalarList (as opposed to Jlist) never constructs a ranked ``jnp.array``:
it represents its contents as ``max_size`` individual scalar leaves, so that
this interpreter can also be used as input to lowering pipelines (e.g.
CUDA-Q/Quake) whose target dialects have no tensor/linalg equivalent.

Every AbstractQubit value in the context_dic is a plain integer (position
in pre_alloc_array).

Qubit allocation / deallocation
---------------------------------
``jasp.create_qubits``  -- pops N indices from free_indices into a fresh
                           ScalarList that represents the new QubitArray.
``jasp.delete_qubits``  -- pushes the indices back onto free_indices.

Quantum operations
-------------------
Before passing qubits to ``quantum_gate_p`` or ``Measurement_p``, integer
positions are resolved to actual AbstractQubit tracers via
``get_qubit_p.bind(pre_alloc_array, pos)``.

Control flow
-------------
``while``, ``cond``, and ``scan`` primitives are handled by recursively
reinterpreting their body/branch jaxprs with the same evaluator and then
calling the corresponding ``jax.lax`` primitive.  Because ``jax.lax.while_loop``
and ``jax.lax.cond`` support arbitrary JAX pytrees as carry (including
AbstractQuantumState, AbstractQubitArray, and ScalarList), no manual
flatten/unflatten step is required.
"""

import jax.numpy as jnp
from jax import jit, make_jaxpr
from jax.lax import fori_loop
from jax.lax import while_loop as jax_while_loop

from qrisp._cache_config import qrisp_lru_compilation_cache
from qrisp.jasp.primitives import (
    QuantumPrimitive,
    AbstractQuantumState,
    AbstractQubitArray,
    AbstractQubit,
    quantum_gate_p,
    Measurement_p,
    get_qubit_p,
    get_size_p,
    create_qubits_p,
    delete_qubits_p,
    create_quantum_kernel_p,
    consume_quantum_kernel_p,
)
from qrisp.jasp.interpreter_tools.abstract_interpreter import (
    eval_jaxpr,
    extract_invalues,
    insert_call_outvalues,
    insert_outvalues,
    reinterpret,
)
from qrisp.jasp.interpreter_tools.interpreters.traced_control_flow_interpretation import (
    evaluate_cond_under_trace,
    evaluate_scan_under_trace,
    evaluate_while_loop_under_trace,
)
from qrisp.jasp.interpreter_tools.scalar_list import ScalarList


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_static_register_interpreter(size):
    """
    Return an equation evaluator that implements the static-register strategy.

    Parameters
    ----------
    size : int
        Number of qubits to pre-allocate for the whole program.

    Returns
    -------
    callable
        An ``eqn_evaluator`` compatible with ``eval_jaxpr`` / ``reinterpret``.
    """

    def evaluator(eqn, context_dic):
        if isinstance(eqn.primitive, QuantumPrimitive):
            invars = eqn.invars
            outvars = eqn.outvars
            name = eqn.primitive.name

            if name == "jasp.create_quantum_kernel":
                # Pre-allocate all 'size' qubits at once and set up the free-
                # index stack (highest index first, matching catalyst convention).
                abs_qst = create_quantum_kernel_p.bind()
                pre_alloc_array, abs_qst = create_qubits_p.bind(size, abs_qst)
                free_indices = ScalarList(jnp.arange(size)[::-1], max_size=size)
                insert_outvalues(eqn, context_dic, (pre_alloc_array, free_indices, abs_qst))

            elif name == "jasp.consume_quantum_kernel":
                pre_alloc_array, free_indices, abs_qst = extract_invalues(eqn, context_dic)[0]
                abs_qst = delete_qubits_p.bind(pre_alloc_array, abs_qst)
                result = consume_quantum_kernel_p.bind(abs_qst)
                insert_outvalues(eqn, context_dic, result)

            elif name == "jasp.create_qubits":
                _process_create_qubits(invars, outvars, context_dic, size)

            elif name == "jasp.delete_qubits":
                _process_delete_qubits(eqn, context_dic)

            elif name == "jasp.get_qubit":
                _process_get_qubit(invars, outvars, context_dic)

            elif name == "jasp.get_size":
                _process_get_size(invars, outvars, context_dic)

            elif name == "jasp.slice":
                _process_slice(invars, outvars, context_dic)

            elif name == "jasp.fuse":
                _process_fuse(eqn, context_dic, size)

            elif name == "jasp.quantum_gate":
                _process_op(eqn.params["gate"], invars, outvars, context_dic)

            elif name == "jasp.measure":
                _process_measurement(invars, outvars, context_dic)

            elif name == "jasp.parity":
                _process_parity(eqn, context_dic)

            elif name == "jasp.reset":
                _process_reset(eqn, context_dic, evaluator)

            else:
                raise Exception(
                    f"static_register_interpreter: don't know how to process QuantumPrimitive '{eqn.primitive}'"
                )

        else:
            name = eqn.primitive.name
            if name == "while":
                evaluate_while_loop_under_trace(eqn, context_dic, evaluator)
            elif name == "cond":
                evaluate_cond_under_trace(eqn, context_dic, evaluator)
            elif name == "scan":
                evaluate_scan_under_trace(eqn, context_dic, evaluator)
            elif name == "jit":
                _process_pjit(eqn, context_dic, evaluator)
            else:
                return True  # fall back to default binding

    def transform(jaspr):
        """
        Transform a jaspr using the static-register strategy.

        Produces a new jaspr with the *same* input/output signature as the
        original.  Internally, every AbstractQuantumState is expanded to a
        (pre_alloc_array, free_indices, abs_qst) 3-tuple for the body, and
        the pre-allocated register is deleted again before the function
        returns, so callers see no signature change.
        """
        from jax import make_jaxpr
        from jax.extend.core import Literal
        from qrisp.jasp.interpreter_tools.abstract_interpreter import eval_jaxpr as _eval_jaxpr

        # ------------------------------------------------------------------ #
        # Step 1: build the inner (expanded-signature) jaxpr                 #
        # ------------------------------------------------------------------ #
        inner_args = []
        for invar in jaspr.jaxpr.invars:
            if isinstance(invar, Literal):
                if isinstance(invar.val, int):
                    inner_args.append(jnp.asarray(invar.val, dtype="int64"))
                elif isinstance(invar.val, float):
                    inner_args.append(jnp.asarray(invar.val, dtype="float64"))
                else:
                    inner_args.append(invar.val)
            elif isinstance(invar.aval, AbstractQuantumState):
                inner_args.append(
                    (
                        AbstractQubitArray(),
                        ScalarList(jnp.arange(size)[::-1], max_size=size),
                        AbstractQuantumState(),
                    )
                )
            elif isinstance(invar.aval, AbstractQubitArray):
                inner_args.append(ScalarList(max_size=size))
            elif isinstance(invar.aval, AbstractQubit):
                inner_args.append(jnp.asarray(0, dtype="int64"))
            else:
                inner_args.append(invar.aval)

        inner_closed_jaxpr = make_jaxpr(_eval_jaxpr(jaspr, eqn_evaluator=evaluator))(*inner_args)

        # ------------------------------------------------------------------ #
        # Step 2: count QuantumState outvars so we know how many expansions  #
        # to undo in the output                                               #
        # ------------------------------------------------------------------ #
        qst_outvar_count = sum(1 for v in jaspr.jaxpr.outvars if isinstance(v.aval, AbstractQuantumState))

        # If no QuantumState in outputs the inner jaxpr is already fine.
        if qst_outvar_count == 0:
            return inner_closed_jaxpr

        # ------------------------------------------------------------------ #
        # Step 3: build an outer wrapper with the *original* signature       #
        # ------------------------------------------------------------------ #
        outer_args = []
        for invar in jaspr.jaxpr.invars:
            if isinstance(invar, Literal):
                pass
            elif isinstance(invar.aval, AbstractQuantumState):
                outer_args.append(AbstractQuantumState())
            elif isinstance(invar.aval, AbstractQubitArray):
                outer_args.append(AbstractQubitArray())
            elif isinstance(invar.aval, AbstractQubit):
                outer_args.append(jnp.asarray(0, dtype="int64"))
            else:
                outer_args.append(invar.aval)

        # Pre-compute constant initial free-index state (captured in closure).
        # ScalarList has no backing tensor: each free-index slot is its own
        # scalar leaf, so the initial state is `size` individual constants
        # instead of one array + one counter.
        _initial_free_slots = tuple(jnp.asarray(size - 1 - i, dtype=jnp.int64) for i in range(size))
        _initial_free_counter = jnp.array(size, dtype=jnp.int64)

        # Every expanded QuantumState contributes this many flat leaves to
        # the inner jaxpr's signature: QubitArray(1) + ScalarList(size slots
        # + 1 counter) + QuantumState(1).
        leaves_per_qst = size + 3

        def outer_fn(*args):
            # Expand every QuantumState arg to the flat values expected by
            # the inner jaxpr: (pre_alloc_array, *free_slots, free_counter, qst).
            inner_call_args = []
            for invar, arg in zip(jaspr.jaxpr.invars, args):
                if isinstance(invar.aval, AbstractQuantumState):
                    pre_alloc, new_qst = create_qubits_p.bind(size, arg)
                    inner_call_args.append(pre_alloc)
                    inner_call_args.extend(_initial_free_slots)
                    inner_call_args.append(_initial_free_counter)
                    inner_call_args.append(new_qst)
                else:
                    inner_call_args.append(arg)

            # Call the body.
            inner_result = _eval_jaxpr(inner_closed_jaxpr)(*inner_call_args)
            if not isinstance(inner_result, tuple):
                inner_result = (inner_result,)

            # Inner output layout (per expanded QuantumState outvar):
            #   (classical_outs...,
            #    [pre_alloc_array, *scalar_list_slots, scalar_list_counter,
            #     QuantumState] * n)
            n_classical = len(inner_result) - leaves_per_qst * qst_outvar_count
            classical_outs = list(inner_result[:n_classical])

            final_qsts = []
            for i in range(qst_outvar_count):
                base = n_classical + i * leaves_per_qst
                pre_alloc_out = inner_result[base]  # QubitArray
                abs_qst_out = inner_result[base + leaves_per_qst - 1]  # QuantumState
                # Free the whole pre-allocated register before returning.
                cleaned_qst = delete_qubits_p.bind(pre_alloc_out, abs_qst_out)
                final_qsts.append(cleaned_qst)

            result = classical_outs + final_qsts
            return result[0] if len(result) == 1 else tuple(result)

        return make_jaxpr(outer_fn)(*outer_args)

    return transform


@qrisp_lru_compilation_cache(maxsize=int(1e5))
def jaspr_to_static_register_jaspr(jaspr, size):
    """
    Transform a jaspr to use a statically pre-allocated qubit register.

    The returned jaspr is semantically equivalent to the input, but all
    qubit allocations are replaced by index bookkeeping over a single
    register of ``size`` qubits that is created at program start.

    Parameters
    ----------
    jaspr : Jaspr | ClosedJaxpr
        The jaspr to transform.
    size : int
        Total number of qubits to pre-allocate.

    Returns
    -------
    Jaspr | ClosedJaxpr
        A new jaspr of the same type as the input.
    """
    transform = make_static_register_interpreter(size)
    result = transform(jaspr)
    return type(jaspr)(result)


# ---------------------------------------------------------------------------
# Qubit allocation / deallocation
# ---------------------------------------------------------------------------


def _process_create_qubits(invars, outvars, context_dic, register_size):
    """Pop ``n_qubits`` indices from the free pool into a new ScalarList (QubitArray)."""

    pre_alloc_array, free_qubits, abs_qst = context_dic[invars[1]]
    n_qubits = context_dic[invars[0]]

    reg_qubits = ScalarList(max_size=register_size)

    def loop_body(i, val_tuple):
        free_qubits, reg_qubits = val_tuple
        reg_qubits.append(free_qubits.pop())
        return free_qubits, reg_qubits

    # Ensure n_qubits is a JAX tracer so fori_loop can trace through it.
    @jit
    def _make_tracer(x):
        return x

    n_qubits = _make_tracer(n_qubits)

    free_qubits, reg_qubits = fori_loop(0, n_qubits, loop_body, (free_qubits, reg_qubits))

    context_dic[outvars[0]] = reg_qubits
    context_dic[outvars[1]] = (pre_alloc_array, free_qubits, abs_qst)


def _process_delete_qubits(eqn, context_dic):
    """Push all indices of a QubitArray back into the free pool."""

    pre_alloc_array, free_qubits, abs_qst = context_dic[eqn.invars[1]]
    reg_qubits = context_dic[eqn.invars[0]]

    def loop_body(i, val_tuple):
        free_qubits, reg_qubits = val_tuple
        free_qubits.append(reg_qubits.pop())
        return free_qubits, reg_qubits

    free_qubits, reg_qubits = fori_loop(0, reg_qubits.counter, loop_body, (free_qubits, reg_qubits))

    context_dic[eqn.outvars[0]] = (pre_alloc_array, free_qubits, abs_qst)


# ---------------------------------------------------------------------------
# Qubit array indexing / sizing / slicing / fusing
# ---------------------------------------------------------------------------


def _process_get_qubit(invars, outvars, context_dic):
    """Index into the ScalarList to retrieve the integer position of a qubit."""
    qubit_list = context_dic[invars[0]]
    context_dic[outvars[0]] = qubit_list[context_dic[invars[1]]]


def _process_get_size(invars, outvars, context_dic):
    context_dic[outvars[0]] = context_dic[invars[0]].counter


def _process_slice(invars, outvars, context_dic):
    qubit_reg = context_dic[invars[0]]
    start = context_dic[invars[1]]
    stop = context_dic[invars[2]]
    context_dic[outvars[0]] = qubit_reg[start:stop]


def _process_fuse(eqn, context_dic, register_size):
    """Merge two ScalarLists (or a ScalarList and a single qubit position) into one."""
    invalues = extract_invalues(eqn, context_dic)

    if isinstance(eqn.invars[0].aval, AbstractQubit) and isinstance(eqn.invars[1].aval, AbstractQubit):
        res_qubits = ScalarList(invalues, max_size=register_size)
    elif isinstance(eqn.invars[0].aval, AbstractQubitArray) and isinstance(eqn.invars[1].aval, AbstractQubit):
        res_qubits = invalues[0].copy()
        res_qubits.append(invalues[1])
    elif isinstance(eqn.invars[0].aval, AbstractQubit) and isinstance(eqn.invars[1].aval, AbstractQubitArray):
        res_qubits = invalues[1].copy()
        res_qubits.prepend(invalues[0])
    else:
        res_qubits = invalues[0].copy()
        res_qubits.extend(invalues[1])

    insert_outvalues(eqn, context_dic, res_qubits)


# ---------------------------------------------------------------------------
# Gate application
# ---------------------------------------------------------------------------


def _process_op(op, invars, outvars, context_dic):
    """
    Resolve qubit positions to actual AbstractQubit tracers via get_qubit_p,
    then bind quantum_gate_p with those tracers and the original parameters.
    """
    pre_alloc_array, free_indices, abs_qst = context_dic[invars[-1]]

    # Resolve integer positions → AbstractQubit tracers
    qb_tracers = [get_qubit_p.bind(pre_alloc_array, context_dic[invars[i]]) for i in range(op.num_qubits)]

    # Parameter values sit between the qubit invars and the state invar
    n_params = len(invars) - op.num_qubits - 1
    param_values = [context_dic[invars[op.num_qubits + i]] for i in range(n_params)]

    new_abs_qst = quantum_gate_p.bind(*qb_tracers, *param_values, abs_qst, gate=op)

    context_dic[outvars[-1]] = (pre_alloc_array, free_indices, new_abs_qst)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _process_measurement(invars, outvars, context_dic):
    """Measure a single qubit or a QubitArray (returns integer bit-string)."""
    pre_alloc_array, free_indices, abs_qst = context_dic[invars[-1]]

    if isinstance(invars[0].aval, AbstractQubitArray):
        qubit_list = context_dic[invars[0]]
        abs_qst, meas_res = _exec_multi_measurement(pre_alloc_array, qubit_list, abs_qst)
    else:
        qb_pos = context_dic[invars[0]]
        qb_tracer = get_qubit_p.bind(pre_alloc_array, qb_pos)
        meas_res, abs_qst = Measurement_p.bind(qb_tracer, abs_qst)

    context_dic[outvars[0]] = meas_res
    context_dic[outvars[1]] = (pre_alloc_array, free_indices, abs_qst)


def _exec_multi_measurement(pre_alloc_array, qubit_list, abs_qst):
    """
    Measure all qubits in ``qubit_list`` and return their results packed into
    a single int64 (bit 0 = first qubit, etc.).

    Uses jax.lax.while_loop so the loop is captured in the output jaspr.
    """
    list_size = qubit_list.counter

    def loop_body(arg_tuple):
        i, acc, abs_qst = arg_tuple
        qb_index = qubit_list[i]
        qb_tracer = get_qubit_p.bind(pre_alloc_array, qb_index)
        meas_res, abs_qst = Measurement_p.bind(qb_tracer, abs_qst)
        acc = acc + (jnp.asarray(1, dtype="int64") << i) * meas_res
        i = i + jnp.asarray(1, dtype="int64")
        return i, acc, abs_qst

    def cond_body(arg_tuple):
        i, acc, abs_qst = arg_tuple
        return i < list_size

    zero = jnp.asarray(0, dtype="int64")
    _, acc, abs_qst = jax_while_loop(cond_body, loop_body, (zero, zero, abs_qst))

    return abs_qst, acc


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


def _process_parity(eqn, context_dic):
    """Compute XOR of measurement results (same logic as catalyst_interpreter)."""
    invalues = extract_invalues(eqn, context_dic)
    expectation = eqn.params.get("expectation", 0)

    result = 0
    for val in invalues:
        result = result + val
    result = result % 2
    result = (result + expectation) % 2

    insert_outvalues(eqn, context_dic, jnp.array(result, dtype=bool))


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def _process_reset(eqn, context_dic, evaluator):
    """
    Reset a QubitArray to |0> by measuring each qubit and conditionally
    applying X.  Re-uses the reset_jaxpr from catalyst_interpreter but
    evaluates it with *our* evaluator so the static-register representation
    is threaded through correctly.
    """
    # Lazy import to avoid a hard dependency on Catalyst at module load time.
    from qrisp.jasp.interpreter_tools.interpreters.catalyst_interpreter import (
        reset_jaxpr,
    )

    invalues = extract_invalues(eqn, context_dic)
    outvalues = eval_jaxpr(reset_jaxpr.jaxpr, eqn_evaluator=evaluator)(*invalues)
    insert_outvalues(eqn, context_dic, outvalues)


# ---------------------------------------------------------------------------
# Control flow
# ---------------------------------------------------------------------------


def _process_pjit(eqn, context_dic, evaluator):
    """Inline a jit'd call by evaluating its jaxpr with the static-register evaluator."""
    invalues = extract_invalues(eqn, context_dic)
    result = eval_jaxpr(eqn.params["jaxpr"], eqn_evaluator=evaluator)(*invalues)
    insert_call_outvalues(eqn, context_dic, result, len(eqn.params["jaxpr"].jaxpr.outvars))
