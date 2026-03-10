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

"""
Boolean Simulation Interpreter for Classical Function Evaluation
=================================================================

This module provides an interpreter that transforms Jaspr (JAX-based quantum 
program representations) into purely classical JAX expressions. This enables 
efficient simulation of quantum programs that contain only classical/boolean 
logic (X, CX, CCX, SWAP, etc.) without any superposition-creating gates.

Core Concepts:
--------------
1. **Bit Array Representation**: The quantum state is represented as a packed 
   array of uint64 integers, where each integer stores 64 qubit states. This 
   allows efficient bit manipulation using JAX's array operations.

2. **Free Qubit Stack**: A dynamic list (Jlist) tracks which qubit indices are 
   available for allocation. When qubits are created, indices are popped from 
   this stack; when deleted, they are pushed back.

3. **Context Dictionary**: Maps JAX variables to their runtime values during 
   interpretation. For quantum types, this includes:
   - AbstractQuantumState -> (bit_array, free_qubits_jlist)
   - AbstractQubitArray -> Jlist of qubit indices
   - AbstractQubit -> single qubit index (int64)

4. **Supported Gates**: Only gates that preserve classical states are allowed:
   - X (NOT), SWAP: unconditional bit flips/swaps
   - Controlled variants (CX, CCX, CSWAP, etc.): conditional bit flips
   - Phase gates (Z, S, T, RZ, P): no-op (phase has no effect on classical states)
   - Gates creating superposition (H, RX, RY, etc.) raise an error.

The transformation works by:
1. Replacing quantum primitives with classical bit manipulation operations
2. Converting control flow (while, cond) to use the classical reinterpretation
3. Producing a standard ClosedJaxpr that can be JIT-compiled by JAX
"""

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from jax import make_jaxpr, jit, debug
from jax.extend.core import ClosedJaxpr, Literal, Var
from jax.lax import fori_loop, cond, while_loop, switch
import jax.numpy as jnp
from jax import Array

from qrisp.circuit import PTControlledOperation, Operation
from qrisp.jasp import (
    QuantumPrimitive,
    AbstractQuantumState,
    AbstractQubitArray,
    AbstractQubit,
    eval_jaxpr,
    Jaspr,
    extract_invalues,
    insert_outvalues,
    Jlist,
)

from jax.extend.core import JaxprEqn
from qrisp.jasp.interpreter_tools.abstract_interpreter import ContextDict

# Type aliases for clarity
BitArray = Array  # uint64 array where each element stores 64 bits
QubitIndex = Array  # int64 scalar representing a qubit's position
BooleanQuantumState = tuple[BitArray, Jlist]  # (bit_array, free_qubits)


def cl_func_eqn_evaluator(eqn: JaxprEqn, context_dic: ContextDict) -> bool | None:
    """
    Custom equation evaluator for transforming quantum primitives into classical
    bit operations.

    This function is passed to eval_jaxpr as a custom equation handler. It intercepts
    quantum primitives and replaces them with equivalent classical bit manipulations.
    For non-quantum primitives, it returns True to indicate that the default JAX
    evaluation should be used.

    Parameters
    ----------
    eqn : JaxprEqn
        The JAX equation (primitive application) to evaluate.
    context_dic : ContextDict
        Dictionary mapping JAX variables to their runtime values. This is a special
        dictionary subclass that handles Literal keys by returning their values.

    Returns
    -------
    bool | None
        Returns True if the equation should be handled by default JAX evaluation.
        Returns None (implicitly) if the equation was handled by this function.

    Raises
    ------
    Exception
        If an unknown QuantumPrimitive is encountered.
    """
    # If the equation's primitive is a Qrisp primitive, we process it according
    # to one of the implementations below. Otherwise we return True to indicate
    # default interpretation.
    if isinstance(eqn.primitive, QuantumPrimitive):

        invars = eqn.invars
        outvars = eqn.outvars

        if eqn.primitive.name == "jasp.quantum_gate":
            process_op(eqn.params["gate"], invars, outvars, context_dic)
        elif eqn.primitive.name == "jasp.create_qubits":
            process_create_qubits(invars, outvars, context_dic)
        elif eqn.primitive.name == "jasp.get_qubit":
            process_get_qubit(invars, outvars, context_dic)
        elif eqn.primitive.name == "jasp.measure":
            process_measurement(invars, outvars, context_dic)
        elif eqn.primitive.name == "jasp.get_size":
            process_get_size(invars, outvars, context_dic)
        elif eqn.primitive.name == "jasp.slice":
            process_slice(invars, outvars, context_dic)
        elif eqn.primitive.name == "jasp.delete_qubits":
            process_delete_qubits(eqn, context_dic)
        elif eqn.primitive.name == "jasp.reset":
            process_reset(eqn, context_dic)
        elif eqn.primitive.name == "jasp.fuse":
            process_fuse(eqn, context_dic)
        elif eqn.primitive.name == "jasp.parity":
            process_parity(eqn, context_dic)
        else:
            raise Exception(
                f"Don't know how to process QuantumPrimitive {eqn.primitive}"
            )
    else:
        # Handle classical control flow primitives that may contain quantum operations
        if eqn.primitive.name == "while":
            return process_while(eqn, context_dic)
        elif eqn.primitive.name == "cond":
            return process_cond(eqn, context_dic)  #
        elif eqn.primitive.name == "scan":
            return process_scan(eqn, context_dic)
        elif eqn.primitive.name == "jit":
            process_pjit(eqn, context_dic)
        else:
            # Return True to indicate default JAX evaluation should be used
            return True


def process_create_qubits(
    invars: list[Var], outvars: list[Var], context_dic: ContextDict
) -> None:
    """
    Process the create_qubits primitive by allocating qubit indices from the free pool.

    This function handles qubit allocation by:
    1. Popping the requested number of qubit indices from the free_qubits stack
    2. Creating a new Jlist containing these indices to represent the QubitArray
    3. Updating the context dictionary with the new QubitArray and updated circuit state

    Parameters
    ----------
    invars : list[Var]
        Input variables: [size, quantum_circuit]
        - size: Number of qubits to allocate
        - quantum_circuit: The current (bit_array, free_qubits) tuple
    outvars : list[Var]
        Output variables: [qubit_array, updated_quantum_circuit]
    context_dic : ContextDict
        The variable-to-value mapping dictionary.
    """
    # The first invar of the create_qubits primitive is an AbstractQuantumState
    # which is represented by a BitArray and a Jlist of free qubit indices
    qreg, free_qubits = context_dic[invars[1]]

    size = context_dic[invars[0]]

    # Create a new Jlist to hold the allocated qubit indices
    # The max_size is proportional to the total bit array capacity
    reg_qubits = Jlist(
        init_val=jnp.array([], dtype=jnp.int64), max_size=64 * qreg.size // 64
    )

    def loop_body(i: int, val_tuple: tuple[Jlist, Jlist]) -> tuple[Jlist, Jlist]:
        """Pop a qubit index from free pool and add to the new register."""
        reg_qubits, free_qubits = val_tuple
        reg_qubits.append(free_qubits.pop())
        return reg_qubits, free_qubits

    @jit
    def make_tracer(x: Any) -> Any:
        """Ensure value is a JAX tracer for compatibility with fori_loop."""
        return x

    size = make_tracer(size)

    # Allocate 'size' qubits by popping from the free qubit stack
    reg_qubits, free_qubits = fori_loop(0, size, loop_body, (reg_qubits, free_qubits))

    # Store the new QubitArray representation
    context_dic[outvars[0]] = reg_qubits

    # Update the quantum circuit state with the modified free qubit pool
    context_dic[outvars[1]] = (qreg, free_qubits)


def process_fuse(eqn: JaxprEqn, context_dic: ContextDict) -> None:
    """
    Process the fuse primitive which concatenates qubit arrays or individual qubits.

    The fuse operation combines two quantum registers (or individual qubits) into
    a single register. It handles four cases:
    1. Qubit + Qubit -> QubitArray (creates new list with both)
    2. QubitArray + Qubit -> QubitArray (appends qubit to array)
    3. Qubit + QubitArray -> QubitArray (prepends qubit to array)
    4. QubitArray + QubitArray -> QubitArray (extends first with second)

    Parameters
    ----------
    eqn : JaxprEqn
        The fuse equation with two input variables to combine.
    context_dic : ContextDict
        The variable-to-value mapping dictionary.
    """
    invalues = extract_invalues(eqn, context_dic)

    if isinstance(eqn.invars[0].aval, AbstractQubit) and isinstance(
        eqn.invars[1].aval, AbstractQubit
    ):

        # To find the maximum size of the newly created Jlist,
        # we need to search through the context dic for an
        # AbstractQuantumState.
        # This could in principle lead to very bad scaling
        # behavior but in reality, this is not the case.
        # Why? For each stack frame, the arguments of the
        # calling Jaxpr are inserted into the context_dic
        # first, i.e. also at least one AbstractQuantumState.
        # Since they .keys() iterator returns them in the
        # order they were inserted, a fitting entry is quickly
        # found.
        for k in context_dic.keys():
            if isinstance(k.aval, AbstractQuantumState):
                max_size = 64 * context_dic[k][0].size // 64
                break

        # Case 1: Two individual qubits -> create a new list containing both
        res_qubits = Jlist(invalues, max_size=max_size)
    elif isinstance(eqn.invars[0].aval, AbstractQubitArray) and isinstance(
        eqn.invars[1].aval, AbstractQubit
    ):
        # Case 2: Array + Qubit -> append qubit to a copy of the array
        res_qubits = invalues[0].copy()
        res_qubits.append(invalues[1])
    elif isinstance(eqn.invars[0].aval, AbstractQubit) and isinstance(
        eqn.invars[1].aval, AbstractQubitArray
    ):
        # Case 3: Qubit + Array -> prepend qubit to a copy of the array
        res_qubits = invalues[1].copy()
        res_qubits.prepend(invalues[0])
    else:
        # Case 4: Array + Array -> extend first array with second
        res_qubits = invalues[0].copy()
        res_qubits.extend(invalues[1])

    insert_outvalues(eqn, context_dic, res_qubits)


def process_get_qubit(
    invars: list[Var], outvars: list[Var], context_dic: ContextDict
) -> None:
    """
    Process the get_qubit primitive to retrieve a single qubit index from a QubitArray.

    Parameters
    ----------
    invars : list[Var]
        Input variables: [qubit_array, index]
    outvars : list[Var]
        Output variable: [qubit_index]
    context_dic : ContextDict
        The variable-to-value mapping dictionary.
    """
    reg_qubits = context_dic[invars[0]]
    index = context_dic[invars[1]]
    context_dic[outvars[0]] = reg_qubits[index]


def process_slice(
    invars: list[Var], outvars: list[Var], context_dic: ContextDict
) -> None:
    """
    Process the slice primitive to extract a sub-range of qubits from a QubitArray.

    Parameters
    ----------
    invars : list[Var]
        Input variables: [qubit_array, start, stop]
    outvars : list[Var]
        Output variable: [sliced_qubit_array]
    context_dic : ContextDict
        The variable-to-value mapping dictionary.
    """
    reg_qubits = context_dic[invars[0]]
    start = context_dic[invars[1]]
    stop = context_dic[invars[2]]

    context_dic[outvars[0]] = reg_qubits[start:stop]


def process_get_size(
    invars: list[Var], outvars: list[Var], context_dic: ContextDict
) -> None:
    """
    Process the get_size primitive to retrieve the number of qubits in a QubitArray.

    Parameters
    ----------
    invars : list[Var]
        Input variable: [qubit_array]
    outvars : list[Var]
        Output variable: [size]
    context_dic : ContextDict
        The variable-to-value mapping dictionary.
    """
    # The size is stored in the counter field of the Jlist
    context_dic[outvars[0]] = context_dic[invars[0]].counter


def process_op(
    op: Operation, invars: list[Var], outvars: list[Var], context_dic: ContextDict
) -> None:
    """
    Process quantum gate operations by applying classical bit manipulations.

    This function handles gates that preserve classical states:
    - X gate: unconditional bit flip
    - SWAP gate: exchange two bits (implemented as 3 CNOTs)
    - Controlled variants (CX, CCX, CSWAP, etc.): conditional bit flips
    - Phase gates (Z, S, T, RZ, P): no-op since phase doesn't affect classical bits

    Gates that create superposition (H, RX, RY, etc.) will raise an exception.

    Parameters
    ----------
    op : Operation
        The quantum gate operation to process.
    invars : list[Var]
        Input variables: [qubit_0, ..., qubit_n, quantum_circuit]
    outvars : list[Var]
        Output variables: [updated_quantum_circuit]
    context_dic : ContextDict
        The variable-to-value mapping dictionary.

    Raises
    ------
    Exception
        If the gate cannot be simulated classically (e.g., H, RX, RY).
    """
    # Extract the bit array from the quantum circuit tuple
    bit_array = context_dic[invars[-1]][0]

    # Collect the qubit positions (indices in the bit array) for all participating qubits
    qb_pos: list[QubitIndex] = []
    for i in range(op.num_qubits):
        qb_pos.append(context_dic[invars[i]])

    # Determine the control state string for multi-controlled operations
    # Empty string means unconditional (X, SWAP)
    if op.name in ["x", "y", "swap"]:
        ctrl_state = ""
    elif isinstance(op, PTControlledOperation) and op.base_operation.name in [
        "x",
        "y",
        "z",
        "pt2cx",
        "swap",
    ]:
        # Phase gates have no effect on classical states - skip them
        if op.base_operation.name in ["z", "rz", "t", "s", "t_dg", "s_dg", "p"]:
            context_dic[outvars[-1]] = context_dic[invars[-1]]
            return
        ctrl_state = op.ctrl_state
    elif op.name in ["z", "rz", "t", "s", "t_dg", "s_dg", "p"]:
        # Single-qubit phase gates - no effect on classical bits
        context_dic[outvars[-1]] = context_dic[invars[-1]]
        return
    else:
        raise Exception(f"Classical function simulator can't process gate {op.name}")

    # Handle SWAP gates (including controlled SWAP)
    # SWAP is implemented as three CX gates: CX(a,b), CX(b,a), CX(a,b)
    if "swap" in op.name:
        # Swap the last two qubit positions to set up the CX sequence
        swapped_qb_pos = list(qb_pos)
        swapped_qb_pos[-1], swapped_qb_pos[-2] = swapped_qb_pos[-2], swapped_qb_pos[-1]
        ctrl_state = ctrl_state + "1"

        # Apply SWAP as three controlled-NOT operations
        bit_array = cl_multi_cx(bit_array, ctrl_state, swapped_qb_pos)
        bit_array = cl_multi_cx(bit_array, ctrl_state, qb_pos)
        bit_array = cl_multi_cx(bit_array, ctrl_state, swapped_qb_pos)
    else:
        # Standard (multi-)controlled X gate
        bit_array = cl_multi_cx(bit_array, ctrl_state, qb_pos)

    # Update the quantum circuit with the modified bit array
    context_dic[outvars[-1]] = (bit_array, context_dic[invars[-1]][1])


def cl_multi_cx(
    bit_array: BitArray, ctrl_state: str, bit_pos: list[QubitIndex]
) -> BitArray:
    """
    Apply a multi-controlled X (NOT) gate to the bit array.

    This function computes the AND of all control conditions and conditionally
    flips the target bit. The control state string specifies whether each control
    should be active-high ('1') or active-low ('0').

    Parameters
    ----------
    bit_array : BitArray
        The packed bit array representing qubit states.
    ctrl_state : str
        String of '0' and '1' characters specifying the control condition for each
        control qubit. '1' means flip target when control is 1, '0' means flip when 0.
    bit_pos : list[QubitIndex]
        List of qubit indices. The last element is the target qubit; all preceding
        elements are control qubits.

    Returns
    -------
    BitArray
        The updated bit array after applying the gate.

    Examples
    --------
    - ctrl_state="" (empty): Unconditional X gate on target
    - ctrl_state="1": CX gate (flip target if control is 1)
    - ctrl_state="11": CCX/Toffoli gate (flip target if both controls are 1)
    - ctrl_state="01": Flip target if first control is 0 AND second control is 1
    """
    # Start with True (represented as 1) - will be ANDed with all control conditions
    flip = jnp.asarray(1, dtype=bit_array.dtype)

    # Compute the AND of all control conditions
    for i in range(len(bit_pos) - 1):
        if ctrl_state[i] == "1":
            # Active-high control: AND with the bit value
            flip = flip & (get_bit_array(bit_array, bit_pos[i]))
        else:
            # Active-low control: AND with the negated bit value
            flip = flip & (~get_bit_array(bit_array, bit_pos[i]))

    # Conditionally flip the target bit
    bit_array = conditional_bit_flip_bit_array(bit_array, bit_pos[-1], flip)

    return bit_array


def process_measurement(
    invars: list[Var], outvars: list[Var], context_dic: ContextDict
) -> None:
    """
    Process measurement primitives by reading bit values from the bit array.

    Since we're simulating classical states only, measurement simply reads the
    current bit value(s). For a QubitArray, the result is an integer formed by
    treating the qubit values as binary digits.

    Parameters
    ----------
    invars : list[Var]
        Input variables: [qubits_to_measure, quantum_circuit]
        - qubits_to_measure: Either a single qubit or a QubitArray
        - quantum_circuit: The (bit_array, free_qubits) tuple
    outvars : list[Var]
        Output variables: [measurement_result, updated_quantum_circuit]
    context_dic : ContextDict
        The variable-to-value mapping dictionary.
    """
    # Retrieve the bit array from the quantum circuit
    bit_array = context_dic[invars[-1]][0]

    # Handle QubitArray measurement (returns integer) vs single qubit (returns bool)
    if isinstance(invars[0].aval, AbstractQubitArray):
        # Multi-qubit measurement: combine bits into an integer
        qubit_reg = context_dic[invars[0]]
        bit_array, meas_res = exec_multi_measurement(bit_array, qubit_reg)
    else:
        # Single qubit measurement: return the bit value
        meas_res = get_bit_array(bit_array, context_dic[invars[0]])

    # Store results in context dictionary
    context_dic[outvars[1]] = (bit_array, context_dic[invars[1]][1])
    context_dic[outvars[0]] = meas_res


def exec_multi_measurement(
    bit_array: BitArray, qubit_reg: Jlist
) -> tuple[BitArray, Array]:
    """
    Perform measurement on multiple qubits, returning an integer result.

    Each qubit's bit value is read and combined into an integer where qubit i
    contributes to bit position i of the result (little-endian ordering).

    Parameters
    ----------
    bit_array : BitArray
        The packed bit array representing qubit states.
    qubit_reg : Jlist
        A Jlist containing the indices of qubits to measure.

    Returns
    -------
    tuple[BitArray, Array]
        A tuple of (unchanged bit_array, measurement result as int64).
    """

    def loop_body(
        i: int, arg_tuple: tuple[int, BitArray, Jlist]
    ) -> tuple[int, BitArray, Jlist]:
        """Accumulate bit values into an integer."""
        acc, bit_array, qubit_reg = arg_tuple
        qb_index = qubit_reg[i]
        res_bl = get_bit_array(bit_array, qb_index)
        # Shift the bit value to position i and OR into accumulator
        acc = acc + (jnp.asarray(res_bl, dtype="int64") << i)
        return (acc, bit_array, qubit_reg)

    acc, bit_array, qubit_reg = fori_loop(
        0, qubit_reg.counter, loop_body, (0, bit_array, qubit_reg)
    )

    return bit_array, acc


def process_parity(eqn, context_dic):
    """Process parity primitive: compute XOR of measurement results."""
    from jax import debug

    invalues = extract_invalues(eqn, context_dic)
    expectation = eqn.params.get("expectation", 0)
    observable = eqn.params.get("observable", False)

    # Compute parity (XOR) of all measurements
    result = sum(invalues) % 2

    # If not an observable (i.e. detector mode), check and emit warning if mismatch
    if not observable:
        # Define callbacks for match/mismatch
        def warn_mismatch():
            debug.print(
                "WARNING: Parity expectation deviated from simulation result {inval}",
                inval=invalues[1],
            )

        def no_warning():
            return

        # Check if expectation matches result
        cond(result != expectation, warn_mismatch, no_warning)

    # XOR result with expectation (0 if match, 1 if mismatch)
    result = result ^ expectation

    context_dic[eqn.outvars[0]] = jnp.array(result, dtype=bool)


def process_while(eqn: JaxprEqn, context_dic: ContextDict) -> bool | None:
    """
    Process while loop primitives that may contain quantum operations.

    If the while loop involves quantum state (AbstractQuantumState), the body
    and condition Jaxprs are converted to their classical equivalents before
    executing the loop.

    Parameters
    ----------
    eqn : JaxprEqn
        The while loop equation.
    context_dic : ContextDict
        The variable-to-value mapping dictionary.

    Returns
    -------
    bool | None
        Returns True if no quantum types are involved (use default evaluation).
        Returns None if the loop was handled by this function.
    """

    body_jaxpr = eqn.params["body_jaxpr"]
    cond_jaxpr = eqn.params["cond_jaxpr"]
    overall_constant_amount = eqn.params["body_nconsts"] + eqn.params["cond_nconsts"]

    invalues = extract_invalues(eqn, context_dic)

    if isinstance(eqn.invars[-1].aval, AbstractQuantumState):
        bit_array_padding = invalues[-1][0].shape[0] * 64
    else:
        bit_array_padding = 0

    converted_body_jaxpr = jaspr_to_cl_func_jaxpr(body_jaxpr.jaxpr, bit_array_padding)
    converted_cond_jaxpr = jaspr_to_cl_func_jaxpr(cond_jaxpr.jaxpr, bit_array_padding)

    def body_fun(args: list) -> list:
        """Execute the loop body with flattened/unflattened signature conversion."""
        constants = args[eqn.params["cond_nconsts"] : overall_constant_amount]
        carries = args[overall_constant_amount:]

        flattened_invalues = flatten_signature(
            constants + carries, body_jaxpr.jaxpr.invars
        )
        body_res = eval_jaxpr(converted_body_jaxpr)(*(flattened_invalues))
        unflattened_body_outvalues = unflatten_signature(
            body_res, body_jaxpr.jaxpr.outvars
        )

        return list(args[:overall_constant_amount]) + list(unflattened_body_outvalues)

    def cond_fun(args: list) -> Array:
        """Evaluate the loop condition."""
        constants = args[: eqn.params["cond_nconsts"]]
        carries = args[overall_constant_amount:]

        flattened_invalues = flatten_signature(
            constants + carries, cond_jaxpr.jaxpr.invars
        )
        return eval_jaxpr(converted_cond_jaxpr)(*flattened_invalues)

    # Execute the while loop and extract the carry values (skip constants)
    outvalues = while_loop(cond_fun, body_fun, invalues)[overall_constant_amount:]

    insert_outvalues(eqn, context_dic, outvalues)


def process_cond(eqn: JaxprEqn, context_dic: ContextDict) -> None:
    """
    Process conditional (if/else) primitives that may contain quantum operations.

    Converts the branch Jaxprs to their classical equivalents if they contain
    quantum operations, then uses JAX's switch primitive to select the branch.

    Parameters
    ----------
    eqn : JaxprEqn
        The conditional equation with branches.
    context_dic : ContextDict
        The variable-to-value mapping dictionary.
    """
    invalues = extract_invalues(eqn, context_dic)

    # Convert each branch Jaxpr to classical equivalent if needed
    branch_list: list[Callable] = []
    for i in range(len(eqn.params["branches"])):
        converted_jaxpr = ensure_conversion(
            eqn.params["branches"][i].jaxpr, invalues[1:]
        )
        branch_list.append(eval_jaxpr(converted_jaxpr))

    # Flatten signature for JAX's switch, execute, then unflatten results
    flattened_invalues = flatten_signature(invalues, eqn.invars)
    outvalues = switch(flattened_invalues[0], branch_list, *flattened_invalues[1:])

    if isinstance(outvalues, tuple):
        unflattened_outvalues = unflatten_signature(outvalues, eqn.outvars)
    else:
        unflattened_outvalues = (outvalues,)

    insert_outvalues(eqn, context_dic, unflattened_outvalues)


def process_scan(eqn, context_dic):
    """
    Process scan primitive for cl_func_interpreter.
    Reinterprets the scan body and calls jax.lax.scan to preserve loop structure.
    """
    from jax.lax import scan as jax_scan

    invalues = extract_invalues(eqn, context_dic)

    # Extract scan parameters
    num_consts = eqn.params["num_consts"]
    num_carry = eqn.params["num_carry"]
    length = eqn.params["length"]
    reverse = eqn.params.get("reverse", False)
    unroll = eqn.params.get("unroll", 1)

    # Separate inputs: constants, initial carry, scanned inputs
    consts = invalues[:num_consts]
    init = invalues[num_consts : num_consts + num_carry]
    xs = invalues[num_consts + num_carry :]

    # Reinterpret the scan body with cl_func_eqn_evaluator
    scan_body_jaxpr = eqn.params["jaxpr"]
    scan_body = eval_jaxpr(scan_body_jaxpr, eqn_evaluator=cl_func_eqn_evaluator)

    # Create wrapper function that includes constants
    if num_consts > 0:

        def wrapped_body(carry, x):
            args = consts + list(carry) + (list(x) if isinstance(x, tuple) else [x])
            result = scan_body(*args)
            if not isinstance(result, tuple):
                result = (result,)
            return result[:num_carry], result[num_carry:]

    else:

        def wrapped_body(carry, x):
            args = list(carry) + (list(x) if isinstance(x, tuple) else [x])
            result = scan_body(*args)
            if not isinstance(result, tuple):
                result = (result,)
            return result[:num_carry], result[num_carry:]

    # Prepare inputs for JAX scan
    if len(xs) == 1:
        xs_arg = xs[0]
    else:
        xs_arg = tuple(xs)

    if len(init) == 1:
        init_arg = init[0]
    else:
        init_arg = tuple(init)

    # Call JAX scan
    final_carry, ys = jax_scan(
        wrapped_body, init_arg, xs_arg, length=length, reverse=reverse, unroll=unroll
    )

    # Prepare output
    if not isinstance(final_carry, tuple):
        final_carry = (final_carry,)
    if not isinstance(ys, tuple):
        ys = (ys,)

    outvalues = final_carry + ys

    insert_outvalues(eqn, context_dic, outvalues)


@lru_cache(maxsize=int(1e5))
def get_traced_fun(jaxpr: Jaspr | ClosedJaxpr, bit_array_size: int) -> Callable:
    """
    Get a JIT-compiled function from a Jaxpr, with caching for efficiency.

    This function converts a Jaspr to its classical equivalent (if applicable)
    and returns a JIT-compiled evaluator. Results are cached to avoid repeated
    compilation of the same Jaxpr.

    Parameters
    ----------
    jaxpr : Jaspr | ClosedJaxpr
        The Jaxpr to compile.
    bit_array_size : int
        The size of the bit array (number of bits that can be simulated).

    Returns
    -------
    Callable
        A JIT-compiled function that evaluates the Jaxpr.
    """
    from jax.core import eval_jaxpr

    if isinstance(jaxpr, Jaspr):
        cl_func_jaxpr = jaspr_to_cl_func_jaxpr(jaxpr, bit_array_size)
    else:
        cl_func_jaxpr = jaxpr

    @jit
    def jitted_fun(*args):
        return eval_jaxpr(cl_func_jaxpr.jaxpr, cl_func_jaxpr.consts, *args)

    return jitted_fun


def process_pjit(eqn: JaxprEqn, context_dic: ContextDict) -> None:
    """
    Process JIT-compiled function calls (pjit primitive).

    This handles calls to JIT-compiled functions, including special cases for
    Gidney-style MCX gates which are optimized multi-controlled X implementations.

    Parameters
    ----------
    eqn : JaxprEqn
        The pjit equation representing the function call.
    context_dic : ContextDict
        The variable-to-value mapping dictionary.
    """
    invalues = extract_invalues(eqn, context_dic)

    # Special handling for Gidney MCX gates (optimized multi-controlled X)
    if eqn.params["name"] in ["gidney_mcx", "gidney_mcx_inv"]:
        # Gidney MCX is a 2-controlled X gate
        unflattened_outvalues = [
            (cl_multi_cx(invalues[-1][0], "11", invalues[:-1]), invalues[-1][1])
        ]
    elif eqn.params["name"] in ["gidney_CCCX", "gidney_CCCX_dg"]:
        # Gidney CCCX is a 3-controlled X gate
        controls = [invalues[0][i] for i in range(3)]
        unflattened_outvalues = [
            (
                cl_multi_cx(invalues[-1][0], "111", controls + [invalues[1]]),
                invalues[-1][1],
            )
        ]
    else:
        # General case: flatten inputs, call the function, unflatten outputs
        flattened_invalues = []
        for value in invalues:
            if isinstance(value, tuple):
                flattened_invalues.extend(value)
            else:
                flattened_invalues.append(value)

        jaxpr = eqn.params["jaxpr"]

        # Determine bit array padding from the quantum circuit if present
        if isinstance(jaxpr, Jaspr):
            bit_array_padding = invalues[-1][0].shape[0] * 64
        else:
            bit_array_padding = 0

        traced_fun = get_traced_fun(jaxpr, bit_array_padding)

        flattened_invalues = flatten_signature(invalues, eqn.invars)
        outvalues = traced_fun(*flattened_invalues)
        unflattened_outvalues = unflatten_signature(outvalues, eqn.outvars)

    insert_outvalues(eqn, context_dic, unflattened_outvalues)


def process_delete_qubits(eqn: JaxprEqn, context_dic: ContextDict) -> None:
    """
    Process qubit deletion with uncomputation verification.

    This function handles the deletion of qubits by:
    1. Checking that each qubit is in the |0⟩ state (proper uncomputation)
    2. Returning the qubit indices to the free pool
    3. Printing a warning if any qubit is not properly uncomputed

    This verification is valuable for debugging quantum algorithms at scale,
    as faulty uncomputation is a common source of errors.

    Parameters
    ----------
    eqn : JaxprEqn
        The delete_qubits equation.
    context_dic : ContextDict
        The variable-to-value mapping dictionary.
    """
    invalues = extract_invalues(eqn, context_dic)

    bit_array = invalues[-1][0]
    free_qubits = invalues[-1][1]
    qubit_reg = invalues[0]

    def true_fun() -> None:
        """Called when a qubit is not properly uncomputed (still in |1⟩ state)."""
        debug.print("WARNING: Faulty uncomputation found during simulation.")

    def false_fun() -> None:
        """Called when qubit is properly uncomputed (in |0⟩ state)."""
        return

    def loop_body(
        i: int, value_tuple: tuple[Jlist, BitArray, Jlist]
    ) -> tuple[Jlist, BitArray, Jlist]:
        """Check each qubit and return its index to the free pool."""
        qubit_reg, bit_array, free_qubits = value_tuple

        qubit = qubit_reg.pop()
        # Verify the qubit is in |0⟩ state before deletion
        cond(get_bit_array(bit_array, qubit), true_fun, false_fun)
        # Return the qubit index to the free pool
        free_qubits.append(qubit)

        return qubit_reg, bit_array, free_qubits

    qubit_reg, bit_array, free_qubits = fori_loop(
        0, qubit_reg.counter, loop_body, (qubit_reg, bit_array, free_qubits)
    )

    outvalues = (bit_array, free_qubits)
    insert_outvalues(eqn, context_dic, outvalues)


def process_reset(eqn: JaxprEqn, context_dic: ContextDict) -> None:
    """
    Process reset operations that set qubits to the |0⟩ state.

    This function resets qubits by flipping any bits that are currently 1.
    It handles both single qubit resets and QubitArray resets.

    Parameters
    ----------
    eqn : JaxprEqn
        The reset equation.
    context_dic : ContextDict
        The variable-to-value mapping dictionary.
    """
    invalues = extract_invalues(eqn, context_dic)

    bit_array = invalues[-1][0]

    if isinstance(eqn.invars[0].aval, AbstractQubitArray):
        # Reset all qubits in the array
        start = invalues[0][0]
        stop = start + invalues[0][1]

        def loop_body(i: int, bit_array: BitArray) -> BitArray:
            """Reset a single qubit by flipping it if it's 1."""
            bit_array = conditional_bit_flip_bit_array(
                bit_array, i, get_bit_array(bit_array, i)
            )
            return bit_array

        bit_array = fori_loop(start, stop, loop_body, bit_array)
    else:
        # Reset a single qubit
        bit_array = conditional_bit_flip_bit_array(
            bit_array, invalues[0], get_bit_array(bit_array, invalues[0])
        )

    outvalues = (bit_array, invalues[-1][1])
    insert_outvalues(eqn, context_dic, outvalues)


def jaspr_to_cl_func_jaxpr(jaspr: Jaspr, bit_array_padding: int) -> ClosedJaxpr:
    """
    Transform a Jaspr (quantum program) into a classical ClosedJaxpr.

    This is the main transformation function that converts a quantum program
    representation into a purely classical JAX expression. The transformation
    replaces:
    - AbstractQuantumState with (BitArray, Jlist) tuples
    - AbstractQubitArray with Jlist of qubit indices
    - AbstractQubit with int64 qubit indices
    - Quantum primitives with classical bit manipulation operations

    Parameters
    ----------
    jaspr : Jaspr
        The quantum program representation to transform.
    bit_array_padding : int
        The maximum number of qubits that can be simulated. This determines
        the size of the bit array (ceiling of bit_array_padding / 64 uint64s).

    Returns
    -------
    ClosedJaxpr
        A classical JAX expression that performs the equivalent boolean simulation.

    Notes
    -----
    The transformation works by:
    1. Creating dummy input values with appropriate shapes/types
    2. Using make_jaxpr to trace through eval_jaxpr with the custom equation evaluator
    3. The resulting Jaxpr contains only classical operations
    """
    # Create dummy input values for tracing
    args: list[Any] = []
    for invar in jaspr.invars:
        if isinstance(invar.aval, AbstractQuantumState):
            # Quantum circuit -> (bit_array, free_qubits_jlist)
            args.append(
                (
                    jnp.zeros(int(np.ceil(bit_array_padding / 64)), dtype=jnp.uint64),
                    Jlist(jnp.arange(bit_array_padding), max_size=bit_array_padding),
                )
            )
        elif isinstance(invar.aval, AbstractQubitArray):
            # QubitArray -> empty Jlist (will be filled during execution)
            qreg = Jlist(
                init_val=jnp.array([], dtype=jnp.int64),
                max_size=bit_array_padding // 64,
            )
            args.append(qreg)
        elif isinstance(invar.aval, AbstractQubit):
            # Single qubit -> int64 index
            args.append(jnp.asarray(0, dtype="int64"))
        elif isinstance(invar, Literal):
            # Literal values are converted to appropriate JAX arrays
            if isinstance(invar.val, int):
                args.append(jnp.asarray(invar.val, dtype="int64"))
            if isinstance(invar.val, float):
                args.append(jnp.asarray(invar.val, dtype="float64"))
        else:
            # Classical abstract values pass through unchanged
            args.append(invar.aval)

    # Trace the evaluation to produce a classical Jaxpr
    return make_jaxpr(eval_jaxpr(jaspr, eqn_evaluator=cl_func_eqn_evaluator))(*args)


def conditional_bit_flip_bit_array(
    bit_array: BitArray, index: QubitIndex, condition: Array
) -> BitArray:
    """
    Conditionally flip a bit in the packed bit array.

    The bit array stores 64 bits per uint64 element. This function computes
    the array index and bit position within that element, then XORs the bit
    with the condition value.

    Parameters
    ----------
    bit_array : BitArray
        The packed bit array (uint64 elements, each storing 64 bits).
    index : QubitIndex
        The bit index to flip (0-indexed across the entire array).
    condition : Array
        If 1, flip the bit; if 0, leave unchanged.

    Returns
    -------
    BitArray
        The updated bit array.

    Notes
    -----
    Bit packing scheme:
    - array_index = index // 64 (which uint64 element)
    - int_index = index % 64 (which bit within that element)
    - The bit is flipped by XORing with (condition << int_index)
    """
    index = jnp.uint64(index)
    # Compute which uint64 element contains this bit (index >> 6 == index // 64)
    array_index = index >> 6
    # Compute which bit within that element (index ^ (array_index << 6) == index % 64)
    int_index = index ^ (array_index << 6)
    # XOR the bit with the condition shifted to the correct position
    set_value = bit_array[array_index] ^ (condition << int_index)
    return bit_array.at[array_index].set(set_value)


# JIT-compile with donate_argnums=0 to enable in-place mutation of bit_array
conditional_bit_flip_bit_array = jit(conditional_bit_flip_bit_array, donate_argnums=0)


def get_bit_array(bit_array: BitArray, index: QubitIndex) -> Array:
    """
    Get the value of a single bit from the packed bit array.

    Parameters
    ----------
    bit_array : BitArray
        The packed bit array (uint64 elements, each storing 64 bits).
    index : QubitIndex
        The bit index to read (0-indexed across the entire array).

    Returns
    -------
    Array
        The bit value (0 or 1) as a JAX array.
    """
    index = jnp.uint64(index)
    # Compute which uint64 element contains this bit
    array_index = index >> 6
    # Compute which bit within that element
    int_index = index ^ (array_index << 6)
    # Extract the bit by shifting right and masking
    return (bit_array[array_index] >> int_index) & 1


def unflatten_signature(values: tuple, variables: list[Var]) -> list[Any]:
    """
    Convert flattened JAX values back to structured quantum types.

    During JAX tracing, quantum types (AbstractQuantumState, AbstractQubitArray)
    are flattened into multiple array values. This function reconstructs the
    original structure.

    Parameters
    ----------
    values : tuple
        Flattened tuple of JAX array values.
    variables : list[Var]
        List of JAX variables describing the expected types.

    Returns
    -------
    list[Any]
        List of values with quantum types reconstructed:
        - AbstractQuantumState -> (bit_array, Jlist)
        - AbstractQubitArray -> Jlist
        - Other types -> unchanged
    """
    values = list(values)
    unflattened_values: list[Any] = []

    for var in variables:
        if isinstance(var.aval, AbstractQuantumState):
            # Reconstruct (bit_array, free_qubits_jlist) tuple
            bit_array = values.pop(0)
            jlist_tuple = (values.pop(0), values.pop(0))
            unflattened_values.append((bit_array, Jlist.unflatten([], jlist_tuple)))
        elif isinstance(var.aval, AbstractQubitArray):
            # Reconstruct Jlist from (array, counter) tuple
            jlist_tuple = (values.pop(0), values.pop(0))
            unflattened_values.append(Jlist.unflatten([], jlist_tuple))
        else:
            # Classical values pass through unchanged
            unflattened_values.append(values.pop(0))

    return unflattened_values


def flatten_signature(values: list[Any], variables: list[Var]) -> list[Array]:
    """
    Flatten structured quantum types into plain JAX arrays.

    Quantum types need to be flattened for JAX operations like switch and
    while_loop that expect flat argument lists.

    Parameters
    ----------
    values : list[Any]
        List of values that may include quantum types.
    variables : list[Var]
        List of JAX variables describing the types.

    Returns
    -------
    list[Array]
        Flattened list of JAX arrays:
        - AbstractQuantumState (bit_array, Jlist) -> [bit_array, array, counter]
        - AbstractQubitArray Jlist -> [array, counter]
        - Other types -> unchanged
    """
    values = list(values)
    flattened_values: list[Array] = []

    for i in range(len(variables)):
        var = variables[i]
        value = values.pop(0)
        if isinstance(var.aval, AbstractQuantumState):
            # Flatten (bit_array, Jlist) -> [bit_array, jlist.array, jlist.counter]
            flattened_values.extend((value[0], *value[1].flatten()[0]))
        elif isinstance(var.aval, AbstractQubitArray):
            # Flatten Jlist -> [array, counter]
            flattened_values.extend(value.flatten()[0])
        else:
            # Classical values pass through unchanged
            flattened_values.append(value)

    return flattened_values


def ensure_conversion(jaxpr: Jaspr, invalues: list[Any]) -> ClosedJaxpr:
    """
    Ensure a Jaxpr is converted to classical form if it contains quantum types.

    This function checks if the Jaxpr contains any quantum abstract types and
    converts it to a classical Jaxpr if necessary. If no quantum types are
    present, it wraps the Jaxpr in a ClosedJaxpr unchanged.

    Parameters
    ----------
    jaxpr : Jaspr
        The Jaxpr to potentially convert.
    invalues : list[Any]
        The input values, used to determine the bit array size.

    Returns
    -------
    ClosedJaxpr
        Either the converted classical Jaxpr or the original wrapped as ClosedJaxpr.
    """
    bit_array_padding = 0

    # Check if any input variable is a quantum type
    for i in range(len(jaxpr.invars)):
        invar = jaxpr.invars[i]
        if isinstance(
            invar.aval, (AbstractQuantumState, AbstractQubitArray, AbstractQubit)
        ):
            if isinstance(invar.aval, AbstractQuantumState):
                # Get bit array size from the quantum circuit value
                bit_array_padding = invalues[i][0].shape[0] * 64

    return jaspr_to_cl_func_jaxpr(jaxpr, bit_array_padding)
