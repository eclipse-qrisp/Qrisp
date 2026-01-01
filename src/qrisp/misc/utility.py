"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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

import functools
import traceback

import jax.numpy as jnp
import numpy as np
import sympy

# A small epsilon value for numerical stability.
# Defined here for convenience, so it can be imported elsewhere.
_EPSILON = jnp.sqrt(jnp.finfo(jnp.float64).eps)


def swap_endianness(vec: np.ndarray, n: int) -> np.ndarray:
    """
    Convert between big-endian and little-endian qubit ordering.

    This transformation is its own inverse, so it works in both directions.

    Parameters
    ----------
    vec : np.ndarray
        The state vector to convert.
    n : int
        The number of qubits.

    Returns
    -------
    np.ndarray
        The state vector with reversed qubit ordering.
    """
    return vec.reshape([2] * n).transpose(*reversed(range(n))).flatten()


def bin_rep(n, bits):
    if n < 0:
        raise Exception("Only positive numbers are supported")

    if n >= 2**bits:
        raise Exception(
            str(n) + " can't be represented as a " + str(bits) + " bit number"
        )

    return bin(n)[2:].zfill(bits)
    zero_string = "".join(["0" for k in range(bits)])
    return (zero_string + bin(n)[2:])[-bits:]


def int_encoder(qv, encoding_number):

    from qrisp import control, x
    from qrisp.jasp import TracingQuantumSession, check_for_tracing_mode, jrange

    if not check_for_tracing_mode():
        if encoding_number > 2 ** len(qv) - 1:
            raise Exception(
                "Not enough qubits to encode integer " + str(encoding_number)
            )

        for i in range(len(qv)):
            if (1 << i) & encoding_number:
                x(qv[i])

    else:

        from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import (
            BigInteger,
        )

        if isinstance(encoding_number, BigInteger):
            for i in jrange(qv.size):
                with control(encoding_number.get_bit(i)):
                    x(qv[i])
        else:
            for i in jrange(qv.size):
                with control(encoding_number & (1 << i)):
                    x(qv[i])

        # def true_fun(qc, cond, qb):
        #     tr_qs.abs_qc = qc
        #     x(qb)
        #     return tr_qs.abs_qc

        # def false_fun(qc, cond, qb):
        #     return qc

        # def loop_fun(i, qc):
        #     cond_bool = (1<<i) & encoding_number
        #     qb = qv[i]
        #     qc = cond(cond_bool, true_fun, false_fun, qc, cond_bool, qb)

        #     return qc
        # qc = fori_loop(0, qv.size, loop_fun, (tr_qs.abs_qc))
        # tr_qs.abs_qc = qc


# Calculates the binary expression of a given integer and returns it as an array of
# length bits
def int_as_array(k, bit):
    bin_str = bin_rep(k, bit)
    return np.array([int(c) for c in bin_str])


def array_as_int(array):
    result = 0
    for k in range(len(array)):
        if array[::-1][k]:
            result += 2 ** (k)

    return result


# Decomposes the circuit qc until no more decompositions are possible and then counts
# the cnot operations
def cnot_count(qc):
    qc = qc.transpile()

    gate_count_dic = qc.count_ops()
    cnot_count = 0
    for gate_name in ["cx", "cy", "cz"]:
        cnot_count += gate_count_dic.get(gate_name, 0)

    return cnot_count


def is_inv(x, bit):
    # return (math.gcd(int(np.round(x, 3)),2**bit) == 1)

    # The only divisors 2**bit has is powers of 2
    # ie. if tha factorization of x doesn't contain any powers of 2 it is invertible
    # in other words: x is invertible if it is uneven
    return bool(int(x) % 2)


def get_depth_dic(qc, transpile_qc=True, depth_indicator=lambda x: 1):
    if len(qc.qubits) == 0:
        return {}

    if transpile_qc:
        qc = qc.transpile()

    # Assign each bit in the circuit a unique integer
    # to index into op_stack.
    bit_indices = {bit: idx for idx, bit in enumerate(qc.qubits + qc.clbits)}

    # If no bits, return 0
    if not bit_indices:
        return 0

    # A list that holds the height of each qubit
    # and classical bit.
    op_stack = [0] * len(bit_indices)

    # Here we are playing a modified version of
    # Tetris where we stack gates, but multi-qubit
    # gates, or measurements have a block for each
    # qubit or cbit that are connected by a virtual
    # line so that they all stacked at the same depth.
    # Conditional gates act on all cbits in the register
    # they are conditioned on.
    # We treat barriers or snapshots different as
    # They are transpiler and simulator directives.
    # The max stack height is the circuit depth.

    for instr in qc.data:
        if instr.op.name in ["qb_alloc", "qb_dealloc", "gphase"]:
            continue
        qargs = instr.qubits
        cargs = instr.clbits

        levels = []
        reg_ints = []
        # If count then add one to stack heights

        gate_depth = depth_indicator(instr.op)

        for ind, reg in enumerate(qargs + cargs):
            # Add to the stacks of the qubits and
            # cbits used in the gate.
            reg_ints.append(bit_indices[reg])
            levels.append(op_stack[reg_ints[ind]] + gate_depth)

        max_level = max(levels)
        for ind in reg_ints:
            op_stack[ind] = max_level

    return {qc.qubits[i]: op_stack[i] for i in range(len(qc.qubits))}


def gate_wrap(*args, permeability=None, is_qfree=None, name=None, verify=False):
    """
    Decorator to bundle up the quantum instructions of a function into a single gate
    object. Bundled gate objects can help debugging as it allows for a more clear
    QuantumCircuit visualisation.

    Furthermore, bundling up functions is relevant for Qrisps uncomputation algorithm.
    When bundling up for uncomputation, this decorator provides the means to annotate
    the gate objects with information about its permeability and qfree-ness. For further
    information about these concepts check the
    :ref:`uncomputation documentation<uncomputation>`. Specifying this information
    allows to skip the computationally costly automatic determination at runtime.

    Note that the specified information is not checked for correctness as this would
    defy the purpose.

    A shorthand for ``gate_wrap(permeability = "args", is_qfree = True)`` is the
    :meth:`lifted <qrisp.lifted>` decorator.


    .. warning::

        Using ``gate_wrap`` without specifying permeability and ``qfree``-ness on
        functions processing a lot of qubits, can causes long compile times, since the
        unitaries of these gates have to be determined numerically.

    .. warning::

        Incorrect information about permeability and ``qfree``-ness can yield incorrect
        compilation results. If you are unsure, use the ``verify`` keyword on a small
        scale first.


    Parameters
    ----------

    permeability : string or list, optional
        Specify the permeability behavior of the function. When given "args", it is
        assumed that the gate is permeable only on the qubits of the arguments. When
        given "full", it is assumed that the gate is permeable on every qubit it acts on
        (i.e. also the result). When given a list of integers it is assumed, that the
        gate is permeable on the qubits of the arguments corresponding to the integers.
        The default is None.
    is_qfree : bool, optional
        Specify the qfree-ness of the function. The default is None.
    name : string, optional
        String which will be used for naming the gate object. The default is None.
    verify : bool, optional
        If set to ``True``, the specified information about permeability and
        ``qfree``-ness will be checked numerically. The default is ``False``.

    Examples
    --------

    We create a simple function wrapping up multiple gates: ::


        from qrisp import QuantumVariable, cx, x, h, z, gate_wrap

        @gate_wrap
        def example_function(a, b):

            cx(a,b)
            x(a)
            cx(b,a)
            h(b)

        a = QuantumVariable(3)
        b = QuantumVariable(3)

        example_function(a, b)

    >>> print(a.qs)

    .. code-block:: none

        QuantumCircuit:
        --------------
             ┌───────────────────┐
        b.0: ┤0                  ├
             │                   │
        b.1: ┤1                  ├
             │                   │
        b.2: ┤2                  ├
             │  example_function │
        a.0: ┤3                  ├
             │                   │
        a.1: ┤4                  ├
             │                   │
        a.2: ┤5                  ├
             └───────────────────┘
        Live QuantumVariables:
        ---------------------
        QuantumVariable a
        QuantumVariable b

    >>> print(a.qs.transpile())

    .. code-block:: none

             ┌───┐                    ┌───┐
        b.0: ┤ X ├─────────────────■──┤ H ├──────────
             └─┬─┘┌───┐            │  └───┘┌───┐
        b.1: ──┼──┤ X ├────────────┼────■──┤ H ├─────
               │  └─┬─┘┌───┐       │    │  └───┘┌───┐
        b.2: ──┼────┼──┤ X ├───────┼────┼────■──┤ H ├
               │    │  └─┬─┘┌───┐┌─┴─┐  │    │  └───┘
        a.0: ──■────┼────┼──┤ X ├┤ X ├──┼────┼───────
                    │    │  ├───┤└───┘┌─┴─┐  │
        a.1: ───────■────┼──┤ X ├─────┤ X ├──┼───────
                         │  ├───┤     └───┘┌─┴─┐
        a.2: ────────────■──┤ X ├──────────┤ X ├─────
                            └───┘          └───┘


    In the next example, we create a function that performs no quantum gates and specify
    that it is qfree and permeable on the second argument but not on the first. ::

        from qrisp import QuantumCircuit

        @gate_wrap(permeability = [1], is_qfree = True)
        def example_function(arg_0, arg_1):

            res = QuantumVariable(1)

            #Append an identity gate
            res.qs.append(QuantumCircuit(3).to_gate(), [arg_0, arg_1, res])

            return res

        qv_0 = QuantumVariable(1)
        qv_1 = QuantumVariable(1)

        res = example_function(qv_0, qv_1)

    >>> print(qv_0.qs)

    .. code-block:: none

        QuantumCircuit:
        --------------
                ┌───────────────────┐
        qv_0.0: ┤0                  ├
                │                   │
        qv_1.0: ┤1 example_function ├
                │                   │
         res.0: ┤2                  ├
                └───────────────────┘
        Live QuantumVariables:
        ---------------------
        QuantumVariable qv_0
        QuantumVariable qv_1
        QuantumVariable res


    >>> qv_1.uncompute()
    >>> print(qv_0.qs)

    .. code-block:: none

        QuantumCircuit:
        --------------
                ┌───────────────────┐
        qv_0.0: ┤0                  ├
                │                   │
        qv_1.0: ┤1 example_function ├
                │                   │
         res.0: ┤2                  ├
                └───────────────────┘
        Live QuantumVariables:
        ---------------------
        QuantumVariable qv_0
        QuantumVariable res

    Since ``arg_1`` is marked as permeable, there are no further gates required for
    uncomputation. The situation is different for the other two QuantumVariables, where
    #the qubits are not marked as permeable.

    >>> qv_0.uncompute(do_it = False)
    >>> res.uncompute()
    >>> print(qv_0.qs)

    .. code-block:: none

        QuantumCircuit:
        --------------
                ┌───────────────────┐┌──────────────────────┐
        qv_0.0: ┤0                  ├┤0                     ├
                │                   ││                      │
        qv_1.0: ┤1 example_function ├┤1 example_function_dg ├
                │                   ││                      │
         res.0: ┤2                  ├┤2                     ├
                └───────────────────┘└──────────────────────┘
        Live QuantumVariables:
        ---------------------
    """

    """
    if len(args):
        return gate_wrap_inner(args[0])

    else:

        def gate_wrap_helper(function):
            return gate_wrap_inner(
                function,
                permeability=permeability,
                is_qfree=is_qfree,
                name=name,
                verify=verify,
            )

        return gate_wrap_helper
    """

    def gate_wrap_helper(function):
        @functools.wraps(function)
        def wrapper(*fargs, **fkwargs):
            # gate_wrap_inner is assumed to return another callable (the actual decorated function),
            # which we then call with fargs, fkwargs
            return gate_wrap_inner(
                function,
                permeability=permeability,
                is_qfree=is_qfree,
                name=name,
                verify=verify,
            )(*fargs, **fkwargs)

        return wrapper

    # If the decorator is called directly with a function (e.g. @gate_wrap)
    # args will contain that function as args[0].
    if len(args) == 1 and callable(args[0]):
        return gate_wrap_helper(args[0])
    else:
        # Otherwise, return the decorator that can be applied to a function later
        return gate_wrap_helper


def gate_wrap_inner(
    function, permeability=None, is_qfree=None, name=None, verify=False
):

    qached_function = function

    def wrapped_function(
        *args, permeability=permeability, is_qfree=is_qfree, verify=verify, **kwargs
    ):

        from qrisp.jasp import check_for_tracing_mode

        if check_for_tracing_mode():
            return qached_function(*args, **kwargs)

        wrapped_function.__name__ = function.__name__
        from qrisp import QuantumArray, QuantumVariable, merge
        from qrisp.circuit import Qubit
        from qrisp.core import recursive_qs_search, recursive_qv_search
        from qrisp.environments import GateWrapEnvironment

        try:
            qs = find_qs(args)
        except:
            qs_list = recursive_qs_search([args, kwargs])
            qs = qs_list[0]

        initial_qubits = set(qs.qubits)

        if name is None:
            gwe = GateWrapEnvironment(name=function.__name__)
        else:
            gwe = GateWrapEnvironment(name=name)

        with gwe:
            result = function(*args, **kwargs)

        if len(qs.env_stack):
            gwe.compile()

        if gwe in qs.data:
            qs.data.remove(gwe)

        if gwe.instruction is None:
            return result

        created_qubits = set(qs.qubits) - initial_qubits

        ancillas = []

        for qb in created_qubits:
            if qb.allocated is False:
                ancillas.append(qb)

        if is_qfree is not None:
            if verify and is_qfree:
                from qrisp.permeability import is_qfree as is_qfree_function

                if not is_qfree_function(gwe.instruction.op):
                    raise Exception(
                        f"Verification of qfree-ness for function {function.__name__} "
                        f"failed"
                    )

            gwe.instruction.op.is_qfree = is_qfree

        if permeability is not None:
            permeability_dict = {i: None for i in range(gwe.instruction.op.num_qubits)}
            permeable_qubits = []
            not_permeable_qubits = []

            if isinstance(permeability, list):

                for i in range(len(args)):

                    if i in permeability:
                        extension_list = permeable_qubits
                    else:
                        extension_list = not_permeable_qubits

                    arg = args[i]

                    if isinstance(arg, QuantumVariable):
                        extension_list += arg.reg
                    elif isinstance(arg, (tuple, list)):
                        for item in arg:
                            if isinstance(item, Qubit):
                                extension_list.append(item)
                            elif isinstance(item, QuantumVariable):
                                extension_list += item.reg
                    elif isinstance(arg, QuantumArray):
                        for qv in arg.flatten():
                            extension_list += qv.reg

                if isinstance(result, QuantumVariable):
                    not_permeable_qubits += result.reg
                elif isinstance(result, (tuple, list)):
                    for item in result:
                        if isinstance(item, Qubit):
                            not_permeable_qubits.append(item)
                        elif isinstance(item, QuantumVariable):
                            not_permeable_qubits += item.reg
                elif isinstance(result, QuantumArray):
                    for qv in result.flatten():
                        not_permeable_qubits += qv.reg

            elif isinstance(permeability, str):

                for arg in args:
                    if isinstance(arg, QuantumVariable):
                        permeable_qubits += arg.reg
                    elif isinstance(arg, (tuple, list)):
                        for item in arg:
                            if isinstance(item, Qubit):
                                permeable_qubits.append(item)
                            elif isinstance(item, QuantumVariable):
                                permeable_qubits += item.reg
                    elif isinstance(arg, QuantumArray):
                        for qv in arg.flatten():
                            permeable_qubits += qv.reg

                if permeability == "full":
                    extension_list = permeable_qubits
                elif permeability == "args":
                    extension_list = not_permeable_qubits
                else:
                    raise Exception(f"Don't know permeability option {permeability}")

                if isinstance(result, QuantumVariable):
                    extension_list += result.reg
                elif isinstance(result, (tuple, list)):
                    for item in result:
                        if isinstance(item, Qubit):
                            extension_list.append(item)
                        elif isinstance(item, QuantumVariable):
                            extension_list += item.reg
                elif isinstance(result, QuantumArray):
                    for qv in result.flatten():
                        extension_list += qv.reg

            for i in range(len(gwe.instruction.qubits)):

                qb = gwe.instruction.qubits[i]
                if qb in permeable_qubits:
                    permeability_dict[i] = True
                elif qb in not_permeable_qubits:
                    permeability_dict[i] = False
                elif qb in ancillas:

                    # Even though ancilla qubits are permeable, we want to be able to
                    # use the gate_wrap decorator as an interface to perform
                    # recomputation. If we mark them as permeable, Unqomp won't  wrap
                    # the uncomputed gate in alloc/dealloc gates but instead wrap both
                    # the computation gate and the recomputation in alloc/dealloc gates

                    # To undestand this behavior better consider the following example

                    # from qrisp import QuantumFloat
                    # a = QuantumFloat(2)
                    # qf_res = a * a
                    # qf_res.uncompute()
                    # print(qf_res.qs)

                    # If we mark the ancilla qubits as permeable, this gives

                    # QuantumCircuit:
                    # ---------------
                    #              ┌──────────┐┌──────────┐┌─────────────┐
                    #         a.0: ┤ qb_alloc ├┤0         ├┤0            ├──────────────
                    #              ├──────────┤│          ││             │
                    #         a.1: ┤ qb_alloc ├┤1         ├┤1            ├──────────────
                    #              ├──────────┤│          ││             │┌────────────┐
                    #    return.0: ┤ qb_alloc ├┤2         ├┤2            ├┤ qb_dealloc ├
                    #              ├──────────┤│          ││             │├────────────┤
                    #    return.1: ┤ qb_alloc ├┤3 __mul__ ├┤3 __mul___dg ├┤ qb_dealloc ├
                    #              ├──────────┤│          ││             │├────────────┤
                    #    return.2: ┤ qb_alloc ├┤4         ├┤4            ├┤ qb_dealloc ├
                    #              ├──────────┤│          ││             │├────────────┤
                    #    return.3: ┤ qb_alloc ├┤5         ├┤5            ├┤ qb_dealloc ├
                    #              ├──────────┤│          ││             │├────────────┤
                    # sbp_anc_0.0: ┤ qb_alloc ├┤6         ├┤6            ├┤ qb_dealloc ├
                    #              └──────────┘└──────────┘└─────────────┘└────────────┘
                    # Live QuantumVariables:
                    # ----------------------
                    # QuantumFloat a

                    # If we set them to non-permeable, we get instead

                    # QuantumCircuit:
                    # ---------------
                    #              ┌──────────┐┌──────────┐                          ┌─────────────┐»
                    #         a.0: ┤ qb_alloc ├┤0         ├──────────────────────────┤0            ├»
                    #              ├──────────┤│          │                          │             │»
                    #         a.1: ┤ qb_alloc ├┤1         ├──────────────────────────┤1            ├»
                    #              ├──────────┤│          │                          │             │»
                    #    return.0: ┤ qb_alloc ├┤2         ├──────────────────────────┤2            ├»
                    #              ├──────────┤│          │                          │             │»
                    #    return.1: ┤ qb_alloc ├┤3 __mul__ ├──────────────────────────┤3 __mul___dg ├»
                    #              ├──────────┤│          │                          │             │»
                    #    return.2: ┤ qb_alloc ├┤4         ├──────────────────────────┤4            ├»
                    #              ├──────────┤│          │                          │             │»
                    #    return.3: ┤ qb_alloc ├┤5         ├──────────────────────────┤5            ├»
                    #              ├──────────┤│          │┌────────────┐┌──────────┐│             │»
                    # sbp_anc_0.0: ┤ qb_alloc ├┤6         ├┤ qb_dealloc ├┤ qb_alloc ├┤6            ├»
                    #              └──────────┘└──────────┘└────────────┘└──────────┘└─────────────┘»
                    # «
                    # «        a.0: ──────────────
                    # «
                    # «        a.1: ──────────────
                    # «             ┌────────────┐
                    # «   return.0: ┤ qb_dealloc ├
                    # «             ├────────────┤
                    # «   return.1: ┤ qb_dealloc ├
                    # «             ├────────────┤
                    # «   return.2: ┤ qb_dealloc ├
                    # «             ├────────────┤
                    # «   return.3: ┤ qb_dealloc ├
                    # «             ├────────────┤
                    # «sbp_anc_0.0: ┤ qb_dealloc ├
                    # «             └────────────┘
                    # Live QuantumVariables:
                    # ----------------------
                    # QuantumFloat a

                    # In both cases, sbp_anc is recomputed, but in the second case, the
                    # reallocation allows the compiler to use the free qubit elsewhere
                    # and afterwards pick a potentially different qubit for the
                    # recomputation.

                    permeability_dict[i] = False

                # for i in range(len(gwe.instruction.qubits)):
                #     if permeability_dict[i] is None:
                #         permeability_dict[i] = False

                if verify:
                    from qrisp.permeability import is_permeable

                    permeable_qubit_indices = []

                    for i in range(gwe.instruction.op.num_qubits):
                        if permeability_dict[i]:
                            permeable_qubit_indices.append(i)

                    if not is_permeable(gwe.instruction.op, permeable_qubit_indices):
                        raise Exception(
                            f"Verification of permeability for function "
                            f"{function.__name__} failed"
                        )

                gwe.instruction.op.permeability = permeability_dict

        return result

    return wrapped_function


def find_qs(args):

    from qrisp.jasp import TracingQuantumSession, check_for_tracing_mode

    if check_for_tracing_mode():
        return TracingQuantumSession.get_instance()

    if hasattr(args, "qs"):
        return args.qs()

    from qrisp import QuantumArray, QuantumVariable, Qubit

    for arg in args:
        if isinstance(arg, (QuantumVariable, QuantumArray)):
            return arg.qs
        if isinstance(arg, Qubit):
            return arg.qs()

    for arg in args:
        if isinstance(arg, (list, tuple)):
            try:
                return find_qs(arg)
            except:
                pass
        if isinstance(arg, dict):
            try:
                return find_qs(arg.items())
            except:
                pass

    raise Exception(f"Couldn't find QuantumSession in input {args}")


# Function to measure multiple quantum variables at once to assess their entanglement
def multi_measurement(qv_list, shots=None, backend=None):
    """
    This functions facilitates the measurement of multiple QuantumVariables at the same
    time. This can be used if the entanglement structure between several
    QuantumVariables is of interest.

    Parameters
    ----------
    qv_list : list[QuantumVariable]
        A list of QuantumVariables.
    shots : int, optional
        The amount of shots to perform. The default is given by the backend used.
    backend : BackendClient, optional
        The backend to evaluate the compiled QuantumCircuit on. By default, the backend
        from default_backend.py will be used.

    Raises
    ------
    Exception
        Tried to perform measurement with open environments.

    Returns
    -------
    counts_list : list
        A list of tuples. The first element of each tuple is a tuple again and contains
        the labels of the QuantumVariables. The second element is a float and indicates
        the probability of measurement.

    Examples
    --------

    We entangle three QuantumFloats via addition and perform a multi-measurement:

    >>> from qrisp import QuantumFloat, h, multi_measurement
    >>> qf_0 = QuantumFloat(4)
    >>> qf_1 = QuantumFloat(4)
    >>> qf_0[:] = 3
    >>> qf_1[:] = 2
    >>> h(qf_1[0])
    >>> qf_sum = qf_0 + qf_1
    >>> multi_measurement([qf_0, qf_1, qf_sum])
    {(3, 2, 5): 0.5, (3, 3, 6): 0.5}

    """

    from qrisp.jasp import check_for_tracing_mode

    if check_for_tracing_mode():
        raise Exception(
            "Tried to call multi_measurement in Jasp mode. Please use terminal_sampling instead"
        )

    if backend is None:
        if qv_list[0].qs.backend is None:
            from qrisp.default_backend import def_backend

            backend = def_backend
        else:
            backend = qv_list[0].qs.backend

    if len(qv_list[0].qs.env_stack) != 0:
        raise Exception("Tried to perform measurement with open environments")

    from qrisp import merge

    merge(qv_list)

    # Copy circuit in order to prevent modification
    from qrisp import (
        QuantumArray,
        QuantumVariable,
        recursive_qa_search,
        recursive_qv_search,
    )
    from qrisp.core.compilation import qompiler

    temp = recursive_qv_search(qv_list)

    for qa in recursive_qa_search(qv_list):
        temp.extend(list(qa.flatten()))

    compiled_qc = qompiler(
        qv_list[0].qs, intended_measurements=sum([qv.reg for qv in temp], [])
    )
    # compiled_qc = qv_list[0].qs.copy()
    # Add classical registers for the measurement results to be stored in
    cl_reg_list = []

    for var in qv_list[::-1]:
        cl_reg = []

        if isinstance(var, QuantumArray):
            qubits = sum([qv.reg for qv in var.flatten()[::-1]], [])
        elif isinstance(var, QuantumVariable):
            qubits = var.reg
        else:
            raise Exception(f"Found type {type(var)} in measurement list")

        for i in range(len(qubits)):
            cl_reg.append(compiled_qc.add_clbit())

        cl_reg_list.append(cl_reg)

        # Add measurement instruction
        compiled_qc.measure(qubits, cl_reg)

    # counts = execute(qs_temp, backend, basis_gates = basis_gates,
    # noise_model = noise_model, shots = shots).result().get_counts()
    counts = backend.run(compiled_qc, shots)
    counts = {k: counts[k] for k in sorted(counts)}
    shots = sum(counts.values())

    # Convert the labeling bistrings of counts into list of labels
    new_counts = {}
    for i in range(len(counts)):
        # Retrieve the separated strings of each measurement variable

        counts_strings = []
        counts_bitstring = list(counts.keys())[i]

        bitstring_adress = 0
        for j in range(len(cl_reg_list)):
            cl_reg = cl_reg_list[::-1][j]
            counts_strings.append(
                counts_bitstring[bitstring_adress : bitstring_adress + len(cl_reg)][
                    ::-1
                ]
            )
            bitstring_adress += len(cl_reg)

        # Convert to integers and insert outcome labels
        counts_values = []
        for j in range(len(counts_strings)):
            outcome_int = int(counts_strings[j][::-1], 2)
            try:
                label = qv_list[j].decoder(outcome_int)
                if isinstance(label, np.ndarray):
                    from qrisp import OutcomeArray

                    label = OutcomeArray(label)
                counts_values.append(label)
            except AttributeError:
                counts_values.append(outcome_int)

        # Create array
        array_state = tuple(counts_values)
        try:
            no_of_shots_executed = sum(counts.values())
            new_counts[array_state] = (
                counts[list(counts.keys())[i]] / no_of_shots_executed
            )
        except TypeError:
            raise Exception(
                "Tried to create measurement outcome dic for QuantumVariable "
                "with unhashable labels"
            )
        # Append to the counts list
        # counts_list.append((array_state, counts[list(counts.keys())[i]]/shots))

    # Sort counts_list such the most probable values come first
    new_counts = dict(sorted(new_counts.items(), key=lambda item: -item[1]))

    return new_counts


# Function to apply a phase function of signature phase_function(x,y,z..) -> float
# which specifies the phase for each constellation of outcome labels of the quantum
# variables in qv_list.
def app_phase_function(qv_list, phase_function, t=1, **kwargs):
    # Prepare the list of index tuples
    # For this we first create a list of outcome indices of each qv first
    index_lists = [list(range(2**qv.size)) for qv in qv_list]

    # We now calculate the direct product in order to obtain every possible combination
    from itertools import product

    product_index_list = list(product(*index_lists))

    # The next step is to iterate over every combination in order to determine the
    # phases.
    phases = []
    for i in range(len(product_index_list)):
        # Calculate the outcome labels of the current constellation of indices
        labels = [
            qv_list[j].decoder(product_index_list[i][j]) for j in range(len(qv_list))
        ]

        # Calculate the phase
        phases.append(phase_function(*labels, **kwargs) * t)

    # Synthesize phase
    from qrisp import gray_phase_synth_qb_list

    gray_phase_synth_qb_list(
        qv_list[0].qs, sum([qv.reg[::-1] for qv in qv_list], []), phases
    )


def as_hamiltonian(hamiltonian):
    r"""
    Decorator that recieves a regular Python function (returning a float) and returns a
    function of QuantumVariables, applying phases based on the function's output.

    Parameters
    ----------
    hamiltonian : function
        A function of arbitrary (non-quantum) variables returning a float.

    Returns
    -------
    hamiltonian_application : function
        A function of QuantumVariables, which applies the phase dictated by the
        hamiltonian to the corresponding states.

    Examples
    --------


    In this example we will demonstrate how a phase function with multiple arguments can
    be synthesized. For this we will create a phase function which encodes the fourier
    transform of different integers on the QuantumFloat x conditioned on the value of a
    QuantumChar ch. We will then apply the inverse Fourier transform to x and measure
    the results.
    ::

        import numpy as np
        from qrisp import QuantumChar, QuantumFloat, QFT, h
        #Create Variables
        x_size = 3
        x = QuantumFloat(x_size, 0, signed = False)

        ch = QuantumChar()

        # Bring x into uniform superposition so the phase function application yields
        # a fourier transformed computation basis state
        h(x)


        #Bring ch into partial superposition (here |a> + |b> + |c> + |d>)
        h(ch[0])
        h(ch[1])

        from qrisp import multi_measurement, as_hamiltonian

        #In order to define the hamiltonian, we use regular Python syntax
        #The decorator "as_hamiltonian" turns it into a function 
        #that takes Quantum Variables as arguments. The decorator will add the 
        #keyword argument t to the function which mimics the t in exp(i*H*t)


        @as_hamiltonian
        def apply_multi_var_hamiltonian(ch, x):
            if ch == "a":
                k = 2
            elif ch == "b":
                k = 7
            elif ch == "c":
                k = 3
            else:
                k = 4

            #Return phase value
            #This is the phase distribution of the Fourier-transform
            #of the computational basis state |k>
            return k*x * 2*np.pi/2**x_size


        #Apply Hamiltonian
        apply_multi_var_hamiltonian(ch,x, t = 1)


        #Apply inverse Fourier transform
        QFT(x, inv = True)


    Acquire measurement results

    >>> multi_measurement([ch, x])
    {('a', 2): 0.25, ('b', 7): 0.25, ('c', 3): 0.25, ('d', 4): 0.25}

    We see that the measurement results correspond to what we specified in the
    Hamiltonian.

    In Bra-Ket notation, before applying the Hamiltonian, we are in the state

    .. math::

        \ket{\psi} = \frac{1}{\sqrt{4}}(\ket{a} + \ket{b} + \ket{c} + \ket{d})
        \left( \frac{1}{\sqrt{8}} \sum_{x = 0}^8 \ket{x} \right)

    We then apply the Hamiltonian:

    .. math::

        \text{exp}(i\text{H(ch, x)})\ket{\psi} = \frac{1}{\sqrt{32}}
        ( &\ket{a} \sum_{x = 0}^{2^3-1} \text{exp}(2x \frac{2 \pi i}{2^3}) \ket{x} \\
        +&\ket{b} \sum_{x = 0}^{2^3-1} \text{exp}(7x \frac{2 \pi i}{2^3}) \ket{x} \\
        +&\ket{c} \sum_{x = 0}^{2^3-1} \text{exp}(3x \frac{2 \pi i}{2^3}) \ket{x} \\
        +&\ket{d} \sum_{x = 0}^{2^3-1} \text{exp}(4x \frac{2 \pi i}{2^3}) \ket{x})

    For each branch, the QuantumFloat tensor-factor is in a Fourier-transformed
    computational basis state. Thus, if we apply the inverse QFT, we receive:

    .. math::

        &\text{QFT}^{-1}\text{exp}(i\text{H(ch, x)})\ket{\psi} \\
            = & \frac{1}{\sqrt{4}} (\ket{a} \ket{2} + \ket{b} \ket{7} +\ket{c} \ket{3}
            + \ket{d} \ket{4})

    """

    def hamiltonian_application(*args, t=1, **kwargs):
        gate_wrap(app_phase_function)(args, hamiltonian, t=t, **kwargs)
        # app_phase_function(args, hamiltonian, t = t, **kwargs)

    return hamiltonian_application


def perm_lock(qubits):
    """
    Locks a list of qubits such that only permeable gates can be executed on these
    qubits. This means that an error will be raised if the user attempts to perform any
    operation involving these qubits if the operation does not commute with the
    Z-operator of this qubit. For more information, what a permeable gate is, check the
    :ref:`uncomputation documentation <uncomputation>`.

    This can be helpfull as it forbids all operations that change that computational
    basis state of this qubit but still allow controling on this qubit or applying
    phase gates.


    The effect of this function can be reversed using perm_unlock.

    Parameters
    ----------
    qubits : list[Qubit] or QuantumVariable
        The qubits to phase-tolerantly lock.

    Examples
    --------

    We create a QuantumChar, perm-lock it's Qubits and attempt to initialize.

    >>> from qrisp import QuantumChar, perm_lock, cx, p
    >>> q_ch_0 = QuantumChar()
    >>> perm_lock(q_ch_0)
    >>> q_ch_0[:] = "g"
    Exception: Tried to perform non-permeable operations on perm_locked qubits

    We now create a second QuantumChar and perform a CNOT gate

    >>> q_ch_1 = QuantumChar()
    >>> cx(q_ch_0[3], q_ch_1[2])

    Phase-gates are possible, too

    >>> p(0.1, q_ch_0)

    """
    from qrisp.circuit.quantum_circuit import convert_to_qb_list

    for qb in convert_to_qb_list(qubits):
        if isinstance(qb, list):
            for item in qb:
                perm_lock(item)

            continue
        qb.perm_lock = True


def perm_unlock(qubits):
    """
    Reverses the effect of "perm_lock".

    Parameters
    ----------
    qubits : list[Qubit] or QuantumVariable
        The qubits to phase-tolerantly unlock.

    Examples
    --------

    We create a QuantumChar, perm-lock it's Qubits and attempt to initialize.

    >>> from qrisp import QuantumChar, perm_lock, perm_unlock
    >>> q_ch = QuantumChar()
    >>> perm_lock(q_ch)
    >>> q_ch[:] = "g"
    Exception: Tried to perform non-permeable operations on perm_locked qubits

    >>> perm_unlock(q_ch)
    >>> q_ch[:] = "g"
    >>> print(q_ch)
    {'g': 1.0}

    """
    from qrisp.circuit.quantum_circuit import convert_to_qb_list

    for qb in convert_to_qb_list(qubits):
        if isinstance(qb, list):
            for item in qb:
                perm_unlock(item)
            continue
        qb.perm_lock = False


def lock(qubits):
    """
    Locks a list of qubits, implying an error will be raised if the user tries to
    perform any operation involving these qubits.

    This can be reversed by calling unlock.

    Parameters
    ----------
    qubits : list[Qubit] or QuantumVariable
        The list of Qubits to lock.

    Examples
    --------

    We create a QuantumChar, lock it's Qubits and attempt to initialize.

    >>> from qrisp import QuantumChar, lock
    >>> q_ch = QuantumChar()
    >>> lock(q_ch)
    >>> q_ch[:] = "g"
    Exception: Tried to operation on locked qubits
    """
    from qrisp.circuit.quantum_circuit import convert_to_qb_list

    for qb in convert_to_qb_list(qubits):
        if isinstance(qb, list):
            for item in qb:
                lock(item)
            continue

        qb.lock = True


def unlock(qubits):
    """
    Reverses the effect of "lock".

    Parameters
    ----------
    qubits : list[Qubit] or QuantumVariable
        The list of Qubits to lock.

    Examples
    --------

    We create a QuantumChar, lock it's Qubits and attempt to initialize.

    >>> from qrisp import QuantumChar, lock, unlock
    >>> q_ch = QuantumChar()
    >>> lock(q_ch)
    >>> q_ch[:] = "g"
    Exception: Tried to perform operations on locked qubits

    We now unlock and try again

    >>> unlock(q_ch)
    >>> q_ch[:] = "g"
    >>> print(q_ch)
    {'g': 1.0}

    """
    from qrisp.circuit.quantum_circuit import convert_to_qb_list

    for qb in convert_to_qb_list(qubits):
        if isinstance(qb, list):
            for item in qb:
                unlock(item)
            continue
        qb.lock = False


def benchmark_function(function):
    def benchmarked_function(*args, sort_stats="tottime", stat_amount=20, **kwargs):
        def slow_function():
            function(*args, **kwargs)

        import cProfile
        import pstats

        profile = cProfile.Profile()

        profile.runcall(slow_function)

        ps = pstats.Stats(profile)
        ps.strip_dirs()
        ps.sort_stats(sort_stats)
        ps.print_stats(stat_amount)

    return benchmarked_function


def custom_qv(labels, decoder=None, qs=None, name=None):
    if not isinstance(labels, list):
        raise Exception(
            "Tried to create custom QuantumVariable without providing a list type"
        )

    if len(labels) == 0:
        raise Exception(
            "Tried to create custom QuantumVariable without providing labels"
        )
    elif len(labels) == 1:
        n = 1
    else:
        n = int(np.ceil(np.log2(len(labels))))

    from qrisp import QuantumVariable

    class CustomQuantumVariable(QuantumVariable):
        def __init__(self, qs=None, name=None):
            super().__init__(n, qs=qs, name=name)

        def decoder(self, x):
            if decoder is None:
                if x < len(labels):
                    return labels[x]
                else:
                    return "undefined_label_" + str(x)

            return decoder(x)

    return CustomQuantumVariable(qs=qs, name=name)


# This is required in the qswitch-based state preparation,
# where it is called inside jrange loops, because DynamicQubitArray
# does not support reverse iteration.
def bit_reverse(i, width):
    """
    Jasp-compatible bit-reversal function.

    Interprets ``i`` as a ``width``-bit binary integer
    and returns the decimal integer corresponding to the bit-reversal of ``i``.
    The maximum supported width is 64 bits.

    This function can be used in Jasp-mode and within a `jrange` loop.
    It does not use any Python or Jax control flow, but only Jax array operations.

    Parameters
    ----------
    i : jnp.ndarray
        Index to be bit-reversed.
    width : jnp.ndarray
        Bit-width for the reversal (scalar array).

    Returns
    -------
    jnp.ndarray
        Bit-reversed index.


    Examples
    --------

    For ``i=5`` and ``width=3``, the binary representation
    of ``5`` is ``101``, and its bit-reversal is (again) ``101``, which is ``5`` in decimal.

    >>> from qrisp.misc.utility import bit_reverse
    >>> bit_reverse(5, 3)
    5

    For ``i=3`` and ``width=4``, the binary representation
    of ``3`` is ``0011``, and its bit-reversal is ``1100``, which is ``12`` in decimal.

    >>> bit_reverse(3, 4)
    12

    """
    i = jnp.asarray(i, dtype=jnp.uint64)
    width = jnp.asarray(width, dtype=jnp.uint64)

    m1 = jnp.uint64(0x5555555555555555)
    m2 = jnp.uint64(0x3333333333333333)
    m3 = jnp.uint64(0x0F0F0F0F0F0F0F0F)
    m4 = jnp.uint64(0x00FF00FF00FF00FF)
    m5 = jnp.uint64(0x0000FFFF0000FFFF)
    m6 = jnp.uint64(0x00000000FFFFFFFF)

    i = ((i >> 1) & m1) | ((i & m1) << 1)
    i = ((i >> 2) & m2) | ((i & m2) << 2)
    i = ((i >> 4) & m3) | ((i & m3) << 4)
    i = ((i >> 8) & m4) | ((i & m4) << 8)
    i = ((i >> 16) & m5) | ((i & m5) << 16)
    i = ((i >> 32) & m6) | ((i & m6) << 32)

    return i >> jnp.asarray(64, jnp.uint64) - width


def get_statevector_function(qs, decimals=None):
    if len(qs.qv_list) == 0:
        return lambda x: 0
    else:
        from qrisp.simulator import statevector_sim

        compiled_qc = qs.compile()
        sv_array = statevector_sim(compiled_qc)

        if decimals is not None:
            sv_array = np.round(sv_array, decimals)

        def statevector(label_constellation, round=None):
            from qrisp import bin_rep

            qs = list(label_constellation.keys())[0].qs

            if len(label_constellation) != len(qs.qv_list):
                missing_variables = set([qv.name for qv in qs.qv_list]) - set(
                    [qv.name for qv in label_constellation.keys()]
                )
                raise Exception(
                    "Tried to invoke statevector debugger without specifying an "
                    "outcome label for each QuantumVariable registered in "
                    "QuantumSession. Missing variables are: " + str(missing_variables)
                )

            bitstring = len(compiled_qc.qubits) * ["0"]

            for qf in label_constellation.keys():
                label_int = qf.encoder(label_constellation[qf])
                bin_label_int = bin_rep(label_int, qf.size)[::-1]

                for i in range(qf.size):
                    qubit_pos = compiled_qc.qubits.index(qf[i])
                    bitstring[qubit_pos] = bin_label_int[i]

            bitstring = "".join(bitstring)
            state_index = int(bitstring, base=2)

            if round is None:
                return sv_array[state_index]
            else:
                return np.around(sv_array[state_index], round)

        return statevector


def check_if_fresh(qubits, qs, ignore_q_envs=True):

    from qrisp import QuantumEnvironment

    if not ignore_q_envs:
        temp_data = list(qs.data)
        qs.data = []

        for i in range(len(temp_data)):
            if isinstance(temp_data[i], QuantumEnvironment):
                env = temp_data[i]
                env.compile()
            else:
                qs.append(temp_data[i])

    for qb in qubits:
        reversed_data = qb.qs().data[::-1]
        for instr in reversed_data:
            if isinstance(instr, QuantumEnvironment) and ignore_q_envs:
                continue
            if qb in instr.qubits:
                if instr.op.name == "qb_alloc":
                    break
                else:
                    return False
        else:
            return False

    return True


def get_measurement_from_qc(qc, qubits, backend, shots=None):
    # Add classical registers for the measurement results to be stored in
    cl = []
    for i in range(len(qubits)):
        cl.append(qc.add_clbit())

    # Add measurement instruction
    for i in range(len(qubits)):
        qc.measure(qubits[i], cl[i])

    # Execute circuit
    counts = backend.run(qc, shots=shots)

    # Remove other measurements outcomes from counts dic
    new_counts_dic = {}

    no_of_shots_executed = 0

    for key in counts.keys():
        # Remove possible whitespaces
        new_key = key.replace(" ", "")
        # Remove other measurements
        new_key = new_key[: len(cl)]

        new_key = int(new_key, base=2)
        try:
            new_counts_dic[new_key] += counts[key]
        except KeyError:
            new_counts_dic[new_key] = counts[key]

        no_of_shots_executed += counts[key]

    counts = new_counts_dic

    if abs(1 - no_of_shots_executed) < 1e-3:
        return counts
    # Normalize counts
    for key in counts.keys():
        counts[key] = counts[key] / abs(no_of_shots_executed)

    return counts


def find_calling_line(level=0):
    stack = traceback.extract_stack(limit=level + 3)
    return str(
        traceback.format_list(stack)[1].split("\n")[1].strip()
    )  # prints "a = fct1()"


def retarget_instructions(data, source_qubits, target_qubits):
    from qrisp import QuantumEnvironment, multi_session_merge, recursive_qs_search

    for i in range(len(data)):
        instr = data[i]

        if isinstance(instr, QuantumEnvironment):
            retarget_instructions(instr.original_data, source_qubits, target_qubits)
            retarget_instructions(instr.env_data, source_qubits, target_qubits)
            continue

        for j in range(len(instr.qubits)):
            if instr.qubits[j] in source_qubits:
                instr.qubits[j] = target_qubits[source_qubits.index(instr.qubits[j])]


def redirect_qfunction(function_to_redirect):
    """
    Decorator to turn a function returning a QuantumVariable into an in-place function.
    This can be helpful for manual uncomputation if we have a function returning some
    QuantumVariable, but we want the result to operate on some other variable, which is
    supposed to be uncomputed.

    Parameters
    ----------
    function_to_redirect : function
        A function returning a QuantumVariable.

    Raises
    ------
    Exception
        Given function did not return a QuantumVariable
    Exception
        Tried to redirect quantum function into QuantumVariable of differing size

    Returns
    -------
    redirected_function : function
        A function which performs the same operation as the input but now has the
        keyword argument target. Every instruction that would have been executed on the
        input functions result is executed on the QuantumVariable specified by target
        instead.


    Examples
    --------

    We create a function that determins the AND value of its inputs and redirect it
    onto another QuantumBool. ::

        from qrisp import QuantumBool, mcx, redirect_qfunction

        #This function has only two arguments and returns its result
        def AND(a, b):

            res = QuantumBool()

            mcx([a,b], res)

            return res

        a = QuantumBool(name = "a")
        b = QuantumBool(name = "b")
        c = QuantumBool(name = "c")

        #This function has two arguments and the keyword argument target
        redirected_AND = redirect_qfunction(AND)

        redirected_AND(a, b, target = c)


    >>> print(a.qs)

    .. code-block:: none

        QuantumCircuit:
        --------------
        b.0: ──■──
               │
        a.0: ──■──
             ┌─┴─┐
        c.0: ┤ X ├
             └───┘
        Live QuantumVariables:
        ---------------------
        QuantumBool b
        QuantumBool a
        QuantumBool c



    """
    import weakref

    from qrisp import QuantumArray, QuantumEnvironment, QuantumVariable, merge
    from qrisp.jasp import (
        TracingQuantumSession,
        check_for_tracing_mode,
        eval_jaxpr,
        injection_transform,
        make_jaspr,
    )

    def redirected_qfunction(*args, target=None, **kwargs):

        if check_for_tracing_mode():
            jaspr = make_jaspr(function_to_redirect, garbage_collection="manual")(
                *args, **kwargs
            ).flatten_environments()

            transformed_jaspr = injection_transform(jaspr, jaspr.outvars[0])

            qs = TracingQuantumSession.get_instance()
            abs_qc = qs.abs_qc
            from jax.tree_util import tree_flatten

            flattened_args = []

            flattened_args.append(target.reg.tracer)

            for arg in args:
                flattened_args.extend(tree_flatten(arg)[0])

            flattened_args.append(abs_qc)

            res = eval_jaxpr(transformed_jaspr, [])(*flattened_args)

            if len(transformed_jaspr.outvars) == 1:
                qs.abs_qc = res
            else:
                qs.abs_qc = res[-1]

        else:

            merge(
                [
                    arg
                    for arg in list(args) + [target]
                    if isinstance(arg, (QuantumVariable, QuantumArray))
                ]
            )
            env = QuantumEnvironment()
            env.manual_allocation_management = True
            qs = target.qs

            with env:
                res = function_to_redirect(*args, **kwargs)

                if not isinstance(res, QuantumVariable):
                    raise Exception("Given function did not return a QuantumVariable")

                target = list(target)

                if len(res) != len(target):
                    raise Exception(
                        "Tried to redirect quantum function into QuantumVariable of "
                        "differing size"
                    )

                i = 0
                res_is_new = False
                while i < len(env.env_qs.data):

                    instr = env.env_qs.data[i]

                    if isinstance(instr, QuantumEnvironment):
                        pass
                    elif instr.op.name == "qb_alloc" and instr.qubits[0] in list(res):
                        env.env_qs.data.pop(i)
                        res_is_new = True
                        continue
                    else:
                        for qb in instr.qubits:
                            qb.qs = weakref.ref(qs)

                    i += 1

                retarget_instructions(env.env_qs.data, list(res), target)

            if res_is_new:
                # Remove all traces of res
                res.delete()

                for i in range(res.size):
                    res.qs.qubits.remove(res[i])
                    res.qs.data.pop(-1)

                for i in range(len(res.qs.deleted_qv_list)):
                    qv = res.qs.deleted_qv_list[i]
                    if qv.name == res.name:
                        res.qs.deleted_qv_list.pop(i)
                        break

            return target

    redirected_qfunction.__name__ = function_to_redirect.__name__

    return redirected_qfunction


def get_sympy_state(qs, decimals):
    from sympy import (
        I,
        Rational,
        Symbol,
        cancel,
        cos,
        count_ops,
        exp,
        factor,
        nsimplify,
        pi,
        simplify,
        sin,
    )
    from sympy.physics.quantum import Ket, OrthogonalKet

    from qrisp.simulator import statevector_sim

    qv_list = list(qs.qv_list)

    labels = []
    for qv in qv_list:
        labels.append([qv.decoder(i) for i in range(2**qv.size)])

    compiled_qc = qs.compile()

    sv_array = statevector_sim(compiled_qc)

    if not sv_array.dtype == np.dtype("O"):
        angles = np.angle(sv_array) % (2 * np.pi) / (np.pi)

        if decimals is not None:
            sv_array = np.round(sv_array, decimals)
            angles = np.round(angles, decimals)
        else:
            sv_array = np.round(sv_array, 5)
            angles = np.round(angles, 5)

        nz_indices = np.nonzero(sv_array)[0]
        nnz = len(nz_indices)

    else:
        import sympy as sp

        nz_indices = []

        for i in range(len(sv_array)):
            entry = simplify(sv_array[i])

            for a in sp.preorder_traversal(entry):
                if isinstance(a, sp.Float):
                    entry = entry.subs(a, round(a, 5))

            sv_array[i] = entry

            if not sv_array[i] == 0:
                nz_indices.append(i)

        nnz = len(nz_indices)

    res = 0
    for ind in list(nz_indices):
        amplitude = sv_array[ind]

        if not sv_array.dtype == np.dtype("O"):

            if decimals is None:

                try:
                    abs_amp = trigify_amp(amplitude, nnz)
                except TypeError:
                    abs_amp = amplitude

                # For some reason there is a sympy error, when the angle is equal to 1
                if angles[ind] == 1:
                    phase = 1
                else:
                    phase = nsimplify(float(angles[ind]), tolerance=10**-5)

                if count_ops(phase) > 5:
                    phase = angles[ind]

                ket_expr = exp(I * phase * pi) * abs_amp * nnz**0.5
            else:

                ket_expr = sympy.N(amplitude, decimals)

        else:
            process_stack = [amplitude]
            while process_stack:
                a = process_stack.pop(0)
                if (
                    isinstance(a, (sympy.core.add.Add, sympy.core.mul.Mul))
                    and len(a.free_symbols) != 0
                ):
                    process_stack.extend(a.args)

                elif len(a.free_symbols) == 0:
                    sub_float = np.round(complex(a.evalf()), 5)

                    if np.abs(sub_float - 1) < 10**-5:
                        abs_amp = 1
                        continue
                    elif np.abs(sub_float) < 10**-5:
                        entry = entry.subs(a, 0)
                        continue
                    elif np.abs(sub_float) > 1:
                        continue
                    else:
                        abs_amp = trigify_amp(sub_float, nnz)

                    if np.angle(complex(a.evalf())) / np.pi == 1:
                        phase = -1
                    else:

                        phase = sp.exp(
                            sp.I
                            * nsimplify(
                                np.angle(complex(a.evalf())) / np.pi,
                                tolerance=10**-5,
                            )
                            * Symbol("pi")
                        )

                    expr = abs_amp * phase

                    amplitude = amplitude.subs(a, expr)

            amplitude = amplitude.subs(1j, sp.I)

            ket_expr = sp.trigsimp(amplitude) * nnz**0.5

        int_string = bin_rep(ind, len(compiled_qc.qubits))

        labels = []
        for qv in qv_list:
            bit_string = ""
            for qb in qv.reg:
                bit_string += int_string[compiled_qc.qubits.index(qb)]

            label = qv.decoder(int(bit_string[::-1], 2))
            ket_expr *= OrthogonalKet((label))

        res += ket_expr

    if decimals is None or sv_array.dtype == np.dtype("O"):
        res = cancel(nsimplify(1 / nnz**0.5) * res)

    if isinstance(res, sympy.core.mul.Mul):
        temp = 1
        for arg in res.args[:-1]:
            temp *= nsimplify(arg.subs({Symbol("pi"): pi}))

        res = temp * res.args[-1]

    res = res.subs({Symbol("pi"): pi})
    return res


def trigify_amp(amplitude, nnz):
    from sympy import (
        I,
        Rational,
        Symbol,
        cancel,
        cos,
        count_ops,
        exp,
        factor,
        latex,
        nsimplify,
        pi,
        simplify,
        sin,
    )

    cos_expr = nsimplify(float(np.arccos(np.abs(amplitude)) / np.pi), tolerance=10**-5)
    sin_expr = nsimplify(float(np.arcsin(np.abs(amplitude)) / np.pi), tolerance=10**-5)

    # if count_ops(sin_expr) > count_ops(cos_expr):
    if len(latex(sin_expr)) > len(latex(cos_expr)):
        expr = "cos"
        temp = cos_expr
    # elif count_ops(sin_expr) < count_ops(cos_expr):
    elif len(latex(sin_expr)) < len(latex(cos_expr)):
        expr = "sin"
        temp = sin_expr
    elif len(sin_expr.free_symbols) == 0:
        if sin_expr.evalf() > cos_expr.evalf():
            expr = "cos"
            temp = cos_expr
        else:
            expr = "sin"
            temp = sin_expr
    else:
        temp = nsimplify(np.abs(amplitude) * nnz**0.5, tolerance=10**-5) / nnz**0.5

    # if count_ops(temp) > 4:
    if len(latex(temp)) > 20:
        temp = (
            nsimplify(float(np.abs(amplitude) * nnz**0.5), tolerance=10**-5) / nnz**0.5
        )
        if len(latex(temp)) > 20:
            abs = np.abs(amplitude)

        else:
            abs = temp

    else:
        if expr == "cos":
            abs = cos(cos_expr * Symbol("pi"))
        else:
            abs = sin(sin_expr * Symbol("pi"))

    return abs


def render_qc(qc):
    latex_str = qc.to_latex()
    import os.path
    import subprocess
    import tempfile

    from IPython.display import Image, display

    with tempfile.TemporaryDirectory(prefix="texinpy_") as tmpdir:
        path = os.path.join(tmpdir, "document.tex")
        with open(path, "w") as fp:
            fp.write(latex_str)
        subprocess.run(["lualatex", path], cwd=tmpdir)
        subprocess.run(
            [
                "pdftocairo",
                "-singlefile",
                "-transp",
                "-r",
                "100",
                "-png",
                "document.pdf",
                "document",
            ],
            cwd=tmpdir,
        )

        im = Image(filename=os.path.join(tmpdir, "document.png"))
        display(im)


def lifted(*args, verify=False):
    """
    Shorthand for ``gate_wrap(permability = "args", is_qfree = True)``.

    A lifted function is ``qfree`` and permeable on its inputs. The results of lifted
    functions can be automatically uncomputed even if they contain functions that could
    not be uncomputed on their own.

    You can find more information about these concepts :ref:`here <Uncomputation>` or
    `here <https://silq.ethz.ch/overview#/overview/3_uncomputation>`_. Note that the
    concept of permeability in Qrisp is a more general version of Silq's ``const``.

    .. warning::

        Incorrect information about permeability and ``qfree``-ness can yield incorrect
        compilation results. If you are unsure, use the ``verify`` keyword on a small
        scale first.


    Parameters
    ----------

    verify : bool, optional
        If set to ``True``, the specified information about permeability and
        ``qfree``-ness will be checked numerically. The default is ``False``.

    Examples
    --------

    We create a function performing the `Margolus gate
    <https://arxiv.org/abs/quant-ph/0312225>`_. As it contains ``ry`` rotations,
    there are non-``qfree`` steps involved. Putting on the ``lifted`` decorator however
    marks the function as ``qfree`` as a whole.

    ::

        from qrisp import QuantumVariable, cx, ry, lifted
        from numpy import pi

        @lifted(verify = True)
        def margolus(control):

            res = QuantumVariable(1)
            ry(pi/4, res)
            cx(control[1], res)
            ry(-pi/4, res)
            cx(control[0], res)
            ry(pi/4, res)
            cx(control[1], res)
            ry(-pi/4, res)

            return res


        control = QuantumVariable(2)
        res = margolus(control)

    >>> print(res.qs)

    .. code-block:: none

        QuantumCircuit:
        --------------
                   ┌───────────┐
        control.0: ┤0          ├
                   │           │
        control.1: ┤1 margolus ├
                   │           │
            res.0: ┤2          ├
                   └───────────┘
        Live QuantumVariables:
        ---------------------
        QuantumVariable control
        QuantumVariable res

    >>> res.uncompute()
    >>> print(res.qs)

    .. code-block:: none

        QuantumCircuit:
        --------------
                   ┌───────────┐┌──────────────┐
        control.0: ┤0          ├┤0             ├
                   │           ││              │
        control.1: ┤1 margolus ├┤1 margolus_dg ├
                   │           ││              │
            res.0: ┤2          ├┤2             ├
                   └───────────┘└──────────────┘
        Live QuantumVariables:
        ---------------------
        QuantumVariable control

    Note that we set the ``verify`` keyword to ``True`` in this example. In more complex
    functions, involving many qubits this feature should only be used for bug-fixing on
    a small scale, since the verification can be time-consuming.

    """

    if len(args) == 0:

        def lifted_helper(function):
            return gate_wrap(permeability="args", is_qfree=True, verify=verify)(
                function
            )

        return lifted_helper

    else:
        return gate_wrap(permeability="args", is_qfree=True)(args[0])


def t_depth_indicator(op, epsilon):
    r"""
    This function returns the T-depth of an :ref:`Operation` object.

    According to `this paper <https://arxiv.org/abs/1403.2975>`_, the synthesis of an $RZ(\phi)$
    up to precision $\epsilon$ requires $3\text{log}_2(\frac{1}{\epsilon})$
    T-gates.

    Parameters
    ----------
    op : :ref:`Operation`
        The operation, whose T-depth should be estimated.
    epsilon : float
        The precision of the RZ gate simulation.


    Returns
    -------
    float
        The estimated T-depth of the Operation.

    """

    from qrisp import ClControlledOperation

    if isinstance(op, ClControlledOperation):
        return t_depth_indicator(op.base_op, epsilon)
    elif op.definition is not None:
        return op.definition.t_depth(epsilon)
    elif op.name in [
        "cx",
        "cx",
        "cz",
        "x",
        "y",
        "z",
        "s",
        "h",
        "s_dg",
        "sx",
        "sx_dg",
        "measure",
        "reset",
        "qb_alloc",
        "qb_dealloc",
        "barrier",
        "gphase",
    ]:
        return 0
    elif op.name in ["rx", "ry", "rz", "p", "u1"]:
        par = op.params[0] / (np.pi) % 1
        if par in [0, 1 / 2]:
            return 0
        elif par in [1 / 4, 3 / 4]:
            return 1
        else:
            return 3 * np.log2(1 / epsilon)
    elif op.name in ["t", "t_dg"]:
        return 1
    elif op.name == "u3":
        res = 0
        for i in range(3):
            par = op.params[0] / (np.pi) % 1
            if par in [0, 1 / 2]:
                pass
            elif par in [1 / 4, 3 / 4]:
                res += 1
            else:
                res += 3 * np.log2(1 / epsilon)
        return res
    else:
        raise Exception(f"Gate {op.name} not implemented")


def cnot_depth_indicator(op):
    r"""
    This function returns the CNOT-depth of an :ref:`Operation` object.

    In NISQ-era devices, CNOT gates are the restricting bottleneck for quantum
    circuit execution. This function can be used as a gate-speed specifier for
    the :meth:`compile <qrisp.QuantumSession.compile>` method.

    Parameters
    ----------
    op : :ref:`Operation`
        The operation, whose CNOT-depth should be computed.

    Returns
    -------
    float
        The CNOT-depth of the Operation.

    """

    from qrisp import ClControlledOperation

    if isinstance(op, ClControlledOperation):
        return cnot_depth_indicator(op.base_op)
    elif op.definition is not None:
        return op.definition.cnot_depth()
    if op.num_qubits == 1 or op.name == "barrier":
        return 0
    elif op.name in ["cx", "cx", "cz"]:
        return 1
    else:
        raise Exception(f"Gate {op.name} not implemented")


def inpl_adder_test(inpl_adder):
    """
    This function runs tests on a desired inplace addition function.
    An inplace addition function is a function mapping (a, b) to (a, a+b),
    where a is a :ref:`QuantumVariable`, list[:ref:`Qubit`] or an integer
    and b is either a :ref:`QuantumVariable` or a list[:ref:`Qubit`].

    Parameters
    ----------
    inpl_adder : callable
        A quantum inplace addition function that can either act on single QuantumVariables or on lists of Qubits
        by adding the first one to the second.

    Returns
    -------
    Bool:
        True if all tests are passed, else False/ Exceptions.

    Examples
    --------

    We test the built-in Cuccaro adder:

    ::

        from qrisp import cuccaro_adder, inpl_adder_test

        inpl_adder_test(cuccaro_adder)
        print("The cuccaro adder passed the tests without errors.")

    And now a new user-defined qcla adder:
    ::

        from qrisp import inpl_adder_test, qcla

        qcla_2_0 = lambda x, y : qcla(x, y, radix_base = 2, radix_exponent = 0)
        inpl_adder_test(qcla_2_0)
        print("The qcla_2_0 adder passed the tests without errors.")

    """
    from qrisp import QuantumBool, QuantumFloat, control, h, multi_measurement

    for i in range(1, 7):

        for j in range(1, i + 1):
            a = QuantumFloat(j)
            b = QuantumFloat(i)
            c = QuantumFloat(i)

            h(a)
            h(b)

            c[:] = b

            inpl_adder(a, c)

            statevector_arr = a.qs.compile().statevector_array()
            angles = np.angle(
                statevector_arr[
                    np.abs(statevector_arr) > 1 / 2 ** ((a.size + b.size) / 2 + 1)
                ]
            )

            # Test correct phase behavior
            assert (
                np.sum(np.abs(angles)) < 0.1
            ), f"Quantum-quantum adder produced a faulty phase shift on input sizes, {i},{j}."

            mes_res = multi_measurement([a, b, c])

            for a, b, c in mes_res.keys():
                assert (a + b) % (
                    2**i
                ) == c, f"Quantum-quantum addition result was incorrect for input values {a} += {c} on input sizes, {i},{j}."

        if i < 6:
            for j in range(1, 2**i):
                a = QuantumFloat(i)
                b = QuantumFloat(i)

                h(a)

                b[:] = a

                inpl_adder(j, a)

                statevector_arr = a.qs.compile().statevector_array()
                angles = np.angle(
                    statevector_arr[
                        np.abs(statevector_arr) > 1 / 2 ** ((a.size) / 2 + 1)
                    ]
                )
                assert (
                    np.sum(np.abs(angles)) < 0.1
                ), f"Classical-quantum adder produced a faulty phase shift on input size {i}."

                mes_res = multi_measurement([a, b])

                for a, b in mes_res.keys():
                    assert (b + j) % (
                        2**i
                    ) == a, f"Classical-quantum addition result was incorrect for input values {a} += {c} on input size {i}."

    for i in range(1, 7):

        for j in range(1, i + 1):
            a = QuantumFloat(j)
            b = QuantumFloat(i)
            c = QuantumFloat(i)
            qbl = QuantumBool()

            h(qbl)
            h(a)
            h(b)

            c[:] = b

            with control(qbl):
                inpl_adder(a, c)

            statevector_arr = a.qs.compile().statevector_array()
            angles = np.angle(
                statevector_arr[
                    np.abs(statevector_arr) > 1 / 2 ** ((a.size + b.size) / 2 + 1)
                ]
            )
            assert (
                np.sum(np.abs(angles)) < 0.1
            ), f"Controlled quantum-quantum adder produced a faulty phase shift on input sizes, {i},{j}."

            mes_res = multi_measurement([a, b, c, qbl])

            for a, b, c, qbl in mes_res.keys():

                if qbl:
                    assert (a + b) % (
                        2**i
                    ) == c, f"Controlled quantum-quantum addition result was incorrect for input values {a} += {c} on input sizes, {i},{j}."
                else:
                    assert (
                        c == b
                    ), f"Controlled quantum-quantum addition behaviour was incorrect; an operation was performed without the control qubit in |1> state.Faulty input sizes: {i},{j}"

        if i < 6:
            for j in range(1, 2**i):
                a = QuantumFloat(i)
                b = QuantumFloat(i)
                qbl = QuantumBool()

                h(qbl)
                h(a)

                b[:] = a

                with control(qbl):
                    inpl_adder(j, a)

                statevector_arr = a.qs.compile().statevector_array()
                angles = np.angle(
                    statevector_arr[
                        np.abs(statevector_arr) > 1 / 2 ** ((a.size) / 2 + 1)
                    ]
                )
                assert (
                    np.sum(np.abs(angles)) < 0.1
                ), f"Controlled classical-quantum adder produced a faulty phase shift on input size {i}."

                mes_res = multi_measurement([a, b, qbl])

                for a, b, qbl in mes_res.keys():
                    if qbl:
                        assert (b + j) % (
                            2**i
                        ) == a, f"Controlled classical-quantum addition result was incorrect for input values {b} += {j} on input size, {i}."
                    else:
                        assert (
                            b == a
                        ), f"Controlled classical-quantum addition behaviour was incorrect; an operation was performed without the control qubit in |1> state. Faulty input sizes: {i}"


def batched_measurement(variables, backend, shots=None):
    """
    This functions facilitates the measurement of multiple :ref:`QuantumVariables <QuantumVariable>` with a :ref:`BatchedBackend`.

    Parameters
    ----------
    variables : list[:ref:`QuantumVariable`]
        A list of QuantumVariables.
    backend : :ref:`BatchedBackend`
        The backend to evaluate the compiled QuantumCircuits on.
    shots : int, optional
        The amount of shots to perform. The default is given by the backend used.

    Returns
    -------
    results : list[dict]
        The list of results.

    Examples
    --------

    We set up a BatchedBackend, which sequentially executes the QuantumCircuits
    on the Qrisp simulator.

    ::

        from qrisp import *
        from qrisp.interface import BatchedBackend

        def run_func_batch(batch):
            # Parameters
            # ----------
            # batch : list[tuple[QuantumCircuit, int]]
            #     The circuit and shot batch indicating the backend queries.

            # Returns
            # -------
            # results : list[dict[string, int]]
            #     The list of results.

            results = []
            for i in range(len(batch)):
                qc = batch[i][0]
                shots = batch[i][1]
                results.append(qc.run(shots = shots))

            return results

        # Set up batched backend
        bb = BatchedBackend(run_func_batch)

        a = QuantumFloat(4)
        b = QuantumFloat(3)
        a[:] = 1
        b[:] = 2
        c = a + b

        d = QuantumFloat(4)
        e = QuantumFloat(3)
        d[:] = 2
        e[:] = 3
        f = d + e

        batched_measurement([c,f], backend=bb)
        # Yields: [{3: 1.0}, {5: 1.0}]

    """

    import threading

    results = [0] * len(variables)

    def eval_measurement(qv, i):
        results[i] = qv.get_measurement(backend=backend, shots=shots)

    threads = []
    for i, var in enumerate(variables):
        thread = threading.Thread(
            target=eval_measurement,
            args=(
                var,
                i,
            ),
        )
        threads.append(thread)

    # Start the threads
    for thread in threads:
        thread.start()

    # Call the dispatch routine
    # The min_calls keyword will make it wait
    # until the batch has a size of number of variables
    backend.dispatch(min_calls=len(variables))

    # Wait for the threads to join
    for thread in threads:
        thread.join()

    # Inspect the results
    return results
