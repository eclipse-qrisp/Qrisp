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

"""
Boolean Simulation Decorator
============================

This module provides the ``boolean_simulation`` decorator, which enables efficient
classical simulation of quantum programs that contain only boolean/classical logic.

The decorator transforms Jasp functions into pure JAX expressions, allowing them to
be JIT-compiled and executed without any quantum simulation overhead. This is 
particularly useful for:

1. **Verifying uncomputation correctness**: The simulation checks that all deleted
   qubits are properly uncomputed (in the |0⟩ state) and warns otherwise.

2. **Testing classical quantum algorithms at scale**: Algorithms like adders,
   multipliers, and other arithmetic circuits can be tested with large inputs.

3. **Performance benchmarking**: Since the simulation compiles to efficient JAX
   code, it can be used to benchmark the classical logic portion of algorithms.

Supported Operations:
- X, SWAP gates (unconditional bit operations)
- Controlled variants: CX, CCX, CSWAP, multi-controlled X, etc.
- Phase gates (Z, S, T, RZ, P): These are no-ops in classical simulation
- Measurement: Returns the current classical bit values

Unsupported Operations:
- Any gate creating superposition: H, RX, RY, SX, etc.
- These will raise an exception during simulation
"""

from typing import TYPE_CHECKING, Any, Callable

import jax.numpy as jnp
from jax import jit, Array
from jax.tree_util import tree_unflatten

from qrisp.jasp import make_jaspr, AbstractQubitArray, AbstractQubit

from qrisp.jasp.interpreter_tools.interpreters.cl_func_interpreter import (
    jaspr_to_cl_func_jaxpr,
)
from qrisp.jasp.interpreter_tools import Jlist, eval_jaxpr

from qrisp.jasp import Jaspr


def boolean_simulation(*func: Callable, bit_array_padding: int = 2**16) -> Callable:
    """
    Decorator to simulate Jasp functions containing only classical logic (like X, CX, CCX etc.).

    This decorator transforms the function into a JAX expression without any
    quantum primitives and leverages the JAX compilation pipeline to compile
    a highly efficient simulation.

    .. note::

        The ``boolean_simulation`` decorator will check if deleted
        :ref:`QuantumVariables <QuantumVariable>`
        have been properly uncomputed and submit a warning otherwise.
        It therefore provides a valuable tool for verifying the correctness
        of your algorithms at scale.

    Parameters
    ----------
    func : Callable
        A Python function performing Jasp logic.
    bit_array_padding : int, optional
        An integer specifying the size of the classical array containing the
        (classical) bits which are simulated. Since JAX doesn't allow dynamically
        sized arrays but Jasp supports dynamically sized QuantumVariables, the
        array has to be "padded". The padding therefore indicates an upper boundary
        for how many qubits are required to execute ``func``. A large padding
        slows down the simulation but prevents overflow errors.The default is
        ``2**16``. The minimum value is 64. This threshold describes the maximum
        OVERALL amount of qubits that can appear in the simulation. The maximum
        amount per QuantumVariable/per QuantumArray is tied to this number
        and will always be ``1/64`` of the ``bit_array_padding``.
        The default here is therefore ``2**16/2**6 = 1024``.

    Returns
    -------
    Callable
        A function performing the simulation for the given input parameters.

    Raises
    ------
    Exception
        If ``bit_array_padding`` is less than 64.
    Exception
        If the function returns a quantum value (must measure before returning).

    Examples
    --------
    We create a simple script that demonstrates the functionality:

    ::

        from qrisp import QuantumFloat, measure
        from qrisp.jasp import boolean_simulation, jrange

        @boolean_simulation
        def main(i, j):

            a = QuantumFloat(10)

            b = QuantumFloat(10)

            a[:] = i
            b[:] = j

            c = QuantumFloat(30)

            for i in jrange(150):
                c += a*b

            return measure(c)

    This script evaluates the multiplication of the two inputs 150 times and adds
    them into the same QuantumFloat. The respected result is therefore ``i*j*150``.

    >>> main(1,2)
    Array(300., dtype=float64)
    >>> main(3,4)
    Array(1800., dtype=float64)

    Next we demonstrate the behavior under a faulty uncomputation:

    ::

        @boolean_simulation
        def main(i):

            a = QuantumFloat(10)
            a[:] = i
            a.delete()
            return


    >>> main(0)
    >>> main(1)
    WARNING: Faulty uncomputation found during simulation.
    >>> main(3)
    WARNING: Faulty uncomputation found during simulation.
    WARNING: Faulty uncomputation found during simulation.

    For the first case, the deletion is valid, because ``a`` is initialized in the
    $\\ket{0}$ state. For the second case, the first qubit is in the $\\ket{1}$
    state, so the deletion is not valid. The third case has both the first and
    the second qubit in the $\\ket{1}$ state (because 3 = ``11`` in binary) so
    there are two warnings.

    **Padding**

    We demonstrate the effects of the padding feature. For this we recreate the
    above script but with different padding selections.

    ::

        @boolean_simulation(bit_array_padding = 64)
        def main(i, j):

            a = QuantumFloat(10)

            b = QuantumFloat(10)

            a[:] = i
            b[:] = j

            c = QuantumFloat(30)

            for i in jrange(150):
                c += a*b

            return measure(c)

    >>> main(1,2)
    Array(8.92323439e+08, dtype=float64)

    A faulty result because the script needs more than 64 qubits.

    Increasing the padding ensures that enough qubits are available at the cost
    of simulation speed.
    """
    # Handle both @boolean_simulation and @boolean_simulation(...) syntax
    if len(func) == 0:
        # Called with arguments: @boolean_simulation(bit_array_padding=...)
        return lambda x: boolean_simulation(x, bit_array_padding=bit_array_padding)
    else:
        # Called without arguments: @boolean_simulation
        func = func[0]

    if bit_array_padding < 64:
        raise Exception(
            "Tried to initialize boolean_simulation with less than 512 bits"
        )

    @jit
    def return_function(*args: Any) -> Any:
        """
        JIT-compiled simulation function that transforms and executes the quantum program.

        This inner function:
        1. Creates a Jaspr from the decorated function (with output tree structure)
        2. Validates that no quantum values are returned (must be measured first)
        3. Transforms the Jaspr to a classical Jaxpr
        4. Initializes the boolean quantum circuit state (bit array + free qubit pool)
        5. Executes the classical Jaxpr and reconstructs the PyTree structure

        Parameters
        ----------
        *args : Any
            Arguments to pass to the original function.

        Returns
        -------
        Any
            The result of the simulation, with the original PyTree structure preserved:
            - None if the function has no return value
            - dict, list, tuple, or nested structure matching the original return type
        """
        # Create the Jaspr representation of the quantum program
        # Use return_shape=True to capture the output PyTree structure
        jaspr: Jaspr
        jaspr, out_tree = make_jaspr(func, return_shape=True)(*args)

        # Validate that no quantum values are returned
        # (quantum values must be measured before returning from boolean simulation)
        for var in jaspr.outvars:
            if isinstance(var.aval, (AbstractQubitArray, AbstractQubit)):
                raise Exception(
                    "Tried to perform boolean simulation of a function returning "
                    "a quantum value (please measure before returning)"
                )

        # Transform the Jaspr to a classical Jaxpr
        # flatten_environments() resolves any nested control flow structures
        cl_func_jaxpr = jaspr_to_cl_func_jaxpr(
            jaspr.flatten_environments(), bit_array_padding
        )

        # Initialize the boolean quantum circuit representation:
        # - bit_array: packed uint64 array storing qubit states (all zeros initially)
        # - free_qubit_list: Jlist containing all available qubit indices
        aval = cl_func_jaxpr.jaxpr.invars[-3].aval
        bit_array: Array = jnp.zeros(aval.shape, dtype=aval.dtype)

        # Create the free qubit pool as a flattened Jlist
        free_qubit_list = Jlist(
            jnp.arange(bit_array_padding), max_size=bit_array_padding
        ).flatten()[0]

        # The boolean quantum circuit is represented as a tuple:
        # (bit_array, jlist_array, jlist_counter)
        boolean_quantum_circuit: tuple[Array, ...] = (bit_array, *free_qubit_list)

        # Combine user arguments with the quantum circuit state
        ammended_args: list[Any] = list(args) + list(boolean_quantum_circuit)

        # Execute the classical Jaxpr
        res = eval_jaxpr(cl_func_jaxpr)(*ammended_args)

        # Extract the return values from the result
        # The result always contains 3 trailing elements for quantum circuit state:
        # (bit_array, jlist_array, jlist_counter)
        # The leading elements are the actual return values from the user's function.
        num_output_values = len(res) - 3

        if num_output_values == 0:
            # Function returns nothing (only quantum circuit state)
            return None
        else:
            # Extract user return values and reconstruct the PyTree structure
            flat_results = res[:num_output_values]
            return tree_unflatten(out_tree, flat_results)

    return return_function
