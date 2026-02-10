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

import numpy as np

from qrisp.circuit import Operation, QuantumCircuit
from qrisp.jasp.primitives.quantum_primitive import QuantumPrimitive

parity_p = QuantumPrimitive("parity")

from jax.core import ShapedArray
from jax.lax import while_loop
import jax.numpy as jnp

@parity_p.def_abstract_eval
def parity_abstract_eval(*measurements, expectation = 0, observable = False):
    """
    Abstract evaluation for the parity primitive.
    
    Checks that inputs are boolean (measurement results) and returns a boolean scalar (the detector result).
    """
    for b in measurements:
        if not isinstance(b, ShapedArray) or not isinstance(b.dtype, np.dtypes.BoolDType):
            raise Exception(f"Tried to trace parity primitive with value {b} (permitted is boolean)")
    
    return ShapedArray((), bool)

def parity(*measurements, expectation = 0, observable = False):
    r"""
    Computes the parity on a set of measurement results. This is equivalent to performing a multi-input XOR gate.
    
    In mathematical terms, if given the inputs $\{x_i \in \mathbb{F}_2\| 0 \leq i < n \}$
    the output of this function is therefore
    
    .. math::
        
        p = \bigoplus_{i=0}^{n-1} x_i

    A common usecase for this quanity are certain checks within quantum error 
    correction circuits. In this scenario, the programmer ensures that 
    a certain set of measurements has a deterministic parity. If this is not 
    satisfied during the execution, it indicates the presence of an error.
    
    This type of parity check is also called a "detector". The set of detector
    values together with the "detector error model" is then consumed by the decoder,
    providing an educated guess, which error caused the parity check to fail.
    
    Within Qrisp, the ``parity`` function can receive the ``expectation``
    keyword argument with values ``0`` or ``1``, and the ``observable`` keyword
    argument (boolean). The default is ``expectation=0, observable=False``.
    
    The parity function returns:
    
    .. math::
        
        p = x_{\text{exp}} \oplus \left( \bigoplus_{i=0}^{n-1} x_i \right)

    This implies that the parity function returns ``True`` if the expectation
    is NOT met.
    
    **Array Support**: If the measurement inputs are arrays, all arrays must have exactly
    the same shape (no broadcasting). The function applies the parity computation element-wise,
    returning an array of the same shape. Mixing scalar and array inputs is not allowed.

    .. note::

        When used within a function decorated with :func:`~qrisp.misc.stim_tools.extract_stim`, this function 
        is translated into Stim's ``DETECTOR`` or ``OBSERVABLE_INCLUDE`` instructions in the generated circuit.
        
        * If ``observable=False`` (default), a ``DETECTOR`` instruction is created.
        * If ``observable=True``, an ``OBSERVABLE_INCLUDE`` instruction is created.

        If executed **without** ``extract_stim`` (i.e., in regular Qrisp simulation via :func:`~qrisp.jasp.jaspify`, for instance),
        and ``observable=False``, the function verifies that the measured parity matches the ``expectation`` provided. 
        Deviations from the parity expectation should solely stem from hardware noise,
        i.e. a deviation in a **noiseless** simulation is an error *in the programm*.
        Because of this, an Exception is raised when the verification fails. 
        However, when extracted to Stim, this verification is NOT performed 
        during the extraction or simulation process (Stim detectors record 
        differences but do not stop execution).


    Parameters
    ----------
    *measurements : list[Tracer[boolean] | array[boolean]]
        Variable length argument list of measurement results (typically outcomes of ``measure`` or similar operations).
        Can be scalars or arrays. All array inputs must have exactly the same shape.
        Mixing scalar and array inputs is not allowed.
    expectation : int, optional
        The expected value of the parity of the measurement results.
        Must be ``0`` or ``1``. The return value indicates if the actual parity 
        differs from this expectation. Default is ``0``.
    observable : bool, optional
        If ``True``, this parity is treated as a Stim ``OBSERVABLE_INCLUDE`` 
        instruction rather than a ``DETECTOR``. Default is ``False``.

    Returns
    -------
    Tracer[boolean] | array[boolean]
        A boolean value representing the parity. Returns scalar if all inputs are scalars,
        otherwise returns an array with the broadcasted shape of the inputs.
    
    Examples
    --------
    
    We measure the parity of the 4 qubit GHZ state:
        
    .. math::
        
        \ket{\text{GHZ}} = \frac{\ket{0000} + \ket{1111}}{\sqrt{2}}
    
    ::
        
        from qrisp import *
        
        @jaspify
        def main():
            
            qv = QuantumVariable(4)
            h(qv[0])
            cx(qv[0], qv[1])
            cx(qv[0], qv[2])
            cx(qv[0], qv[3])
            
            a, b, c, d = tuple(measure(qv[i]) for i in range(4))
            return parity(a, b, c, d)
        
        print(main())
        # Yields 0
    
    Array example with element-wise parity:
    
    ::
        
        from qrisp import *
        import jax.numpy as jnp
        
        @jaspify
        def array_parity():
            # Create arrays of measurements
            a = jnp.array([True, False, True])
            b = jnp.array([False, False, True])
            
            # Compute element-wise parity
            result = parity(a, b)
            return result
        
        print(array_parity())
        # Yields [1, 0, 0]
    
    """
    import jax.numpy as jnp
    from jax import lax
    
    expectation = int(expectation)
    observable = bool(observable)
    
    # Check if any inputs are arrays
    shapes = [jnp.shape(m) for m in measurements]
    
    if all(s == () for s in shapes):
        # All scalars - direct call to primitive
        return parity_p.bind(*measurements, expectation=expectation, observable=observable)
    else:
        # At least one array - check that all arrays have the same shape
        # Collect all non-scalar shapes
        non_scalar_shapes = [s for s in shapes if s != ()]
        
        if not non_scalar_shapes:
            # This shouldn't happen given the if-else structure, but just in case
            return parity_p.bind(*measurements, expectation=expectation, observable=observable)
        
        # Check that all non-scalar shapes are identical
        first_shape = non_scalar_shapes[0]
        if not all(s == first_shape for s in non_scalar_shapes):
            raise ValueError(f"All array inputs to parity must have the same shape. Got shapes: {shapes}")
        
        # Also check that scalar and non-scalar inputs are not mixed
        if len(non_scalar_shapes) != len(shapes):
            raise ValueError(f"Cannot mix scalar and array inputs to parity. Got shapes: {shapes}")
        
        result_shape = first_shape
        flat_result = jnp.zeros(measurements[0].size, dtype = bool)
        
        # Flatten for element-wise processing
        flat_measurements = [jnp.ravel(m) for m in measurements]
        
        init_val = (0, flat_result, flat_measurements)
        
        def body_function(val):
            
            index, flat_result, flat_measurements = val
            parity_res = parity(*[meas[index] for meas in flat_measurements], 
                                expectation=expectation, observable=observable)
            flat_result = flat_result.at[index].set(parity_res)
            index += 1
            
            return index, flat_result, flat_measurements
        
        def cond_function(val):
            return val[0] < val[1].size

        while_res = while_loop(cond_function, body_function, init_val)
        
        flat_result = while_res[1]
        
        return jnp.reshape(flat_result, result_shape)

@parity_p.def_impl
def parity_implementation(*measurements, expectation, observable):
    """
    Implementation of the parity primitive.
    
    Handles only scalar inputs. Array broadcasting is handled in the parity function.
    """
    res = sum(measurements) % 2
    if not observable:
        if expectation != res:
            raise Exception("Parity expectation deviated from simulation result")
    return jnp.array((res + expectation) % 2, dtype = bool)

class ParityOperation(Operation):
    """
    Operation class representing a Stim parity instructions (DETECTOR or OBSERVABLE_INCLUDE).
    
    This operation is used to interface with Stim's DETECTOR and OBSERVABLE instructions during the
    conversion process. It acts as a placeholder in the Qrisp QuantumCircuit.
    
    The operation takes n input clbits (the measurements to compute parity of).
    Parity results are tracked via ParityHandle objects returned by jaspr.to_qc().
    """
    def __init__(self, num_inputs, expectation = 0, observable = False):
        
        definition = QuantumCircuit(0, num_inputs)
        self.expectation = expectation
        self.observable = observable
        Operation.__init__(self, "parity", num_clbits = num_inputs, definition = definition)