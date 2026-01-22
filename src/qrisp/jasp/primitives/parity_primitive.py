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

@parity_p.def_abstract_eval
def parity_abstract_eval(*measurements, expectation = 2):
    """
    Abstract evaluation for the parity primitive.
    
    Checks that inputs are boolean (measurement results) and returns a boolean scalar (the detector result).
    """
    for b in measurements:
        if not isinstance(b, ShapedArray) or not isinstance(b.dtype, np.dtypes.BoolDType):
            raise Exception(f"Tried to trace parity primitive with value {b} (permitted is boolean)")
    
    return ShapedArray((), bool)

def parity(*measurements, expectation = 2):
    r"""
    Computes the parity on a set of measurement results. This is equivalent to performing a multi-input XOR gate.

    The primary purpose of this function is to check if the parity of a set of measurements matches an expected value.
    This is common in quantum error correction (e.g., stabilizers) and logical observables.

    .. note::

        When used within a function decorated with :func:`~qrisp.misc.stim_tools.extract_stim`, this function 
        is translated into Stim's ``DETECTOR`` or ``OBSERVABLE_INCLUDE`` instructions in the generated circuit.
        
        * If ``expectation`` is 0 or 1, a ``DETECTOR`` instruction is created.
        * If ``expectation`` is 2, an ``OBSERVABLE_INCLUDE`` instruction is created.

    Parameters
    ----------
    *measurements : list[Tracer[boolean]]
        Variable length argument list of measurement results (typically outcomes of ``measure`` or similar operations).
    expectation : int, optional
        The expected value of the parity of the measurement results. 
        If set to 0 or 1, the return value indicates if the actual parity differs from this expectation. 
        If set to 2 (default), the function simply returns the calculated parity.

    Returns
    -------
    Tracer[boolean]
        A boolean value representing the result of the parity check.
        If ``expectation`` is 0 or 1: Returns ``True`` if the actual parity does not match the expectation (i.e., detector fired), and ``False`` otherwise.
        If ``expectation`` is 2: Returns the actual parity value (0 or 1).
    
    Examples
    --------
    
    >>> from qrisp import QuantumVariable, h, cx, measure, parity
    >>> qv = QuantumVariable(2)
    >>> h(qv[0])
    >>> cx(qv[0], qv[1])
    >>> m0 = measure(qv[0])
    >>> m1 = measure(qv[1])
    >>> # Check if parity is even (0). Should be True (0) for Bell state.
    >>> # Function returns False (0) if parity matches expectation (no error).
    >>> check = parity(m0, m1, expectation=0) 
    
    """
    return parity_p.bind(*measurements, expectation = expectation)

@parity_p.def_impl
def parity_implementation(*measurements):
    """
    Implementation of the parity primitive.
    
    Appends a ParityOperation to the QuantumCircuit.
    """
    res = 0
    for i in range(len(measurements)):
        res ^= measurements[i]
    return res

class ParityOperation(Operation):
    """
    Operation class representing a Stim parity instructions (DETECTOR or OBSERVABLE_INCLUDE).
    
    This operation is used to interface with Stim's DETECTOR and OBSERVABLE instructions during the
    conversion process. It acts as a placeholder in the Qrisp QuantumCircuit.
    """
    def __init__(self, num_inputs, expectation = 2):
        
        definition = QuantumCircuit(0, num_inputs + 1)
        self.expectation = expectation
        Operation.__init__(self, "parity", num_clbits = num_inputs + 1, definition = definition)