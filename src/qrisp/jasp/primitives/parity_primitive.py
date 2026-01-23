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

def parity(*measurements, expectation = None):
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
    keyword argument. This argument can be either ``True``,``False`` or ``None``. 
    The default is None.
    
    If the expectation argument is given, the parity function returns.
    
    .. math::
        
        p = x_{\text{exp}} \oplus \left( \bigoplus_{i=0}^{n-1} x_i \right)

    This implies that the parity function returns ``True`` if the expectation
    is NOT met.

    .. note::

        When used within a function decorated with :func:`~qrisp.misc.stim_tools.extract_stim`, this function 
        is translated into Stim's ``DETECTOR`` or ``OBSERVABLE_INCLUDE`` instructions in the generated circuit.
        
        * If ``expectation`` is ``True`` or ``False``, a ``DETECTOR`` instruction is created.
        * If ``expectation`` is ``None``, an ``OBSERVABLE_INCLUDE`` instruction is created.

        If executed **without** ``extract_stim`` (i.e., in regular Qrisp simulation via :func:`~qrisp.jasp.jaspify`, for instance),
        the function verifies that the measured parity matches the ``expectation`` provided. 
        Deviations from the parity expectation should solely stem from hardware noise,
        i.e. a deviation in a **noiseless** simulation is an error *in the programm*.
        Because of this, an Exception is raised when the verification fails. 
        However, when extracted to Stim, this verification is NOT performed 
        during the extraction or simulation process (Stim detectors record 
        differences but do not stop execution).


    Parameters
    ----------
    *measurements : list[Tracer[boolean]]
        Variable length argument list of measurement results (typically outcomes of ``measure`` or similar operations).
    expectation : None | bool, optional
        The expected value of the parity of the measurement results.
        If set to ``True`` or ``False``, the return value indicates if the actual parity 
        differs from this expectation. 
        If set to None (default), the function simply returns the calculated parity.

    Returns
    -------
    Tracer[boolean]
        A boolean value representing the parity.
    
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
    
    """
    if expectation is None:
        expectation = 2
    else:
        expectation = int(expectation)
    
    return parity_p.bind(*measurements, expectation = expectation)

@parity_p.def_impl
def parity_implementation(*measurements, expectation):
    """
    Implementation of the parity primitive.
    
    Appends a ParityOperation to the QuantumCircuit.
    """
    res = sum(measurements)%2
    if expectation != 2:
        if expectation != res:
            raise Exception("Parity expectation deviated from simulation result")
    return (res + expectation)%2

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