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
    Creates a Stim detector.

    A detector in Stim is an annotation that declares that a set of measurement outcomes has a deterministic parity (in the absence of noise). 
    In the context of quantum error correction, detectors typically correspond to operator measurements (stabilizers) which should yield a specific value (normally 0 for +1 eigenstates). 
    
    If the parity of the given measurements does not match the expected parity (which is implicitly 0), the detector "fires", indicating an error or an event.

    This function should be used within a function decorated with :func:`~qrisp.misc.stim_tools.extract_stim`. When the Qrisp circuit is converted to a Stim circuit, this operation is translated into a ``DETECTOR`` instruction targeting the measurement records corresponding to the input arguments.

    Parameters
    ----------
    *measurements : list[Tracer[boolean]]
        Variable length argument list of measurement results (typically outcomes of ``measure`` or similar operations).

    Returns
    -------
    Tracer[boolean]
        A boolean value representing whether the detector fired. In a noiseless simulation, this will be False (0). In a noisy simulation (handled by Stim), if the noise flips the parity of the measurements, this will be True (1).

    Examples
    --------
    
    **1. Selective Noise Indication**

    In this example, we setup two Bell-pairs. One is kept noiseless, while we insert a Pauli-X error into the other one.
    Because the Bell-states have the structure $|00\rangle + |11\rangle$, the parity $m_0 \oplus m_1$ is always 0.
    The X-error flips this parity and subsequently triggers the detector, which monitors the noisy pair.

    ::

        from qrisp import QuantumVariable, h, cx, measure, extract_stim
        from qrisp.misc.stim_tools import detector, stim_noise
        import stim

        @extract_stim
        def selective_noise_demo():
            # Create two QuantumVariables for independent Bell pairs
            bell_pair_1 = QuantumVariable(2)
            bell_pair_2 = QuantumVariable(2)
            
            # Initialize first Bell pair (Noiseless)
            h(bell_pair_1[0])
            cx(bell_pair_1[0], bell_pair_1[1])
            
            # Initialize second Bell pair (Noisy)
            h(bell_pair_2[0])
            cx(bell_pair_2[0], bell_pair_2[1])
            
            # Apply deterministic X error to one of the qubits in the second pair
            # This will flip the parity of the measurement outcomes
            stim_noise("X_ERROR", 1.0, bell_pair_2[0])
            
            # Measure
            m1_0 = measure(bell_pair_1[0])
            m1_1 = measure(bell_pair_1[1])
            
            m2_0 = measure(bell_pair_2[0])
            m2_1 = measure(bell_pair_2[1])
            
            # Detector 1 checks parity of first pair. No noise -> Parity 0 -> Detector output 0 (False)
            d1 = detector(m1_0, m1_1)
            
            # Detector 2 checks parity of second pair. X error -> Parity 1 -> Detector output 1 (True)
            d2 = detector(m2_0, m2_1)
            
            return d1, d2

        d1, d2, stim_circuit = selective_noise_demo()
        
        sampler = stim_circuit.compile_detector_sampler()
        print(sampler.sample(1))
        # Yields [[False, True]]

    **2. Repetition Code Syndrome Measurement**

    We simulate a repetition code on 3 qubits, which encodes a single logical qubit. We prepare the logical $|+\rangle_L$ state (which is $|000\rangle + |111\rangle$).
    We then insert a stochastic X error on the participating qubits.
    Finally, we measure the stabilizer operators $Z_0 Z_1$ and $Z_1 Z_2$ by measuring the qubits survival state validation.

    ::

        @extract_stim
        def repetition_code_demo():
            qv = QuantumVariable(3)
            
            # Encode logical state |0> + |1> -> |000> + |111>
            h(qv[0])
            cx(qv[0], qv[1])
            cx(qv[0], qv[2])
            
            # Apply stochastic bit flip noise on the qubit (25% chance)
            stim_noise("X_ERROR", 0.25, qv[0])
            stim_noise("X_ERROR", 0.25, qv[1])
            stim_noise("X_ERROR", 0.25, qv[2])
            
            # Measure all data qubits
            m = [measure(qv[i]) for i in range(3)]
            
            # Check parity of neighbors (syndrome extraction)
            
            detector_1 = detector(m[0], m[1])
            detector_2 = detector(m[1], m[2])
            
            return detector_1, detector_2

        d1, d2, stim_circuit = repetition_code_demo()
        
        sampler = stim_circuit.compile_detector_sampler()
        print(sampler.sample(5))
        # Possible output:
        # [[False,  False],  # No error
        #  [True,   False],    # Error on first qubit qubit
        #  [True,   True],    # Error on middle qubit qubit
        #  ... ]
    
    """
    return parity_p.bind(*measurements, expectation = expectation)

@parity_p.def_impl
def parity_implementation(*measurements):
    """
    Implementation of the detector primitive.
    
    Appends a StimDetector operation to the QuantumCircuit.
    """
    res = 0
    for i in range(len(measurements)):
        res ^= measurements[i]
    return res

class ParityOperation(Operation):
    """
    Operation class representing a Stim detector.
    
    This operation is used to interface with Stim's DETECTOR instruction during the
    conversion process. It acts as a placeholder in the Qrisp QuantumCircuit.
    """
    def __init__(self, num_inputs, expectation = 2):
        
        definition = QuantumCircuit(0, num_inputs + 1)
        self.expectation = expectation
        Operation.__init__(self, "parity", num_clbits = num_inputs + 1, definition = definition)