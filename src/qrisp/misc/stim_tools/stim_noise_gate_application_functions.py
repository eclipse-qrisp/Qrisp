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

import stim
from qrisp.core import append_operation
from qrisp.misc.stim_tools.error_class import StimNoiseGate

def stim_noise(stim_name, *parameters_and_qubits, pauli_string = None):
    """
    Applies a ``StimNoiseGate`` to the given qubits.

    For a list of supported error instructions, please check `Stims gate reference <https://github.com/quantumlib/Stim/blob/main/doc/gates.md#noise-channels>`_.
    
    Some of the most common errors are listed below:

    .. list-table::
       :widths: 25 75
       :header-rows: 1

       * - Name
         - Description
       * - ``DEPOLARIZE1``
         - Single qubit depolarizing noise. This channel applies one of the Pauli errors X, Y, Z with probability $p/3$.
       * - ``DEPOLARIZE2``
         - Two qubit depolarizing noise. This channel applies one of the 15 non-identity two-qubit Pauli errors (IX, IY, ..., ZZ) with probability $p/15$.
       * - ``X_ERROR``
         - Single qubit Pauli-X error (Bit flip). Applies X with probability $p$.
       * - ``Y_ERROR``
         - Single qubit Pauli-Y error. Applies Y with probability $p$.
       * - ``Z_ERROR``
         - Single qubit Pauli-Z error (Phase flip). Applies Z with probability $p$.
       * - ``PAULI_CHANNEL_1``
         - Custom single qubit Pauli channel. Takes 3 arguments (px, py, pz) specifying the probabilities of applying X, Y, and Z errors respectively.
       * - ``PAULI_CHANNEL_2``
         - Custom two qubit Pauli channel. Takes 15 arguments specifying the probabilities of applying each of the 15 non-identity two-qubit Pauli errors.
       * - ``E`` (or ``CORRELATED_ERROR``)
         - Correlated Pauli error on multiple qubits (requires ``pauli_string`` argument). Applies the specified Pauli string with probability $p$.
       * - ``ELSE_CORRELATED_ERROR``
         - Similar to ``CORRELATED_ERROR`` but only applies if the *previous* error instruction did NOT apply an error. This allows constructing more complex conditional error models.

    .. warning::

        Every noisy operation described here behaves as a purely unitary identity gate, unless the 
        compilation target is indeed Stim (see :meth:`~qrisp.QuantumCircuit.to_stim`). This means for 
        instance that :meth:`~qrisp.QuantumCircuit.to_qiskit` converts the noisy operations to trivial 
        identity gates. The same applies to the behavior of the Qrisp simulator. In other words - the 
        noisy operations will only behave noisy if pushed through the Stim compiler.


    Parameters
    ----------
    stim_name : str
        The name of the Stim error gate (e.g. ``DEPOLARIZE1``, ``X_ERROR``, ``CORRELATED_ERROR``).
    *parameters : float
        The parameters of the error channel (e.g. error probability). Further 
        details about the semantics of the parameters can be found in the
        `Stims gate reference <https://github.com/quantumlib/Stim/blob/main/doc/gates.md#noise-channels>`_.
    *qubits : Qubit
        The qubits to apply the error channel to.
    pauli_string : str, optional
        A string of Pauli operators (e.g. ``XX``, ``IZ``, ``Y``) characterizing the error. This is required for correlated errors (e.g. ``E``, ``CORRELATED_ERROR``).

    Examples
    --------

    We construct a noisy Bell-pair using the :func:`~qrisp.jasp.extract_stim` decorator.
        
    ::
        
        from qrisp import *
        from qrisp.misc.stim_tools import stim_noise

        @extract_stim
        def generate_noisy_bell_pair():
            
            qv = QuantumVariable(2)
            
            h(qv[0])
            cx(qv[0], qv[1])
            
            # Add single qubit noise
            stim_noise("X_ERROR", 0.1, qv[0])
            stim_noise("X_ERROR", 0.1, qv[1])
            
            # Add correlated multi-qubit noise
            stim_noise("E", 0.1, qv[0], qv[1], pauli_string = "XX")
            
            return measure(qv)

        # Generate result indices and stim circuit
        res_indices, stim_circuit = generate_noisy_bell_pair()
        
        # Compile sampler and sample
        sampler = stim_circuit.compile_sampler()
        all_samples = sampler.sample(1000)
        
        # Extract results through slicing
        samples = all_samples[:, res_indices]
        
        print(samples)
        
        # Yields:
        # array([[False,  True],
        #        [False,  True],
        #        [ True,  True],
        #        ...,
        #        [ True,  True],
        #        [ True,  True],
        #        [False, False]], shape=(1000, 2))


    """
    
    error_data = stim.gate_data(stim_name)
    
    if pauli_string is not None:
        # Check for compatibility
        if not (stim_name in ["E", "CORRELATED_ERROR", "ELSE_CORRELATED_ERROR"]):
             raise Exception(f"Stim error {stim_name} does not support Pauli strings. Supported gates are E, CORRELATED_ERROR, ELSE_CORRELATED_ERROR")

        num_qubits = len(pauli_string)

    elif error_data.is_single_qubit_gate:
        num_qubits = 1
    elif error_data.is_two_qubit_gate:
        num_qubits = 2
    else:
        raise Exception(f"Could not determine qubit amount for Stim error {stim_name}. Please check if the error is supported.")
    
    params = parameters_and_qubits[:-num_qubits]
    qubits = parameters_and_qubits[-num_qubits:]
    
    error_op = StimNoiseGate(stim_name, *params, pauli_string=pauli_string)
    
    append_operation(error_op, qubits = qubits)


