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

import qiskit
from qrisp.interface.virtual_backend import VirtualBackend


class AQTBackend(VirtualBackend):
    """
    This class instantiates a VirtualBackend using an AQT backend.
    This allows easy access to AQT backends through the qrisp interface.

    Parameters
    ----------
    backend : AQT backend object, optional
        An AQT backend object, which runs QuantumCircuits. The default is
        ``provider.get_backend("offline_simulator_no_noise")``.
    port : int, optional
        The port to listen. The default is None.

    Examples
    --------

    We evaluate a QuantumFloat multiplication on the noiseless offline-simulator.

    >>> from qrisp import QuantumFloat
    >>> from qrisp.interface import AQTBackend
    >>> from qiskit_aqt_provider import AQTProvider
    >>> provider = AQTProvider("ACCESS_TOKEN")
    >>> example_backend = AQTBackend(backend = provider.get_backend("offline_simulator_no_noise"))
    >>> qf = QuantumFloat(4)
    >>> qf[:] = 3
    >>> res = qf*qf
    >>> res.get_measurement(backend = example_backend)
    {9: 1.0}

    """

    try:
        from qiskit_aqt_provider import AQTProvider
        from qiskit_aqt_provider.primitives import AQTSampler
    except ImportError:
        raise ImportError(
            "Please install qiskit-aqt-provider to use the AQTBackend. You can do this by running `pip install qiskit-aqt-provider`."
        )

    def __init__(self, backend=None, port=None):
        if backend is None:

            # Select an execution backend.
            # Any token (even invalid) gives access to the offline simulator backends.
            provider = AQTProvider("ACCESS_TOKEN")
            backend = provider.get_backend("offline_simulator_no_noise")


        # Create the run method
        def run(qasm_str, shots=None, token=""):
            if shots is None:
                shots = 1000

            # Convert to qiskit
            qiskit_qc = qiskit.QuantumCircuit.from_qasm_str(qasm_str)

            # Make circuit with one monolithic register
            new_qiskit_qc = qiskit.QuantumCircuit(len(qiskit_qc.qubits), len(qiskit_qc.clbits))
            for instr in qiskit_qc:
                new_qiskit_qc.append(
                    instr.operation,
                    [qiskit_qc.qubits.index(qb) for qb in instr.qubits],
                    [qiskit_qc.clbits.index(cb) for cb in instr.clbits],
                )
            
            # Instantiate a sampler on the execution backend.
            sampler = AQTSampler(backend)

            # Optional: set the transpiler's optimization level.
            # Optimization level 3 typically provides the best results.
            sampler.set_transpile_options(optimization_level=3)

            # Sample the circuit on the execution backend.
            result = sampler.run(new_qiskit_qc).result()

            quasi_dist = result.quasi_dists[0]
           
            # Format to fit the qrisp result format
            result_dic = {}

            len_qc = len(new_qiskit_qc.qubits) # number of qubits
            for item in list(quasi_dist.keys()):

                # transform to binary, fill to given length, and then reverse 
                #new_key = bin(item)[2:].zfill(len_qc)[::-1] 
                new_key = bin(item)[2:].zfill(len_qc) 
                result_dic.setdefault(new_key, quasi_dist[item])

            return result_dic

        # Call VirtualBackend constructor
        if isinstance(backend.name, str):
            name = backend.name
        else:
            name = backend.name()

        super().__init__(run, port=port)

#def VirtualAQTBackend(*args, **kwargs):
#    import warnings
#    warnings.warn("Hardware Backend and Access Token TBD")
#    return AQTBackend(*args, **kwargs)


