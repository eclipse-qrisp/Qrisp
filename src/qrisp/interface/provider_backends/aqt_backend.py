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
from qrisp.interface import VirtualBackend, BatchedBackend


class AQTBackend(BatchedBackend):
    """
    This class instantiates a :ref:`VirtualBackend` using an AQT backend. 
    This allows easy access to AQT backends through the qrisp interface.


    Parameters
    ----------
    api_token : str
        An API token for `AQT ARNICA <https://www.aqt.eu/products/arnica/>`_.
    device_instance : str
        The device instance of the AQT backend such as "ibex" or "simulator_noise".
        For an up-to-date list, see the AQT ARNICA website.
    workspace : str 
        The workspace for a company or project.

    Examples
    --------

    We evaluate a :ref:`QuantumFloat` multiplication on the 12-qubit AQT IBEX.

    >>> from qrisp import QuantumFloat
    >>> from qrisp.interface import AQTBackend
    >>> # qrisp_ibex = AQTBackend(api_token="YOUR_AQT_ARNICA_TOKEN", device_instance = "simulator_noise", workspace="aqt_simulators")
    >>> qrisp_ibex = AQTBackend(api_token="YOUR_AQT_ARNICA_TOKEN", device_instance = "ibex", workspace="YOUR_COMPANY_OR_PROJECT_NAME")
    >>> a = QuantumFloat(2)
    >>> a[:] = 2
    >>> b = a*a
    >>> b.get_measurement(backend = qrisp_ibex, shots = 100)
    {4: 0.49,
    8: 0.11,
    2: 0.08,
    0: 0.06,
    14: 0.06,
    5: 0.04,
    12: 0.04,
    13: 0.03,
    3: 0.02,
    10: 0.02,
    15: 0.02,
    6: 0.01,
    7: 0.01,
    11: 0.01}

    """

    def __init__(self, api_token, device_instance, workspace):

        if not isinstance(api_token, str):
            raise TypeError(
                "api_token must be a string. You can create an API token on the AQT ARNICA website."
            )
        
        if not isinstance(workspace, str):
            raise TypeError(
                "workspace must be a string."
            )

        if not isinstance(device_instance, str):
            raise TypeError(
                "Please provide a device_instance as a string. You can retrieve a list of available devices on the AQT ARNICA website."
            )

        try:
            from qiskit_aqt_provider import AQTProvider
            from qiskit_aqt_provider.primitives import AQTSampler
        except ImportError:
            raise ImportError(
                "Please install qiskit-aqt-provider to use the AQTBackend. You can do this by running `pip install qiskit-aqt-provider`."
            )
        
        provider = AQTProvider(api_token)
        backend = provider.get_backend(name = device_instance, workspace = workspace)

        """
        def run(qasm_str, shots=None, token=""):
            if shots is None:
                shots = 100

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
            result = sampler.run(new_qiskit_qc, shots=shots).result()

            quasi_dist = result.quasi_dists[0]
            
            # Format to fit the qrisp result format
            result_dic = {}

            for item in list(quasi_dist.keys()):
                new_key = bin(item)[2:].zfill(len(qiskit_qc.clbits))
                result_dic.setdefault(new_key, quasi_dist[item])

            return result_dic
        """

        def run_batch_aqt(batch):
        
            circuit_batch = []
            shot_batch = []
            cl_bits_batch = []
            for qc, shots in batch:
                # Sometimes wrong results without transpilation 
                qiskit_qc = qc.transpile().to_qiskit()

                # Make circuit with one monolithic register
                new_qiskit_qc = qiskit.QuantumCircuit(len(qiskit_qc.qubits), len(qiskit_qc.clbits))
                for instr in qiskit_qc:
                    new_qiskit_qc.append(
                        instr.operation,
                        [qiskit_qc.qubits.index(qb) for qb in instr.qubits],
                        [qiskit_qc.clbits.index(cb) for cb in instr.clbits],
                    )

                circuit_batch.append(new_qiskit_qc)
                cl_bits_batch.append(len(qiskit_qc.clbits))

                if shots is None:
                    shots = 100
            
                shot_batch.append(shots)

            # Instantiate a sampler on the execution backend.
            sampler = AQTSampler(backend)

            # Optional: set the transpiler's optimization level.
            # Optimization level 3 typically provides the best results.
            sampler.set_transpile_options(optimization_level=3)

            # Sample the circuit on the execution backend.
            results = sampler.run(circuit_batch, shots = max(shot_batch)).result()

            quasi_dist_batch = []
            for i in range(len(batch)):
                quasi_dist = results.quasi_dists[i]
                cl_bits = cl_bits_batch[i]

                new_quasi_dist = {}
                for item in list(quasi_dist.keys()):
                    new_key = bin(item)[2:].zfill(cl_bits)
                    new_quasi_dist.setdefault(new_key, quasi_dist[item])

                quasi_dist_batch.append(new_quasi_dist)
    
            return quasi_dist_batch

        # Call BatchedBackend constructor
        if isinstance(backend.name, str):
            name = backend.name
        else:
            name = backend.name()

        super().__init__(run_batch_aqt)