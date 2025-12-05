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

from qrisp.interface import BatchedBackend

def IQMBackend(api_token, 
               device_instance = None, 
               server_url = None, 
               compilation_options = None, 
               transpiler = None):
    """
    This function creates a :ref:`BatchedBackend` for executing circuits on IQM hardware.

    Parameters
    ----------
    api_token : str
        An API token retrieved from the IQM Resonance website or IQM backend.
    device_instance : str
        The device instance of the IQM backend such as ``garnet``.
        For an up-to-date list, see the IQM Resonance website.
        Required if server_url is not provided.
    server_url : str, optional
        The server URL of the IQM backend. If not provided, it defaults to IQM resonance
        using the device_instance. If a server URL is provided, a device instance should not be provided.
    compilation_options: CircuitCompilationOptions
        An object to specify several options regarding `pulse-level compilation <https://docs.meetiqm.com/iqm-client/api/iqm.iqm_client.models.CircuitCompilationOptions.html>`_.
    transpiler : callable, optional
        A function receiving and returning a QuantumCircuit, mapping the given
        circuit to a hardware friendly circuit. By default the `transpile_to_iqm <https://iqm-finland.github.io/qiskit-on-iqm/api/iqm.qiskit_iqm.iqm_naive_move_pass.transpile_to_IQM.html>`_
        function will be used.

    Examples
    --------

    We evaluate a :ref:`QuantumFloat` multiplication on the 20-qubit IQM Garnet.

    >>> from qrisp.interface import IQMBackend
    >>> qrisp_garnet = IQMBackend(api_token = "YOUR_IQM_RESONANCE_TOKEN", device_instance = "garnet")
    >>> from qrisp import QuantumFloat
    >>> a = QuantumFloat(2)
    >>> a[:] = 2
    >>> b = a*a
    >>> b.get_measurement(backend = qrisp_garnet, shots = 1000)
    {4: 0.548,
     5: 0.082,
     0: 0.063,
     6: 0.042,
     8: 0.031,
     2: 0.029,
     12: 0.014,
     10: 0.03,
     1: 0.027,
     7: 0.025,
     15: 0.023,
     9: 0.021,
     14: 0.021,
     13: 0.018,
     11: 0.014,
     3: 0.012}
    
    **Manual qubit selection and routing**
    
    In the next example we showcase how to prevent automatic selection of qubits.
    For this we overide the transpilation procedure. The default transpilation
    calls the ``transpile_to_IQM`` function, which performs routing,
    automatic selection of suitable qubits, basis-gate transformation and other
    optimizations.
    
    For our example we will just transform to the required basis gates and ensure
    manually that our circuit has the correct connectivity.
    
    ::
        
        from qrisp import QuantumCircuit
        
        # The custom transpiler should receive and return a QuantumCircuit
        def custom_transpiler(qc: QuantumCircuit) -> QuantumCircuit:
            return qc.transpile(basis_gates = ["cz", "r", "measure", "reset"])

        custom_transpiled_garnet = IQMBackend("YOUR_IQM_RESONANCE_TOKEN", 
                                   device_instance = "garnet",
                                   transpiler = custom_transpiler)
        
        # Create a bell state
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(0)

        # execute
        meas_res = qc.run(shots = 10000, backend = custom_transpiled_garnet)

    """
    if not isinstance(api_token, str):
        raise TypeError(
            "api_token must be a string. You can create an API token on the IQM Resonance website."
        )

    # Validate that either server_url or device_instance is provided, but not both
    if server_url is not None and device_instance is not None:
        raise ValueError(
            "Please provide either a server_url or a device_instance, but not both."
        )
    
    if server_url is None and device_instance is None:
        raise ValueError(
            "Please provide either a server_url or a device_instance."
        )

    if device_instance is not None and not isinstance(device_instance, str):
        raise TypeError(
            "device_instance must be a string. You can retrieve a list of available devices on the IQM Resonance website."
        )
        
    if server_url is not None and not isinstance(server_url, str):
        raise TypeError(
            "server_url must be a string."
        )

    try:
        from iqm.iqm_client.iqm_client import IQMClient
        from iqm.iqm_client import CircuitCompilationOptions
        from iqm.qiskit_iqm.iqm_provider import IQMBackend
        from iqm.qiskit_iqm import transpile_to_IQM
    except ImportError:
        raise ImportError(
            "Please install qiskit-iqm to use the IQMBackend. You can do this by running `pip install qrisp[iqm]`."
        )

    # Construct the server URL based on device_instance if server_url is not provided
    if server_url is None:
        server_url = "https://resonance.meetiqm.com/"
        
    client = IQMClient(iqm_server_url = server_url, token = api_token, quantum_computer = device_instance)
    backend = IQMBackend(client)
    
    if compilation_options is None:
        compilation_options = CircuitCompilationOptions()
        
    if transpiler is None:
        def transpiler(qc):
            from qrisp import QuantumCircuit
            qiskit_qc = qc.to_qiskit()
            transpiled_qiskit_qc = transpile_to_IQM(qiskit_qc, backend)
            return QuantumCircuit.from_qiskit(transpiled_qiskit_qc)

    def run_batch_iqm(batch):
        
        circuit_batch = []
        shot_batch = []
        for qc, shots in batch:
            if device_instance == "sirius":
                qiskit_qc = transpile_to_IQM(qc.to_qiskit(), backend)
            else:
                transpiled_qc = transpiler(qc)
                qiskit_qc = transpiled_qc.to_qiskit()
            circuit_batch.append(backend.serialize_circuit(qiskit_qc))
            if shots is None:
                shots = 1000
            
            shot_batch.append(shots)
            
        

        job = client.submit_circuits(circuit_batch, 
                                      options = compilation_options, 
                                      shots = max(shot_batch))
        
        
        job.wait_for_completion()
        answer = job.result()
        
        import re
        
        counts_batch = []
        for i in range(len(batch)):
            counts = answer[i]
        
            counts_dic = {}
            
            shots = batch[i][1]
            if shots is None:
                shots = 1000
            
            for j in range(shots):
                
                key_str = ""
                
                for k in counts.keys():
                    key_str += str(counts[k][j][0])
                
                if key_str in counts_dic:
                    counts_dic[key_str] +=1
                else:
                    counts_dic[key_str] =1
                    
            counts_batch.append(counts_dic)
    
        return counts_batch

    return BatchedBackend(run_batch_iqm)
