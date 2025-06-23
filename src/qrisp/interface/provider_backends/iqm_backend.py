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

from qrisp.interface import VirtualBackend

def IQMBackend(api_token, device_instance, compilation_options = None):
    """
    This function instantiates an IQMBackend based on VirtualBackend
    using Qiskit and Qiskit-on-IQM.


    Parameters
    ----------
    api_token : str
        An API token retrieved from the IQM Resonance website.
    device_instance : str
        The device instance of the IQM backend such as "garnet".
        For an up-to-date list, see the IQM Resonance website.
    compilation_options: `CircuitCompilationOptions <https://docs.meetiqm.com/iqm-client/api/iqm.iqm_client.models.CircuitCompilationOptions.html>`.
        An object to specify several options regarding pulse-level compilation.

    Examples
    --------

    We evaluate a QuantumFloat multiplication on the 20-qubit IQM Garnet.

    >>> from qrisp.interface import IQMBackend
    >>> qrisp_garnet = IQMBackend(api_token = "YOUR_IQM_RESONANCE_TOKEN", device_instance = "garnet")
    >>> from qrisp import QuantumFloat
    >>> a = QuantumFloat(2)
    >>> a[:] = 2
    >>> b = a*a
    >>> b.get_measurement(backend = qrisp_garnet, shots=1000)
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

    """

    if not isinstance(api_token, str):
        raise TypeError(
            "api_token must be a string. You can create an API token on the IQM Resonance website."
        )

    if not isinstance(device_instance, str):
        raise TypeError(
            "Please provide a device_instance as a string. You can retrieve a list of available devices id on the IQM Resonance website."
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

    server_url = "https://cocos.resonance.meetiqm.com/" + device_instance
    client = IQMClient(url = server_url, token = api_token)
    backend = IQMBackend(client)
    
    if compilation_options is None:
        compilation_options = CircuitCompilationOptions()

    def run_func_iqm(qasm_str, shots=None, token=""):
        if shots is None:
            shots = 1000
        
        from qiskit import QuantumCircuit
        
        qiskit_qc = QuantumCircuit.from_qasm_str(qasm_str)
        qiskit_qc = transpile_to_IQM(qiskit_qc, backend)
        
        iqm_qc = backend.serialize_circuit(qiskit_qc)

        UUID = client.submit_circuits([iqm_qc], options = compilation_options, shots = shots)
        
        client.wait_for_results(UUID)
        answer = client.get_run_counts(UUID)
        
        counts = answer.counts_batch[0].counts

        import re
        new_counts = {}
        for key in counts.keys():
            counts_string = re.sub(r"\W", "", key)
            new_counts[counts_string] = counts[key]

        return new_counts

    return VirtualBackend(run_func_iqm, port=port)
