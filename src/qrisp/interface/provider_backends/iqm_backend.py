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


def IQMBackend(
    api_token,
    device_instance=None,
    server_url=None,
    compilation_options=None,
    transpiler=None,
):
    """
    This function instantiates an IQMBackend based on :ref:`VirtualBackend`
    using Qiskit and Qiskit-on-IQM.


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
    compilation_options: `CircuitCompilationOptions <https://docs.meetiqm.com/iqm-client/api/iqm.iqm_client.models.CircuitCompilationOptions.html>`_.
        An object to specify several options regarding pulse-level compilation.

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
        raise ValueError("Please provide either a server_url or a device_instance.")

    if device_instance is not None and not isinstance(device_instance, str):
        raise TypeError(
            "device_instance must be a string. You can retrieve a list of available devices on the IQM Resonance website."
        )

    if server_url is not None and not isinstance(server_url, str):
        raise TypeError("server_url must be a string.")

    try:
        from iqm.iqm_client import CircuitCompilationOptions
        from iqm.iqm_client.iqm_client import IQMClient
        from iqm.qiskit_iqm import transpile_to_IQM
        from iqm.qiskit_iqm.iqm_provider import IQMBackend
    except ImportError:
        raise ImportError(
            "Please install qiskit-iqm to use the IQMBackend. You can do this by running `pip install qrisp[iqm]`."
        )

    # Construct the server URL based on device_instance if server_url is not provided
    if server_url is None:
        server_url = "https://cocos.resonance.meetiqm.com/" + device_instance

    client = IQMClient(url=server_url, token=api_token)
    backend = IQMBackend(client)

    if compilation_options is None:
        compilation_options = CircuitCompilationOptions()

    if transpiler is None:
        transpiler = lambda qiskit_qc: transpile_to_IQM(qiskit_qc, backend)

    def run_batch_iqm(batch):

        circuit_batch = []
        shot_batch = []
        for qc, shots in batch:
            qiskit_qc = qc.to_qiskit()
            qiskit_qc = transpiler(qiskit_qc)
            
            circuit_batch.append(backend.serialize_circuit(qiskit_qc))
            if shots is None:
                shots = 1000

            shot_batch.append(shots)
            
        

        UUID = client.submit_circuits(
            circuit_batch, options=compilation_options, shots=max(shot_batch)
        )

        client.wait_for_results(UUID)

        answer = client.get_run_counts(UUID)
        import re

        counts_batch = []
        for i in range(len(batch)):
            counts = answer.counts_batch[i].counts

            new_counts = {}
            for key in counts.keys():
                counts_string = re.sub(r"\W", "", key)
                new_counts[counts_string[::-1]] = counts[key]

            counts_batch.append(new_counts)

        return counts_batch

    return BatchedBackend(run_batch_iqm)
