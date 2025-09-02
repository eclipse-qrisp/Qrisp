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


class PlanQKBackend(VirtualBackend):
    """
    This class instantiates a :ref:`VirtualBackend` using an PlanQK backend. It is intended to be used for Qiskit Devices.
    This allows easy access to AQT backends through the qrisp interface.

    Using this backend requires addtional dependencies: ``pip install --upgrade planqk-quantum``. 
    See the `PlanQK documentation website <https://docs.platform.planqk.de/sdk-reference.html>`_ for further infos.
    You need platform access to view the 
    `available backends <https://app.planqk.de/quantum-backends>`_  and `API token availability <https://platform.planqk.de/settings/access-tokens>`_ .
    

    Parameters
    ----------
    api_token : str
        `An API token <https://platform.planqk.de/settings/access-tokens>` for the PlanQK service. Platform access required.
    device_instance : str
        The device instance for the PlanQK service such as "azure.ionq.aria" or "azure.ionq.simulator".
        For an up-to-date list, see the `available PlanQK backends <https://app.planqk.de/quantum-backends>`_. Platform access required.
    organization_id : str 
        The organization ID for a company or project.

    Examples
    --------

    We evaluate a :ref:`QuantumFloat` multiplication on the Azure IONQ Aria QPU.

    >>> from qrisp import QuantumFloat
    >>> from qrisp.interface import PlanQKIBMBackend
    >>> qrisp_ibm = PlanQKIBMBackend(api_token = "YOUR_PLANQK_TOKEN", device_instance = "azure.ionq.aria", organization_id = "YOUR_organization_id")
    >>> a = QuantumFloat(2)
    >>> a[:] = 2
    >>> b = a*a
    >>> b.get_measurement(backend = qrisp_ibm, shots = 100)
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

    def __init__(self, api_token, device_instance, organization_id=None):

        if not isinstance(api_token, str):
            raise TypeError(
                "api_token must be a string. You can create an API token on the PlanQK website."
            )
        
        if organization_id:
            if not isinstance(organization_id, str):
                raise TypeError(
                    "organization_id must be a string."
                )

        if not isinstance(device_instance, str):
            raise TypeError(
                "Please provide a device_instance as a string. You can retrieve a list of available devices on the PlanQK website."
            )

        try:
            from planqk.quantum.sdk import PlanqkQuantumProvider
        except ImportError:
            raise ImportError(
                "Please install planqk-quantum to use the PlanQK service. You can do this by running `pip install --upgrade planqk-quantum`."
            )
        
        try:
            from qiskit import QuantumCircuit, transpile
        except ImportError:
            raise ImportError(
                "Please install qiskit to use the PlanQK service. You can do this by running `pip install qiskit`."
            )
        
        if  organization_id:
            provider = PlanqkQuantumProvider(organization_id=organization_id, access_token=api_token)
        else:
            provider = PlanqkQuantumProvider(access_token=api_token)

        backend = provider.get_backend(device_instance)
        #provider = AQTProvider(api_token)
        

        def run_planqk(qasm_str, shots, token):

            # Create the run method
            if shots is None:
                shots = 1000
            # Convert to qiskit
            from qiskit import QuantumCircuit

            qiskit_qc = QuantumCircuit.from_qasm_str(qasm_str)

            # Make circuit with one monolithic register
            new_qiskit_qc = QuantumCircuit(len(qiskit_qc.qubits), len(qiskit_qc.clbits))
            for instr in qiskit_qc:
                new_qiskit_qc.append(
                    instr.operation,
                    [qiskit_qc.qubits.index(qb) for qb in instr.qubits],
                    [qiskit_qc.clbits.index(cb) for cb in instr.clbits],
                )

            from qiskit import transpile
            qiskit_qc = transpile(new_qiskit_qc, backend=backend)
            job = backend.run([qiskit_qc], shots=shots)     
            result = job.result()
            planqk_result = result.results[0].data.probabilities
            
            # Remove the spaces in the qiskit result keys
            result_dic = {}
            import re

            for key in planqk_result.keys():
                counts_string = re.sub(r"\W", "", key)
                result_dic[counts_string] = planqk_result[key]

            return result_dic  

        super().__init__(run_planqk)

    def close_session(self):
        """
        Method to call in order to close the session started by the init method.
        """
        self.session.close()



