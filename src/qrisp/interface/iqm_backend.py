"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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
********************************************************************************/
"""

def IQMBackend(api_token, device_instance):
    
    from iqm.qiskit_iqm import IQMProvider, transpile_to_IQM
    import qiskit
    def run_func_iqm(qasm_str, shots, token = ""):
        
        server_url = "https://cocos.resonance.meetiqm.com/" + device_instance
        
        backend = IQMProvider(server_url, token = api_token).get_backend()
        qc = qiskit.QuantumCircuit.from_qasm_str(qasm_str)
        qc_transpiled = transpile_to_IQM(qc, backend)
        
        job = backend.run(qc_transpiled, shots=shots)
        import re
        counts = job.result().get_counts()
        new_counts = {}
        for key in counts.keys():
            counts_string = re.sub(r"\W", "", key)
            new_counts[counts_string] = counts[key]
            
        return new_counts
    
    from qrisp.interface import VirtualBackend
    return VirtualBackend(run_func_iqm)