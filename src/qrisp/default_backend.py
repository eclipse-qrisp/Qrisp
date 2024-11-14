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


from qrisp.interface import VirtualBackend, VirtualQiskitBackend
# from qrisp.interface.qunicorn import BackendClient
# def_backend = VirtualQiskitBackend()
from qrisp.simulator.simulator import run
from qrisp import QuantumCircuit
#

def run_wrapper(qasm_str, shots, token = ""):
    qc = QuantumCircuit.from_qasm_str(qasm_str)
    return run(qc, shots)

# def_backend = VirtualBackend(run_wrapper, port=8080)

class DefaultBackend:
    
    def run(self, qc, shots, token = ""):
        return run(qc, shots, token)


def_backend = DefaultBackend()
# def_backend = BackendClient()
# raise