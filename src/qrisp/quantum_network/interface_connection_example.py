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


from qrisp.quantum_network.qn_server import QuantumNetworkServer
from qrisp.quantum_network.quantum_network_session import QuantumNetworkSession
from qrisp import QuantumFloat, QuantumVariable, h, x
from qrisp import QuantumBool, QuantumChar

test = QuantumNetworkServer(socket_ip_address="127.0.0.1", background=True)
test.start()


charlie = QuantumNetworkSession(socket_ip="127.0.0.1", port=7070, name="charlie")
bob = QuantumNetworkSession(socket_ip="127.0.0.1", port=7070, name="bob")

print("Server creation done")
qf_1 = QuantumFloat(3, qs=bob)
qf_2 = QuantumFloat(3, qs=bob)

qf_1[:] = 2
qf_2[:] = 3

qf_3 = qf_2 * qf_1

qf_3 += 3

print("Compilation done")

bob.send_qv(qf_3, "charlie")


print("Communication done")

inbox = charlie.update()

print(inbox[0])


q_ch = QuantumBool()
q_ch[:] = False

bob.send_qv(q_ch, "charlie")

inbox = charlie.update()

print(inbox[0])
