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


import threading

from thrift.protocol import TBinaryProtocol
from thrift.server import TNonblockingServer, TServer
from thrift.transport import TSocket, TTransport

from qrisp import Qubit
from qrisp.interface import PortableQubit
from qrisp.interface.circuit_converter import convert_circuit
from qrisp.interface.thrift_interface.stoppable_thrift_server import (
    StoppableThriftServer,
)
from qrisp.quantum_network.interface.codegen.ttypes import Message
from qrisp.quantum_network.qn_simulator_server import QuantumNetwork
from qrisp.quantum_network.interface.codegen import QuantumNetworkService


# Returns the hosts ip
def get_ip():
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        # doesn't even have to be reachable
        s.connect(("10.255.255.255", 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = "127.0.0.1"
    finally:
        s.close()
    return IP


class QuantumNetworkHandler(QuantumNetwork):
    def run(self, qc, name):
        converted_circuit = convert_circuit(qc, "qrisp")
        return super().run(converted_circuit, name)

    def get_clear_qc(self, name):
        qc = super().get_clear_qc(name)

        return convert_circuit(qc, "thrift")

    def request_qubits(self, amount, name):
        qb_list = super().request_qubits(amount, name)

        return [PortableQubit(qb.identifier) for qb in qb_list]

    def send_qubits(self, sender, recipient, msg):
        super().send_qubits(
            sender,
            recipient,
            [Qubit(qb.identifier) for qb in msg.qubits],
            msg.annotation,
        )

    def get_overall_qc(self):
        qc = super().get_overall_qc()

        return convert_circuit(qc, "thrift")

    def inbox(self, name):
        inbox_list = super().inbox(name)

        res_list = []

        for i in range(len(inbox_list)):
            res_list.append(
                Message(
                    [PortableQubit(qb.identifier) for qb in inbox_list[i][0]],
                    inbox_list[i][1],
                )
            )

        return res_list


class QuantumNetworkServer:
    """
    This class sets up a server to coordinate the QuantumNetwork. All the simulations
    are performed in this instance, so the machine running it should preferably have
    abundant computational resources.

    Parameters
    ----------
    socket_ip : str, optional
        The IP address of the network socket. By default, the IP adress of the executing
        machine is used.
    port : int, optional
        The port to listen for requests. The default is 7070.
    background : bool, optional
        If set to True, the server will run in the background as a separate thread.
        The default is False.

    Examples
    --------

    We create a server listening to the localhost IP address:

    >>> from qrisp.quantum_network import QuantumNetworkServer
    >>> example_server = QuantumNetworkServer(socket_ip = "127.0.0.1")
    >>> example_server.start()

    """

    def __init__(self, socket_ip=None, port=None, background=False):
        if socket_ip is None:
            socket_ip = get_ip()

        self.socket_ip = socket_ip

        self.background = background
        self.port = port

    # Starts the server
    def start(self):
        """
        Starts the server.

        Returns
        -------
        None.

        """

        if self.port is None:
            self.port = 7070

        self.thread_name = "qn" + ":" + str(self.port)

        self.handler = QuantumNetworkHandler()

        processor = QuantumNetworkService.Processor(self.handler)
        tfactory = TTransport.TBufferedTransportFactory()

        self.transport = TSocket.TServerSocket(host=self.socket_ip, port=self.port)

        pfactory = TBinaryProtocol.TBinaryProtocolFactory()

        # server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
        # self.server = TServer.TThreadedServer(processor, self.transport, tfactory,
        # pfactory, daemon = True)
        self.server = StoppableThriftServer(
            processor, self.transport, tfactory, pfactory, daemon=True
        )

        # self.server = TNonblockingServer.TNonblockingServer(processor, self.transport,
        # tfactory, pfactory)

        if self.background:
            # Create thread
            self.thr = threading.Thread(target=self.server.serve, name=self.thread_name)
            self.thr.setDaemon(True)

            # Start the thread
            self.thr.start()
        else:
            self.server.serve()

    def stop(self):
        """
        Stops the server.

        Returns
        -------
        None.

        """
        if hasattr(self, "server"):
            self.server.stop()

    def __del__(self):
        self.handler.reset_network_state()
        self.stop()
