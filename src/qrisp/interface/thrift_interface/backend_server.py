"""
/********************************************************************************
* Copyright (c) 2023 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2 
* or later with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0
********************************************************************************/
"""


import threading

from thrift.protocol import TBinaryProtocol
from thrift.transport import TSocket, TTransport

from qrisp.interface.thrift_interface.stoppable_thrift_server import (
    StoppableThriftServer,
)
from qrisp.interface.thrift_interface.codegen import BackendService


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


# This class describes a Backend Server
class BackendServer:
    """
    This class allows convenient setup of a server respecting the Qrisp interface.

    Parameters
    ----------
    run_func : function
        A function that receives a QuantumCircuit, an integer specifying the amount of
        shots and a token in the form of a string. It returns the counts as a dictionary
        of bitstrings.
    socket_ip_address : str, optional
        The IP address of where the listening socket should be opened. By default, the
        IP address of the hosting machine will be used.
    port : int, optional
        The port on which to listen for requests. By default, 9090 will be used.
    name : str, optional
        A name for the server. The default is "generic_quantum_backend_server".
    ping_func : function, optional
        A function returning a BackendStatus object. The default is None.
    online_status : bool, optional
        A bool specifying whether the server should be able to handle requests directly
        after startup. The default is True.


    Examples
    --------

    We create a server listening on the localhost IP address using a run function which
    prints the token and queries the QASM-simulator. ::



        def run_func(qc, shots, token):

            #Convert to qiskit
            from qrisp.interface.circuit_converter import convert_circuit
            qiskit_qc = convert_circuit(qc, "qiskit")

            print(token)

            from qiskit import Aer
            qiskit_backend = Aer.get_backend('qasm_simulator')

            #Run Circuit on the Qiskit backend
            return qiskit_backend.run(qiskit_qc, shots = shots).result().get_counts()

        from qrisp.interface import BackendServer
        example_server = BackendServer(run_func, socket_ip_address = "127.0.0.1", port = 8080)
        example_server.start()


    """

    def __init__(
        self,
        run_func,
        socket_ip_address=None,
        port=None,
        transport=None,
        name="generic_quantum_backend_server",
        ping_func=None,
        online_status=True,
    ):
        if socket_ip_address is None:
            socket_ip_address = get_ip()

        self.transport = transport

        self.socket_ip_address = socket_ip_address

        self.online_status = online_status

        self.run_func = run_func

        if ping_func is None:

            def ping_func():
                from qrisp.interface import BackendStatus

                return BackendStatus(name=name)

        self.ping_func = ping_func

        self.name = name

        self.port = port

    # Starts the server
    def start(self):
        """
        Starts the server.
        """

        if self.port is None:
            self.port = 9090

        self.thread_name = self.name + ":" + str(self.port)

        # Create the BackendServiceHandler class
        ping_func = self.ping_func
        run_func = self.run_func
        pass_online_status_bool_by_reference = lambda: self.online_status

        class BackendServiceHandler:
            def run(self, qc, shots, token):
                return run_func(qc, shots, token)

            def ping(self):
                status = ping_func()
                status.online = pass_online_status_bool_by_reference()
                return status

        # Create server
        handler = BackendServiceHandler()
        processor = BackendService.Processor(handler)
        tfactory = TTransport.TBufferedTransportFactory()

        if self.transport is None:
            self.transport = TSocket.TServerSocket(
                host=self.socket_ip_address, port=self.port
            )

        pfactory = TBinaryProtocol.TBinaryProtocolFactory()

        # server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
        # self.server = TServer.TThreadedServer(processor, self.transport, tfactory, pfactory, daemon = True)

        # self.server = TNonblockingServer.TNonblockingServer(processor, self.transport, tfactory, pfactory)
        self.server = StoppableThriftServer(
            processor, self.transport, tfactory, pfactory, daemon=True
        )

        # Create thread
        self.thr = threading.Thread(target=self.server.serve, name=self.thread_name)
        self.thr.setDaemon(True)

        # Start the thread
        self.thr.start()

        while True:
            if self.server.is_running:
                break

    def stop(self):
        """
        Stops the server.
        """
        if hasattr(self, "server"):
            self.server.stop()
            self.thr.join()

    def __del__(self):
        self.stop()
