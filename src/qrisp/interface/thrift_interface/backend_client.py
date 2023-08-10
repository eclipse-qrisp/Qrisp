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


import sys

from qrisp.interface.circuit_converter import convert_circuit
from qrisp.interface.thrift_interface.codegen import BackendService


class BackendClient(BackendService.Client):
    """
    This object allows connecting to Qrisp backend servers.

    Parameters
    ----------
    socket_ip : string
        The IP address of the socket of the target server.
    port : int
        The port on which the server is listening.


    Examples
    --------

    We assume that the example from BackendServer has been executed in the same console.

    >>> from qrisp.interface import BackendClient
    >>> example_backend = BackendClient(socket_ip = "127.0.0.1", port = 8080)
    >>> from qrisp import QuantumCircuit
    >>> qc = QuantumCircuit(2)
    >>> qc.h(0)
    >>> qc.cx(0,1)
    >>> qc.measure(qc.qubits)
    >>> example_backend.run(qc, shots = 1000, token = "lorem ipsum")
    lorem ipsum
    {'0 0': 464, '1 1': 536}



    """

    def __init__(self, socket_ip, port=None):
        from thrift.protocol import TBinaryProtocol
        from thrift.transport import TSocket, TTransport

        if port is None:
            port = 9010
        # Create the transport for the User Interface to the server
        self.transport = TSocket.TSocket(socket_ip, port)
        # Buffering is critical. Raw sockets are very slow
        self.transport = TTransport.TBufferedTransport(self.transport)
        # Wrap in a protocol
        protocol = TBinaryProtocol.TBinaryProtocol(self.transport)

        self.socket_ip = socket_ip

        # Create a client to use the protocol encoder
        # client = BackendService.Client(protocol)

        super().__init__(protocol)

        # Connect!
        self.transport.open()

    # Destructor closes the transport
    def __del__(self):
        self.transport.close()

    def run(self, qc, shots, token=""):
        """
        Executes the ``run_func`` of the server.

        Parameters
        ----------
        qc : QuantumCircuit
            The QuantumCircuit to execute.
        shots : int
            The amount of shots.
        token : str, optional
            A token which can be used for backend execution parameter specification.
            The default is "".

        Returns
        -------
        dict
            A dictionary representing the counts where the keys are bitstrings and the
            values are integers.

        """

        converted_circuit = convert_circuit(qc, "thrift", transpile=False)

        return super().run(converted_circuit, shots, token)

    def ping(self):
        """
        Executes the ``ping_func`` on the server.

        Returns
        -------
        BackendStatus object
            An object which contains general information about the backend.
        """

        return super().ping()
