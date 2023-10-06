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


from qrisp import Qubit
from qrisp.interface import PortableQubit
from qrisp.interface.circuit_converter import convert_circuit
from qrisp.quantum_network.interface.codegen import QuantumNetworkService
from qrisp.quantum_network.interface.codegen.ttypes import Message


class QuantumNetworkClient(QuantumNetworkService.Client):
    """
    This class allows connecting to Qrisp quantum networks.

    Parameters
    ----------
    name : str
        A nickname such as "Alice" or "Bob", which is used to direct the communication
        of other users to you.
    socket_ip : str
        The IP adress of the QuantumNetworkServer.
    port : int, optional
        The port of the QuantumNetworkServer. The default is 7070.

    Examples
    --------

    We create a QuantumNetworkServer listening on the localhost IP-address and connect
    the client.

    >>> from qrisp.quantum_network import QuantumNetworkServer, QuantumNetworkClient
    >>> local_server = QuantumNetworkServer("127.0.0.1", background = True)
    >>> local_server.start()
    >>> client = QuantumNetworkClient(name = "alice", socket_ip = "127.0.0.1")


    """

    def __init__(self, name, socket_ip, port=None):
        from thrift.protocol import TBinaryProtocol
        from thrift.transport import TSocket, TTransport

        if port is None:
            port = 7070
        # Create the transport for the User Interface to the server
        self.transport = TSocket.TSocket(socket_ip, port)
        # Buffering is critical. Raw sockets are very slow
        self.transport = TTransport.TBufferedTransport(self.transport)
        # Wrap in a protocol
        protocol = TBinaryProtocol.TBinaryProtocol(self.transport)
        self.socket_ip = socket_ip
        # Create a client to use the protocol encoder

        super().__init__(protocol)

        # Connect!
        self.transport.open()

        self.name = name
        self.register_client(name)

    # Destructor closes the transport
    def __del__(self):
        self.transport.close()

    def run(self, qc, shots=None):
        r"""
        Runs a QuantumCircuit on the client's backend. Note that QuantumNetwork
        simulations do not support multiple shots, as the quantum state of the network
        is stored and updated everytime a client sends a query. Multiple shots could
        yield differing measurement outcomes, which implies an ambiguous quantum state
        of the network. Nevertheless, the results are returned in the form of a
        dictionary in order to comply with the quantum circuit execution backend
        interface.

        Note that it is possible to submit QuantumCircuits which contain qubits, that
        have not been requested previously. In this case, the qubits names are
        internally extended by the string "@client_name"  (if they aren't extended in
        this way already). This is to allow multiple clients to submit circuits with
        matching qubit names.

        Parameters
        ----------
        qc : QuantumCircuit
            The QuantumCircuit to run.

        Returns
        -------
        res : dict
            A dictionary containing a single key/value pair where the key represents the
            measurement outcome.


        Examples
        --------

        We create a local QuantumNetworkServer, connect a client and run a
        QuantumCircuit.

        >>> from qrisp.quantum_network import QuantumNetworkServer, QuantumNetworkClient
        >>> local_server = QuantumNetworkServer("127.0.0.1", background = True)
        >>> local_server.start()
        >>> client = QuantumNetworkClient(name = "alice", socket_ip = "127.0.0.1")
        >>> from qrisp import QuantumCircuit
        >>> qc = QuantumCircuit(2)
        >>> qc.h(0)
        >>> qc.cx(0,1)
        >>> qc.measure([0,1])
        >>> client.run(qc)
        {'11': 1}

        We perform another shot

        >>> client.run(qc)
        {'01': 1}

        After applying the first run command, the quantum state is

        .. math::
            \ket{\psi} = \ket{1}\ket{1}

        Applying the Hadamard yields

        .. math::
            \text{H}_0 \ket{\psi} = \frac{1}{\sqrt{2}}(\ket{0} - \ket{1})\ket{1}

        The CX yields

        .. math::
            \text{CX}_{01} \text{H}_0 \ket{\psi} =
            \frac{1}{\sqrt{2}}(\ket{0}\ket{1} - \ket{1}\ket{0})

        Finally, the measurement collapsed the state into the first summand.

        """
        converted_circuit = convert_circuit(qc, "thrift")

        res = super().run(converted_circuit, self.name)

        return res

        qc = super().get_clear_qc(self.name)

        return convert_circuit(qc, "qrisp")

    def send_qubits(self, recipient, qubits, annotation):
        """
        Sends the specified qubits to another participant of the network.

        Parameters
        ----------
        recipient : str
            The recipients name.
        qubits : list[Qubit]
            The list of qubits to send.
        annotation : str
            A string containing an arbitrary message that is available for the
            reciepient.


        Examples
        --------

        We create a QuantumNetworkServer, connect two clients and distribute a bell pair
        another client.

        >>> from qrisp.quantum_network import QuantumNetworkServer, QuantumNetworkClient
        >>> local_server = QuantumNetworkServer("127.0.0.1", background = True)
        >>> local_server.start()
        >>> alice_client = QuantumNetworkClient(name = "alice", socket_ip = "127.0.0.1")
        >>> bob_client = QuantumNetworkClient(name = "bob", socket_ip = "127.0.0.1")

        Now we create the bell pair:

        >>> from qrisp import QuantumCircuit
        >>> qc = QuantumCircuit(2)
        >>> qc.h(0)
        >>> qc.cx(0,1)
        >>> alice_client.run(qc)
        {'': 1}

        Send one of the qubits to Bob:

        >>> alice_client.send_qubits("bob", [qc.qubits[0]], annotation = "Happy birthday!")

        Now Bob can measure

        >>> messages = bob_client.inbox()
        >>> received_qubits, annotation = messages[0]
        >>> qc = bob_client.get_clear_qc()
        >>> qc.measure(received_qubits)
        >>> bob_client.run(qc)
        {'1': 1}

        After Bob's measurement, we expect Alice's measurement to yield the same due to
        `spukhafter Fernwirkung
        <https://en.wikipedia.org/wiki/Quantum_entanglement#History>`_:

        >>> qc = alice_client.get_clear_qc()
        >>> qc.measure(qc.qubits)
        >>> alice_client.run(qc)
        {'1': 1}



        """

        msg = Message([PortableQubit(qb.identifier) for qb in qubits], annotation)

        super().send_qubits(self.name, recipient, msg)

    def request_qubits(self, amount):
        """
        Creates the specified amount of qubits for the client's backend.

        Parameters
        ----------
        amount : int
            The amount of qubits to create.

        Returns
        -------
        list[Qubit]
            The qubits created.

        Examples
        --------

        >>> from qrisp.quantum_network import QuantumNetworkServer, QuantumNetworkClient
        >>> local_server = QuantumNetworkServer("127.0.0.1", background = True)
        >>> local_server.start()
        >>> client = QuantumNetworkClient(name = "alice", socket_ip = "127.0.0.1")
        >>> qb_list = client.request_qubits(4)
        >>> print(qb_list)
        [Qubit(qb_0@alice), Qubit(qb_1@alice), Qubit(qb_2@alice), Qubit(qb_3@alice)]

        """

        qb_list = super().request_qubits(amount, self.name)

        return [Qubit(qb.identifier) for qb in qb_list]

    def get_clear_qc(self):
        """
        Returns a QuantumCircuit containing all the qubits that belong to the client at
        the moment.

        Note that the qubit names of the run QuantumCircuit are internally extended by
        the string "@client_name"  (if they aren't extended in this way already). This
        is to allow multiple clients to submit circuits with matching qubit names.

        Note that

        Returns
        -------
        QuantumCircuit
            An empty QuantumCircuit containing the qubits, this client may operate on.

        Examples
        --------

        We set up a quantum network, run some QuantumCircuits and retrieve the clear
        QuantumCircuit:

        >>> from qrisp.quantum_network import QuantumNetworkServer, QuantumNetworkClient
        >>> local_server = QuantumNetworkServer("127.0.0.1", background = True)
        >>> local_server.start()
        >>> client = QuantumNetworkClient(name = "alice", socket_ip = "127.0.0.1")
        >>> from qrisp import QuantumCircuit
        >>> qc_0 = QuantumCircuit(1)
        >>> qc_0.x(0)
        >>> print(qc_0)
        
        ::
        
                  ┌───┐
            qb_8: ┤ X ├
                  └───┘

        >>> client.run(qc_0)
        >>> qc_1 = QuantumCircuit(1)
        >>> qc_1.h(0)
        >>> print(qc_1)
        
        ::
        
                    ┌───┐
             qb_26: ┤ H ├
                    └───┘
                    
        >>> client.run(qc_1)
        >>> print(client.get_clear_qc().qubits)
        [Qubit(qb_58@alice), Qubit(qb_77@alice)]

        """
        qc = super().get_clear_qc(self.name)

        return convert_circuit(qc, "qrisp")

    def get_overall_qc(self):
        """
        Retrieves the QuantumCircuit of all operations that happened in the network so
        far.

        Returns
        -------
        QuantumCircuit
            The QuantumCircuit containing all operations that happened in this
            QuantumNetwork.

        Examples
        --------

        We assume that the commands of the example of the
        :meth:`send_qubits <qrisp.quantum_network.QuantumNetworkClient.send_qubits>`
        method have been executed.

        >>> print(alice_client.get_overall_qc())
        
        ::
        
                          ░ ┌───┐      ░ ┌─┐ ░
            qb_34@alice: ─░─┤ H ├──■───░─┤M├─░────
                          ░ └───┘┌─┴─┐ ░ └╥┘ ░ ┌─┐
            qb_35@alice: ─░──────┤ X ├─░──╫──░─┤M├
                          ░      └───┘ ░  ║  ░ └╥┘
               cb_bob_0: ═════════════════╩═════╬═
                                                ║
             cb_alice_0: ═══════════════════════╩═


        """

        qc = super().get_overall_qc()

        return convert_circuit(qc, "qrisp")

    def inbox(self):
        """
        Returns a list of tuples containing the received qubits and their annotations.

        Returns
        -------
        res_list : list[tuple[list[Qubit], str]]
            A list of tuples containing the sent qubits and the annotations.


        Examples
        --------

        We send a qubit from one client to another and inspect the inbox

        >>> from qrisp.quantum_network import QuantumNetworkServer, QuantumNetworkClient
        >>> local_server = QuantumNetworkServer("127.0.0.1", background = True)
        >>> local_server.start()
        >>> alice_client = QuantumNetworkClient(name = "alice", socket_ip = "127.0.0.1")
        >>> bob_client = QuantumNetworkClient(name = "bob", socket_ip = "127.0.0.1")

        Prepare the qubit which will be sent to bob:

        >>> from qrisp import QuantumCircuit
        >>> qc = QuantumCircuit(1)
        >>> qc.h(0)
        >>> alice_client.run(qc)
        {'': 1}

        Send the qubit to Bob:

        >>> alice_client.send_qubits("bob", [qc.qubits[0]], annotation = "Merry christmas!")

        Now Bob can check his inbox

        >>> messages = bob_client.inbox()
        >>> received_qubits, annotation = messages[0]
        >>> print(received_qubits)
        [Qubit(qb_20@alice)]
        >>> print(annotation)
        Merry christmas!

        """

        inbox_list = super().inbox(self.name)

        res_list = []

        for i in range(len(inbox_list)):
            res_list.append(
                (
                    [Qubit(qb.identifier) for qb in inbox_list[i].qubits],
                    inbox_list[i].annotation,
                )
            )
        return res_list
