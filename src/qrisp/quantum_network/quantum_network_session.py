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

import numpy as np

from qrisp import QuantumBool, QuantumChar, QuantumFloat, QuantumSession, merge
from qrisp.circuit import QuantumCircuit, QubitAlloc, QubitDealloc, transpile
from qrisp.quantum_network import QuantumNetworkClient


class QuantumNetworkSession(QuantumSession):
    """
    This class allows using Qrisps high-level programming interface to interact with
    QuantumNetworks. As an inheritor of the QuantumSession class, we can create
    QuantumVariables with it which can then be sent to other participators of the
    QuantumNetwork.

    Note that get_measurement calls are also realized by the single-shot simulator. For
    more information on this check the documentation of the run method of the
    QuantumNetworkClient class.

    Parameters
    ----------
    name : str
        A nickname such as "Alice" or "Bob", which is used to direct the communication
        of other users to you.
    socket_ip : str
        The IP of the QuantumNetworkServer to connect to.
    port : int
        The port of the QuantumNetworkServer to connect to. The default is 7070.

    Examples
    --------

    We create a QuantumNetworkServer and connect a QuantumNetworkSession to it.

    >>> from qrisp.quantum_network import QuantumNetworkServer, QuantumNetworkSession
    >>> example_server = QuantumNetworkServer(socket_ip = "127.0.0.1", background = True)
    >>> example_server.start()
    >>> alice_session = QuantumNetworkSession(socket_ip = "127.0.0.1", port = 7070, name = "alice")

    Now we can create QuantumVariables as we are used to

    >>> from qrisp import QuantumFloat
    >>> qf = QuantumFloat(4, 5, qs = alice_session)


    """

    def __init__(self, name, socket_ip, port=None):
        qn_client = QuantumNetworkClient(name=name, socket_ip=socket_ip, port=port)
        self.inbox = []

        super().__init__(backend=qn_client)

    def send_qv(self, qv, recipient):
        """
        Sends a :ref:`QuantumVariable` to another participant of the network.

        Parameters
        ----------
        qv : QuantumVariable
            The QuantumVariable to send.
        recipient : str
            The name of the recipient.

        Examples
        --------

        We create a QuantumNetworkServer and connect a QuantumNetworkSession to it.

        >>> from qrisp.quantum_network import QuantumNetworkServer, QuantumNetworkSession
        >>> example_server = QuantumNetworkServer(socket_ip = "127.0.0.1", background = True)
        >>> example_server.start()
        >>> alice_session = QuantumNetworkSession(socket_ip = "127.0.0.1", port = 7070, name = "alice")
        >>> bob_session = QuantumNetworkSession(socket_ip = "127.0.0.1", port = 7070, name = "bob")

        We create a :ref:`QuantumFloat` and perform some arithmetic

        >>> from qrisp import QuantumFloat
        >>> qf = QuantumFloat(4, qs = alice_session)
        >>> qf += 3
        >>> alice_session.send_qv(qf, "bob")


        """

        if not qv.qs == self:
            merge(self, qv.qs)

        self.backend.run(self)

        qv_qubits = list(qv.reg)

        type_information = {}

        if isinstance(qv, QuantumFloat):
            type_information["qtype"] = "qf"
            type_information["msize"] = str(qv.msize)
            type_information["exponent"] = str(qv.exponent)
            type_information["signed"] = str(int(qv.signed))
        elif isinstance(qv, QuantumChar):
            type_information["qtype"] = "qc"
            type_information["nisq_char"] = str(int(qv.nisq_char))
        elif isinstance(qv, QuantumBool):
            type_information["qtype"] = "qb"

        self.backend.send_qubits(recipient, qv_qubits, str(type_information))

        qv.delete()

        for i in range(len(qv_qubits)):
            self.qubits.remove(qv_qubits[i])

        self.data = []

    def update(self):
        """
        This updates the inbox attribute and returns the received QuantumVariables.

        Returns
        -------
        received_qvs : list[QuantumVariable]
            The QuantumVariables which have been received since the last update.

        Examples
        --------

        We assume that the commands from the example of the
        :meth:`send_qv <qrisp.quantum_network.QuantumNetworkSession.send_qv>` method
        have been executed.

        >>> inbox = bob_session.update()
        >>> print(inbox[0])
        {3.0: 1.0}

        """

        from qrisp import QuantumVariable

        self.backend.run(self)

        updated_inbox = self.backend.inbox()

        received_qvs = []
        starting_index = int(len(self.inbox))
        from ast import literal_eval

        for i in range(starting_index, len(updated_inbox)):
            for qb in updated_inbox[i][0]:
                self.add_qubit(qb)

            try:
                type_information = literal_eval(updated_inbox[i][1])
            except:
                type_information = {}

            new_qv = QuantumVariable(0, qs=self)

            new_qv.extend(len(updated_inbox[i][0]), proposal=updated_inbox[i][0])

            if len(type_information):
                if type_information["qtype"] == "qf":
                    new_qv.__class__ = QuantumFloat
                    new_qv.msize = int(type_information["msize"])
                    new_qv.exponent = int(type_information["exponent"])
                    new_qv.signed = bool(int(type_information["signed"]))

                if type_information["qtype"] == "qc":
                    new_qv.__class__ = QuantumChar
                    new_qv.nisq_char = bool(int(type_information["nisq_char"]))

                elif type_information["qtype"] == "qb":
                    new_qv.__class__ = QuantumBool

            received_qvs.append(new_qv)

        self.inbox += received_qvs
        return received_qvs
