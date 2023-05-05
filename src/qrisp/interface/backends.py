"""
/*********************************************************************
* Copyright (c) 2023 the Qrisp Authors
*
* This program and the accompanying materials are made
* available under the terms of the Eclipse Public License 2.0
* which is available at https://www.eclipse.org/legal/epl-2.0/
*
* SPDX-License-Identifier: EPL-2.0
**********************************************************************/
"""


from qrisp.interface import BackendClient


class VirtualBackend(BackendClient):
    """
    This class provides a virtual backend for circuit execution.
    Virtual means that the server is running on the same machine as a separate
    Python thread. This structure allows setting up convenient wrappers for foreign/
    modified circuit dispatching code.

    Circuits can be run using ``virtual_backend_instance.run(qc)``.
    The function that should be used to run a circuit can be specified during
    construction using the ``run_func`` parameter.


    Parameters
    ----------
    run_func : function
        A function that recieves a QuantumCircuit, an integer specifiying the amount of
        shots and a token in the form of a string. It returns the counts as a dictionary
        of bitstrings.
    name : str, optional
        A name for the virtual backend. The default is None.
    port : int, optional
        The port on which to listen. The default is None.
    ping_func : TYPE, optional
        A function which returns a BackendStatus object. The default is None.


    Examples
    --------


    We set up a VirtualBackend, which prints the received QuantumCircuit and returns the
    results of the QASM simulator. ::

        from qrisp.interface import convert_to_qiskit

        def run_func(qc, shots, token = ""):

            qiskit_qc = convert_to_qiskit(qc)

            print(qiskit_qc)

            from qiskit import Aer
            qiskit_backend = Aer.get_backend('qasm_simulator')

            #Run Circuit on the Qiskit backend
            return qiskit_backend.run(qiskit_qc, shots = shots).result().get_counts()


    >>> from qrisp.interface import VirtualBackend
    >>> example_backend = VirtualBackend(run_func)
    >>> from qrisp import QuantumFloat
    >>> qf = QuantumFloat(3)
    >>> qf[:] = 4
    >>> qf.get_measurement(backend = example_backend)
                 ┌─┐
    qf.0:   ─────┤M├──────
                 └╥┘┌─┐
    qf.1:   ──────╫─┤M├───
            ┌───┐ ║ └╥┘┌─┐
    qf.2:   ┤ X ├─╫──╫─┤M├
            └───┘ ║  ║ └╥┘
    cb_0: 1/══════╩══╬══╬═
                  0  ║  ║
    cb_1: 1/═════════╩══╬═
                     0  ║
    cb_2: 1/════════════╩═
    {4: 1.0}

    """

    def __init__(self, run_func, name=None, port=None, ping_func=None):
        if name is None:
            name = "virtual_quantum_backend"
        else:
            name = "virtual_quantum_backend_" + name

        from qrisp.interface import BackendServer

        # Create BackendServer
        self.backend_server = BackendServer(
            run_func, "::1", port=port, name=name, ping_func=ping_func
        )
        # Run the server (runs in the background)
        self.backend_server.start()
        # Connect client

        super().__init__(socket_ip="::1", port=port)

    def run(self, qc, shots, token=""):
        """
        Executes the function run_func specified at object creation.

        Parameters
        ----------
        qc : QuantumCircuit
            The QuantumCircuit to run.
        shots : int
            The amount of shots to perform.

        Returns
        -------
        res : dict
            A dictionary containing the measurement results.

        """
        return super().run(qc, shots, token)

    def ping(self):
        """
        Executes the function ping_func specified at object creation.


        Returns
        -------
        res : BackendStatus object

        """
        return super().ping()


class VirtualQiskitBackend(VirtualBackend):
    """
    This class instantiates a VirtualBackend using a Qiskit backend.
    This allows easy access to Qiskit backends through the qrisp interface.


    Parameters
    ----------
    backend : Qiskit backend object, optional
        A Qiskit backend object, which runs QuantumCircuits. The default is
        Aer.get_backend('qasm_simulator').
    port : int, optional
        The port to listen. The default is 8079.

    Examples
    --------

    We evaluate a QuantumFloat multiplication on the QASM-simulator.

    >>> from qrisp import QuantumFloat
    >>> from qrisp.interface import VirtualQiskitBackend
    >>> from qiskit import Aer
    >>> example_backend = VirtualQiskitBackend(backend = Aer.get_backend('qasm_simulator'))
    >>> qf = QuantumFloat(4)
    >>> qf[:] = 3
    >>> res = qf*qf
    >>> res.get_measurement(backend = example_backend)
    {9: 1.0}


    """

    from qiskit import Aer

    def __init__(self, backend=None, port=8079):
        if backend is None:
            from qiskit import Aer

            backend = Aer.get_backend("qasm_simulator")

        # Create the run method
        def run(qc, shots, token):
            # Convert to qiskit
            from qrisp.interface.circuit_converter import convert_circuit

            qiskit_qc = convert_circuit(qc, "qiskit", transpile=True)

            from qiskit import transpile

            qiskit_qc = transpile(qiskit_qc, backend=backend)
            # Run Circuit on the Qiskit backend
            qiskit_result = backend.run(qiskit_qc, shots=shots).result().get_counts()
            # Remove the spaces in the qiskit result keys
            result_dic = {}
            import re

            for key in qiskit_result.keys():
                counts_string = re.sub(r"\W", "", key)
                result_dic[counts_string] = qiskit_result[key]

            return result_dic

        # Create the ping method
        def ping():
            # Collect information about backend to create BackendStatus object

            config = backend.configuration()

            # Information about Qubits and their conznectivity
            from qrisp.interface import ConnectivityEdge, PortableQubit

            qubit_list = []
            for i in range(config.n_qubits):
                qubit_list.append(PortableQubit(str(i)))

            connectivity_map = []

            if config.coupling_map:
                for edge in config.coupling_map:
                    connectivity_map.append(
                        ConnectivityEdge(qubit_list[edge[0]], qubit_list[edge[1]])
                    )

            # Information about elementary gates
            from qrisp.circuit.standard_operations import op_list

            # TO-DO fix elementary ops
            for gate_name in config.basis_gates:
                for op in op_list:
                    if op().name == gate_name:
                        pass
                        # elementary_ops.append(op())

            elementary_ops = []
            from qrisp.interface import BackendStatus

            # Create BackendStatus object
            backend_status = BackendStatus(
                name=config.backend_name,
                qubit_list=qubit_list,
                elementary_ops=elementary_ops,
                connectivity_map=connectivity_map,
                online=True,
            )

            return backend_status

        # Call VirtualBackend constructor
        if isinstance(backend.name, str):
            name = backend.name
        else:
            name = backend.name()

        super().__init__(run, name=name, ping_func=ping, port=port)
