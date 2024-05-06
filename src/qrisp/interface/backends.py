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

    Examples
    --------


    We set up a VirtualBackend, which prints the received QuantumCircuit and returns the
    results of the QASM simulator. ::


        def run_func(qasm_str, shots, token = ""):
            
            from qiskit import QuantumCircuit
            
            qiskit_qc = QuantumCircuit.from_qasm_str(qasm_str)

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

    def __init__(self, run_func, port=None):

        from qrisp.interface import BackendServer

        # Create BackendServer
        self.backend_server = BackendServer(
            run_func, "localhost", port=port
        )
        # Run the server (runs in the background)
        self.backend_server.start()
        # Connect client

        super().__init__(api_endpoint="localhost", port=port)

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
        return super().run(qc, shots)


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

    def __init__(self, backend=None, port=None):
        if backend is None:
            
            try:
                from qiskit import Aer
                backend = Aer.get_backend("qasm_simulator")
            except ImportError:
                from qiskit.providers.basic_provider import BasicProvider
                backend = BasicProvider().get_backend('basic_simulator')

        # Create the run method
        def run(qasm_str, shots, token = ""):
            # Convert to qiskit
            from qiskit import QuantumCircuit

            qiskit_qc = QuantumCircuit.from_qasm_str(qasm_str)

            
            #Make circuit with one monolithic register
            new_qiskit_qc = QuantumCircuit(len(qiskit_qc.qubits), len(qiskit_qc.clbits))
            for instr in qiskit_qc:
                new_qiskit_qc.append(instr.operation, 
                                     [qiskit_qc.qubits.index(qb) for qb in instr.qubits],
                                     [qiskit_qc.clbits.index(cb) for cb in instr.clbits])

            from qiskit import transpile
            qiskit_qc = transpile(new_qiskit_qc, backend=backend)
            # Run Circuit on the Qiskit backend
            qiskit_result = backend.run(qiskit_qc, shots=shots).result().get_counts()
            # Remove the spaces in the qiskit result keys
            result_dic = {}
            import re

            for key in qiskit_result.keys():
                counts_string = re.sub(r"\W", "", key)
                result_dic[counts_string] = qiskit_result[key]

            return result_dic

        # Call VirtualBackend constructor
        if isinstance(backend.name, str):
            name = backend.name
        else:
            name = backend.name()

        super().__init__(run, port=port)


class QiskitRuntimeBackend(VirtualBackend):
    """
    This class instantiates a VirtualBackend using a Qiskit Runtime backend.
    This allows easy access to Qiskit Runtime backends through the qrisp interface.
    It is imporant to close the session after the execution of the algorithm
    (as reported in the example below).

    Parameters
    ----------
    backend : str, optional
        A string associated to the name of a Qiskit Runtime Backend.
        The default one is the "ibmq_qasm_simulator", but it also 
        possible to provide other ones like "simulator_statevector"
        or real backends from the Qiskit Runtime.
    port : int, optional
        The port to listen. The default is 8079.
    token : str
        The token is necessary to create correctly the Qiskit Runtime
        service and be able to run algorithms on their backends.

    Example
    --------

    >>> from qrisp import QuantumFloat
    >>> from qrisp.interface import QiskitRuntimeBackend
    >>> example_backend = QiskitRuntimeBackend(backend="simulator_statevector",token='YOUR_TOKEN')
    >>> qf = QuantumFloat(4)
    >>> qf[:] = 3
    >>> res = qf*qf
    >>> result=res.get_measurement(backend = example_backend)
    >>> print(results)
    >>> example_backend.close_session()
    {9: 1.0}

    """

    def __init__(self, backend=None, port=8079, token=''):
        from qiskit_ibm_runtime import QiskitRuntimeService,Sampler,Session,Estimator

        service = QiskitRuntimeService(channel="ibm_quantum", token=token)
        if backend is None:
            backend = service.get_backend("ibmq_qasm_simulator")
        else:
            backend = service.get_backend(backend)

            
        session = Session(service,backend)
        sampler = Sampler(session=session)

        # Create the run method
        def run(qasm_str, shots, token = ""):
            # Convert to qiskit
            from qiskit import QuantumCircuit
            qiskit_qc = QuantumCircuit.from_qasm_str(qasm_str)

            from qiskit import transpile

            qiskit_qc = transpile(qiskit_qc, backend=backend)
            # Run Circuit with the Sampler Primitive
            qiskit_result = sampler.run(qiskit_qc, shots=shots).result().quasi_dists[0].binary_probabilities()
            qiskit_result.update((key, round(value * shots)) for key, value in qiskit_result.items())
            
            # Remove the spaces in the qiskit result keys
            result_dic = {}
            import re

            for key in qiskit_result.keys():
                counts_string = re.sub(r"\W", "", key)
                result_dic[counts_string] = qiskit_result[key]
            
            return result_dic

        super().__init__(run, port=port)

    def close_session(self): 
        """
        Method to call in order to close the session started by the init method.
        """
        self.session.close()