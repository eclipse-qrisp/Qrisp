"""
\********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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
    results of the QASM simulator. It is required that the run_func specifies a default
    value for the `shots` parameter. ::


        def run_func(qasm_str, shots = 1000, token = ""):

            from qiskit import QuantumCircuit

            qiskit_qc = QuantumCircuit.from_qasm_str(qasm_str)

            print(qiskit_qc)

            from qiskit_aer import AerSimulator
            qiskit_backend = AerSimulator()

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

        self.port = port
        if port is None:
            self.run_func = run_func
        else:
            # Create BackendServer
            self.backend_server = BackendServer(run_func, "localhost", port=port)
            # Run the server (runs in the background)
            self.backend_server.start()
            # Connect client

            super().__init__(api_endpoint="localhost", port=port)

    def run(self, qc, shots=None, token=""):
        """
        Executes the function run_func specified at object creation.

        Parameters
        ----------
        qc : QuantumCircuit
            The QuantumCircuit to run.
        shots : int, optional
            The amount of shots to perform.

        Returns
        -------
        res : dict
            A dictionary containing the measurement results.

        """
        if self.port is None:
            return self.run_func(qc.qasm(), shots, token)
        else:
            return super().run(qc, shots)