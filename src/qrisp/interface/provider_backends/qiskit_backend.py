"""
********************************************************************************
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
********************************************************************************
"""

from qrisp.interface.virtual_backend import VirtualBackend


class QiskitBackend(VirtualBackend):
    """
    This class instantiates a VirtualBackend using a Qiskit backend.
    This allows easy access to Qiskit backends through the qrisp interface.


    Parameters
    ----------
    backend : Qiskit backend object, optional
        A Qiskit backend object, which runs QuantumCircuits. The default is
        ``AerSimulator()``.
    port : int, optional
        The port to listen. The default is 8079.

    Examples
    --------

    We evaluate a :ref:`QuantumFloat` multiplication on the Aer simulator.

    >>> from qrisp import QuantumFloat
    >>> from qrisp.interface import QiskitBackend
    >>> from qiskit_aer import AerSimulator
    >>> example_backend = QiskitBackend(backend = AerSimulator())
    >>> qf = QuantumFloat(4)
    >>> qf[:] = 3
    >>> res = qf*qf
    >>> res.get_measurement(backend = example_backend)
    {9: 1.0}


    """

    def __init__(self, backend=None, port=None):
        if backend is None:

            try:
                from qiskit_aer import AerSimulator

                backend = AerSimulator()
            except ImportError:
                import qiskit_aer as Aer

                backend = Aer.AerSimulator()

        # Create the run method
        def run(qasm_str, shots=None, token=""):
            if shots is None:
                shots = 100000
            # Convert to qiskit
            from qiskit import QuantumCircuit

            qiskit_qc = QuantumCircuit.from_qasm_str(qasm_str)

            # Make circuit with one monolithic register
            new_qiskit_qc = QuantumCircuit(len(qiskit_qc.qubits), len(qiskit_qc.clbits))
            for instr in qiskit_qc:
                new_qiskit_qc.append(
                    instr.operation,
                    [qiskit_qc.qubits.index(qb) for qb in instr.qubits],
                    [qiskit_qc.clbits.index(cb) for cb in instr.clbits],
                )

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

            print(result_dic)

            return result_dic

        # Call VirtualBackend constructor
        if isinstance(backend.name, str):
            name = backend.name
        else:
            name = backend.name()

        super().__init__(run, port=port)


def VirtualQiskitBackend(*args, **kwargs):
    import warnings

    warnings.warn(
        "VirtualQiskitBackend will be deprecated in a future release of Qrisp. Use QiskitBackend instead."
    )
    return QiskitBackend(*args, **kwargs)


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

    Examples
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

    def __init__(self, backend=None, port=8079, token=""):
        from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session, Estimator

        service = QiskitRuntimeService(channel="ibm_quantum", token=token)
        if backend is None:
            backend = service.get_backend("ibmq_qasm_simulator")
        else:
            backend = service.get_backend(backend)

        session = Session(service, backend)
        sampler = Sampler(session=session)

        # Create the run method
        def run(qasm_str, shots=None, token=""):
            if shots is None:
                shots = 100000
            # Convert to qiskit
            from qiskit import QuantumCircuit

            qiskit_qc = QuantumCircuit.from_qasm_str(qasm_str)

            from qiskit import transpile

            qiskit_qc = transpile(qiskit_qc, backend=backend)
            # Run Circuit with the Sampler Primitive
            qiskit_result = (
                sampler.run(qiskit_qc, shots=shots)
                .result()
                .quasi_dists[0]
                .binary_probabilities()
            )
            qiskit_result.update(
                (key, round(value * shots)) for key, value in qiskit_result.items()
            )

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
