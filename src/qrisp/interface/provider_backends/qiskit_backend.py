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

import re

from qiskit import QuantumCircuit, transpile

from qrisp.interface.virtual_backend import VirtualBackend

from ..backend import Backend

# class QiskitBackend(VirtualBackend):
#     """
#     This class instantiates a :ref:`VirtualBackend` using a Qiskit backend.
#     This allows easy access to Qiskit backends through the qrisp interface.

#     Parameters
#     ----------
#     backend : Qiskit backend object, optional
#         A Qiskit backend object, which runs QuantumCircuits. The default is
#         ``AerSimulator()``.
#     port : int, optional
#         The port to listen. The default is None.

#     Examples
#     --------

#     We evaluate a :ref:`QuantumFloat` multiplication on the Aer simulator.

#     >>> from qrisp import QuantumFloat
#     >>> from qrisp.interface import QiskitBackend
#     >>> from qiskit_aer import AerSimulator
#     >>> example_backend = QiskitBackend(backend = AerSimulator())
#     >>> qf = QuantumFloat(4)
#     >>> qf[:] = 3
#     >>> res = qf*qf
#     >>> res.get_measurement(backend = example_backend)
#     {9: 1.0}

#     We evaluate a :ref:`QuantumFloat` multiplication on the FakeWashingtonV2 backend.

#     >>> from qrisp import QuantumFloat
#     >>> from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2
#     >>> from qrisp.interface import QiskitBackend
#     >>> example_backend = QiskitBackend(backend = FakeWashingtonV2())
#     >>> qf = QuantumFloat(2)
#     >>> qf[:] = 2
#     >>> res = qf*qf
#     >>> res.get_measurement(backend = example_backend)
#     {4: 0.6962,
#     12: 0.0967,
#     0: 0.0607,
#     8: 0.0572,
#     6: 0.028,
#     2: 0.0128,
#     14: 0.0126,
#     5: 0.0103,
#     10: 0.0062,
#     3: 0.0057,
#     9: 0.0042,
#     13: 0.0037,
#     1: 0.0029,
#     7: 0.001,
#     15: 0.001,
#     11: 0.0008}

#     We evaluate a :ref:`QuantumFloat` addition on a real IBM quantum backend.

#     >>> from qrisp import QuantumFloat
#     >>> from qrisp.interface import QiskitBackend
#     >>> from qiskit_ibm_runtime import QiskitRuntimeService
#     >>> service = QiskitRuntimeService(channel="ibm_cloud", token="YOUR_IBM_CLOUD_TOKEN")
#     >>> brisbane = service.backend("ibm_brisbane")
#     >>> qrisp_brisbane = QiskitBackend(backend)
#     >>> qf = QuantumFloat(2)
#     >>> qf[:] = 2
#     >>> qf+=1
#     >>> qf.get_measurement(backend = qrisp_brisbane)
#     {3: 0.919, 1: 0.044, 2: 0.021, 0: 0.016}

#     """

#     def __init__(self, backend=None, port=None):

#         if backend is None:
#             try:
#                 from qiskit_aer import AerSimulator

#                 backend = AerSimulator()
#             except ImportError:
#                 raise ImportError(
#                     "Encountered ImportError when trying to import AerSimulator. Likely caused by incompatible qiskit and qiskit-aer versions."
#                 )

#         try:
#             from qiskit_ibm_runtime import SamplerV2
#         except ImportError:
#             raise ImportError(
#                 "Please install qiskit-ibm-runtime to use the QiskitBackend. You can do this by running `pip install qiskit-ibm-runtime`."
#             )
#         sampler = SamplerV2(backend)

#         # Create the run method
#         def run(qasm_str, shots=None, token=""):
#             if shots is None:
#                 shots = 1000
#             # Convert to qiskit
#             from qiskit import QuantumCircuit

#             qiskit_qc = QuantumCircuit.from_qasm_str(qasm_str)

#             # Make circuit with one monolithic register
#             new_qiskit_qc = QuantumCircuit(len(qiskit_qc.qubits), len(qiskit_qc.clbits))
#             for instr in qiskit_qc:
#                 new_qiskit_qc.append(
#                     instr.operation,
#                     [qiskit_qc.qubits.index(qb) for qb in instr.qubits],
#                     [qiskit_qc.clbits.index(cb) for cb in instr.clbits],
#                 )

#             from qiskit import transpile

#             qiskit_qc = transpile(new_qiskit_qc, backend=backend)

#             job = sampler.run([qiskit_qc], shots=shots)

#             qiskit_result = (
#                 job.result()[0].data.c.get_counts()
#                 # https://docs.quantum.ibm.com/migration-guides/v2-primitives
#             )

#             # Remove the spaces in the qiskit result keys
#             result_dic = {}
#             import re

#             for key in qiskit_result.keys():
#                 counts_string = re.sub(r"\W", "", key)
#                 result_dic[counts_string] = qiskit_result[key]

#             return result_dic

#         # Call VirtualBackend constructor
#         if isinstance(backend.name, str):
#             name = backend.name
#         else:
#             name = backend.name()

#         super().__init__(run, port=port)


class QiskitBackend(Backend):
    """
    This class instantiates a :ref:`Backend` using a Qiskit backend.

    This allows easy access to Qiskit backends through the qrisp interface.

    Parameters
    ----------

    TODO: update at the end



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

    We evaluate a :ref:`QuantumFloat` multiplication on the FakeWashingtonV2 backend.

    >>> from qrisp import QuantumFloat
    >>> from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2
    >>> from qrisp.interface import QiskitBackend
    >>> example_backend = QiskitBackend(backend = FakeWashingtonV2())
    >>> qf = QuantumFloat(2)
    >>> qf[:] = 2
    >>> res = qf*qf
    >>> res.get_measurement(backend = example_backend)
    {4: 0.6962,
    12: 0.0967,
    0: 0.0607,
    8: 0.0572,
    6: 0.028,
    2: 0.0128,
    14: 0.0126,
    5: 0.0103,
    10: 0.0062,
    3: 0.0057,
    9: 0.0042,
    13: 0.0037,
    1: 0.0029,
    7: 0.001,
    15: 0.001,
    11: 0.0008}

    """

    def __init__(self, backend=None, name=None, description=None, options=None):
        if backend is None:
            try:
                from qiskit_aer import AerSimulator

                backend = AerSimulator()
            except ImportError:
                raise ImportError(
                    "Encountered ImportError when trying to import AerSimulator."
                )

        self.backend = backend

        try:
            from qiskit_ibm_runtime import SamplerV2
        except ImportError:
            raise ImportError(
                "Please install qiskit-ibm-runtime to use the QiskitBackend. You can do this by running `pip install qiskit-ibm-runtime`."
            )
        self.sampler = SamplerV2(backend)

        # If not specified, we use the Qiskit backend metadata
        if name is None:
            name = getattr(backend, "name", None)

        if description is None:
            description = getattr(backend, "description", None)

        if options is None:
            options = getattr(backend, "options", None)

        super().__init__(name=name, description=description, options=options)

    @classmethod
    def _default_options(cls):
        return {
            "shots": 1000,
            "optimization_level": None,
        }

    @property
    def max_circuits(self):
        return None

    def run(self, qc, shots=None):
        """
        Execute QASM code on a Qiskit backend using SamplerV2.

        Parameters
        ----------
        qc : QuantumCircuit
            Qiskit QuantumCircuit object
        shots : int
            Number of repetitions

        Returns
        -------
        dict
            Measurement results (bitstring → counts)
        """

        qasm_str = qc.qasm()

        if shots is None:
            shots = self._options.get("shots", 1000)

        qiskit_qc = QuantumCircuit.from_qasm_str(qasm_str)

        new_qc = QuantumCircuit(len(qiskit_qc.qubits), len(qiskit_qc.clbits))
        for instr in qiskit_qc:
            new_qc.append(
                instr.operation,
                [qiskit_qc.qubits.index(qb) for qb in instr.qubits],
                [qiskit_qc.clbits.index(cb) for cb in instr.clbits],
            )

        new_qc = transpile(
            new_qc,
            backend=self.backend,
            optimization_level=self._options.get("optimization_level"),
        )

        job = self.sampler.run([new_qc], shots=shots)
        qiskit_counts = job.result()[0].data.c.get_counts()

        result = {}
        for key, value in qiskit_counts.items():
            cleaned = re.sub(r"\W", "", key)
            result[cleaned] = value

        return result


class QiskitRuntimeBackend(VirtualBackend):
    """
    This class instantiates a VirtualBackend using a Qiskit Runtime backend.
    This allows easy access to Qiskit Runtime backends through the qrisp interface.
    It is imporant to close the session after the execution of the algorithm
    (as reported in the example below).

    Parameters
    ----------
    api_token : str
        The token is necessary to create correctly the Qiskit Runtime
        service and be able to run algorithms on their backends.
    backend : str, optional
        A string associated to the name of a Qiskit Runtime backend.
        By default, the least busy available backend is selected.
    channel : str, optional
        The channel type. Available are ``ibm_cloud`` or ``ibm_quantum_platform``.
        The default is ``ibm_cloud``.
    mode : str, optional
        The `execution mode <https://quantum.cloud.ibm.com/docs/en/guides/execution-modes>`_. Available are ``job`` and ``session``.
        The default is ``job``.

    Attributes
    ----------
    session : Session
        The `Qiskit Runtime session <https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/session>`_.

    Examples
    --------

    >>> from qrisp import QuantumFloat
    >>> from qrisp.interface import QiskitRuntimeBackend
    >>> example_backend = QiskitRuntimeBackend(api_token = "YOUR_IBM_CLOUD_TOKEN", backend = "ibm_brisbane", channel = "ibm_cloud")
    >>> qf = QuantumFloat(2)
    >>> qf[:] = 2
    >>> res = qf*qf
    >>> result = res.get_measurement(backend = example_backend)
    >>> print(result)
    >>> # example_backend.close_session() # Use only when mode = "session"
    {4: 0.6133,
    8: 0.1126,
    0: 0.0838,
    12: 0.0798,
    5: 0.0272,
    6: 0.016,
    9: 0.0125,
    1: 0.0117,
    13: 0.0081,
    14: 0.0073,
    3: 0.0071,
    2: 0.0062,
    10: 0.0051,
    7: 0.0044,
    11: 0.0035,
    15: 0.0014}

    """

    def __init__(self, api_token, backend=None, channel="ibm_cloud", mode="job"):

        try:
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Session
        except ImportError:
            raise ImportError(
                "Please install qiskit-ibm-runtime to use the QiskitBackend. You can do this by running `pip install qiskit-ibm-runtime`."
            )

        service = QiskitRuntimeService(channel=channel, token=api_token)
        if backend is None:
            backend = service.least_busy()
        else:
            backend = service.backend(backend)

        if mode == "session":
            self.session = Session(backend)
            sampler = SamplerV2(self.session)
        elif mode == "job":
            sampler = SamplerV2(backend)
        else:
            raise ValueError(f"Execution mode" + str(mode) + " not available.")

        # Create the run method
        def run(qasm_str, shots=None, token=""):
            if shots is None:
                shots = 1000
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

            job = sampler.run([qiskit_qc], shots=shots)

            qiskit_result = (
                job.result()[0].data.c.get_counts()
                # https://docs.quantum.ibm.com/migration-guides/v2-primitives
            )

            # Remove the spaces in the qiskit result keys
            result_dic = {}
            import re

            for key in qiskit_result.keys():
                counts_string = re.sub(r"\W", "", key)
                result_dic[counts_string] = qiskit_result[key]

            return result_dic

        super().__init__(run)

    def close_session(self):
        """
        Method to call in order to close the session started by the init method.
        """
        self.session.close()
