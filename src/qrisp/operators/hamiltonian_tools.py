"""
\********************************************************************************
* Copyright (c) 2024 the Qrisp authors
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

from qrisp.operators.hamiltonian import Hamiltonian
from qrisp.operators.pauli.pauli_measurement import PauliMeasurement


def multi_hamiltonian_measurement(
        hamiltonians, 
        qarg,
        precision=0.01,
        backend=None,
        shots=1000000,
        compile=True,
        compilation_kwargs={},
        subs_dic={},
        circuit_preprocessor=None,
        precompiled_qc=None,
        _measurements=None
    ):
    r"""
    This method returns the expected value of a list of Hamiltonians for the state of a quantum argument.

    Parameters
    ----------
    hamiltonians : list[Hamiltonian]
        The Hamiltonians for wich the expeced value is to be evaluated.
    qarg : QuantumVariable, QuantumArray or list[QuantumVariable]
        The quantum argument to evaluate the Hamiltonians on.
    precision: float, optional
        The precision with which the expectation of the Hamiltonians is to be evaluated.
        The default is 0.01.
    backend : BackendClient, optional
        The backend on which to evaluate the quantum circuit. The default can be
        specified in the file default_backend.py.
    shots : integer, optional
        The maximum amount of shots to evaluate the expectation per Hamiltonian. 
        The default is 100000.
    compile : bool, optional
        Boolean indicating if the .compile method of the underlying QuantumSession
        should be called before. The default is True.
    compilation_kwargs  : dict, optional
        Keyword arguments for the compile method. For more details check
        :meth:`QuantumSession.compile <qrisp.QuantumSession.compile>`. The default
        is ``{}``.
    subs_dic : dict, optional
        A dictionary of Sympy symbols and floats to specify parameters in the case
        of a circuit with unspecified, :ref:`abstract parameters<QuantumCircuit>`.
        The default is {}.
    circuit_preprocessor : Python function, optional
        A function which recieves a QuantumCircuit and returns one, which is applied
        after compilation and parameter substitution. The default is None.

    Returns
    -------
    list[float]
        The expected value of the Hamiltonians.

    """

    expectations = []
    n = len(hamiltonians)
    for i in range(n):
        expectations.append(hamiltonians[i].get_measurement(qarg,
                                precision=precision,
                                backend=backend,
                                shots=shots,
                                compile=compile,
                                compilation_kwargs=compilation_kwargs,
                                subs_dic=subs_dic,
                                circuit_preprocessor=circuit_preprocessor,
                                precompiled_qc=precompiled_qc,
                                _measurement=None if _measurements==None else _measurements[i]
                                ))

    return expectations

    


