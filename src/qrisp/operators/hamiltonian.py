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

from qrisp import QuantumVariable, QuantumArray
from qrisp.core.compilation import qompiler

#
# Helper functions
#

def evaluate_observable(observable: int, x: int):
    """
    This method evaluates an observable that is a tensor product of Pauli-:math:`Z` operators
    with respect to a measurement outcome. 
        
    A Pauli operator of the form :math:`\prod_{i\in I}Z_i`, for some finite set of indices :math:`I\subset \mathbb N`, 
    is identified with an integer:
    We identify the Pauli operator with the binary string that has ones at positions :math:`i\in I`
    and zeros otherwise, and then convert this binary string to an integer.
        
    Parameters
    ----------
        
    observable : int
        The observable represented as integer.
     x : int 
        The measurement outcome represented as integer.
        
    Returns
    -------
    int
        The value of the observable with respect to the measurement outcome.
        
    """
        
    if bin(observable & x).count('1') % 2 == 0:
        return 1
    else:
        return -1  
    

class Hamiltonian:
    r"""  
    Wrapper class for representing different types of Hamiltonians.

    Parameters
    ----------
    hamiltonian : PauliOperator
        The Hermitian operator representing the Hamiltonian.
    
    """

    def __init__(self, hamiltonian):
        self.hamiltonian = hamiltonian

    def _repr_latex_(self):
        return self.hamiltonian._repr_latex_()
        
    def get_measurement(
        self,
        qarg,
        method=None,
        backend=None,
        shots=100000,
        compile=True,
        compilation_kwargs={},
        subs_dic={},
        circuit_preprocessor=None,
        precompiled_qc = None,
        mes_settings = None
    ):
        r"""
        This method returns the expected value of a Hamiltonian for the state of a quantum argument.

        Parameters
        ----------
        qarg : QuantumVariable or QuantumArray
            The quantum argunet to evaluate the Hamiltonian on.
        method : string, optional
            The method for evaluating the expected value of the Hamiltonian.
            Available is ``QWC``: Pauli terms are grouped based on qubit-wise commutativity.
            The default is None: The expected value of each Pauli term is computed independently.
        backend : BackendClient, optional
            The backend on which to evaluate the quantum circuit. The default can be
            specified in the file default_backend.py.
        shots : integer, optional
            The amount of shots to evaluate the circuit. The default is 100000.
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
        mes_settings : list, optional 
            The measurement settings for evaluating the expected value of the Hamiltonian. The default is None.
            Measurement settings are computed depending on the specified ``method``.

        Raises
        ------
        Exception
            If the containing QuantumSession is in a quantum environment, it is not
            possible to execute measurements.

        Returns
        -------
        float
            The expected value of the Hamiltonian.

        Examples
        --------

        We define a Hamiltonian, and measure its expected value for the state of a QuantumVariable.

        ::

            from qrisp import QuantumVariable, h
            from qrisp.operators import X,Y,Z
            qv = QuantumVariable(2)
            h(qv)
            H = Z(0)*Z(1)
            res = H().get_measurement(qv)
            print(res)
            #Yields 0.0

        We define a Hamiltonian, and measure its expected value for the state of a QuantumArray.

        ::

            from qrisp import QuantumVariable, QuantumArray, h
            from qrisp.operators import X,Y,Z
            qtype = QuantumVariable(2)
            q_array = QuantumArray(qtype, shape=(2))
            h(q_array)
            H = Z(0)*Z(1) + X(2)*X(3)
            res = H().get_measurement(q_array)
            print(res)
            #Yields 1.0

        """

        if isinstance(qarg,QuantumVariable):
            if qarg.is_deleted():
                raise Exception("Tried to get measurement from deleted QuantumVariable")
            qubits = qarg.reg
            
        elif isinstance(qarg,QuantumArray):
            for qv in qarg.flatten():
                if qv.is_deleted():
                    raise Exception(
                        "Tried to measure QuantumArray containing deleted QuantumVariables"
                    )
            qubits = sum([qv.reg for qv in qarg.flatten()], [])


        if backend is None:
            if qarg.qs.backend is None:
                from qrisp.default_backend import def_backend

                backend = def_backend
            else:
                backend = qarg.qs.backend

        if len(qarg.qs.env_stack) != 0:
            raise Exception("Tried to get measurement within open environment")


        # Copy circuit in over to prevent modification
        if precompiled_qc is None:        
            if compile:
                qc = qompiler(
                    qarg.qs, **compilation_kwargs
                )
            else:
                qc = qarg.qs.copy()
        else:
            qc = precompiled_qc.copy()

        # Bind parameters
        if subs_dic:
            qc = qc.bind_parameters(subs_dic)
            from qrisp.core.compilation import combine_single_qubit_gates
            qc = combine_single_qubit_gates(qc)

        # Execute user specified circuit_preprocessor
        if circuit_preprocessor is not None:
            qc = circuit_preprocessor(qc)

        qc = qc.transpile()

        from qrisp.misc import get_measurement_from_qc

        # Get measurement settings
        if mes_settings is None:
            meas_circs, meas_ops, meas_coeffs, constant_term = self.hamiltonian.get_measurement_settings(qarg, method=method)
        else:
            meas_circs, meas_ops, meas_coeffs, constant_term = mes_settings
        N = len(meas_circs)

        expectation = constant_term

        for k in range(N):

            curr = qc.copy()
            curr.append(meas_circs[k].to_gate(), qubits)
            res = get_measurement_from_qc(curr, qubits, backend, shots)
            
            # Allow groupings
            M = len(meas_ops[k])
            for l in range(M):
                for outcome,probability in res.items():
                    expectation += probability*evaluate_observable(meas_ops[k][l],outcome)*meas_coeffs[k][l]

        return expectation
    
    def ground_state_energy(self):
        """
        Calculates the ground state energy (i.e., the minimum eigenvalue) of the Hamiltonian classically.
        
        """
        return self.hamiltonian.ground_state_energy()
    
    def get_measurement_settings(self, qarg, method=None):
        """
        Returns the measurement settings to evaluate the Hamiltonian. 

        Parameters
        ----------
        qarg : QuantumVariable or QuantumArray
            The quantum argument the Hamiltonian is evaluated on.
        method : string, optional
            The method for evaluating the expected value of the Hamiltonian.
            Available is ``QWC``: Pauli terms are grouped based on qubit-wise commutativity.
            The default is ``None``: The expected value of each Pauli term is computed independently.

        Returns
        -------
        measurement_circuits : list[QuantumCircuit]
            The change of basis circuits.
        measurement_ops : list[list[int]]
            The Pauli products (after change of basis) to be measured represented as an integer.
        measurement_coeffs : list[list[float]]
            The coefficents of the Pauli products to be measured.
        constant_term : float
            The constant term.

        """
        return self.hamiltonian.get_measurement_settings(qarg, method)
    
    