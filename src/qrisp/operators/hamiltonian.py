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
from sympy import init_printing
# Initialize automatic LaTeX rendering
init_printing()

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
    
from abc import ABC, abstractmethod

class Hamiltonian(ABC):
    r"""
    Central structure to facilitate treatment of quantum Hamiltonians.

    Hamiltonians of the form

    .. math::
        
        H=\sum\limits_{j}\alpha_jP_j,
            
    where each $P_j=\prod_i\sigma_i^j$ is a Pauli product,
    and $\sigma_i^j\in\{I,X,Y,Z\}$ is the Pauli operator acting on qubit $i$,
    are specified in terms of Pauli ``X``, ``Y``, ``Z`` operators. 

    Examples
    --------

    We define a Hamiltonian:

    ::
    
        from qrisp.operators import X,Y,Z           
        H = X(0)*X(1)+Y(0)*Y(1)+Z(0)*Z(1)+0.5*Z(0)+0.5*Z(1)
        H

    Yields $X_0X_1+Y_0Y_1+0.5Z_0+Z_0Z_1+0.5Z_1$.

    We measure the expected value of the Hamiltonian for the state of a :ref:`QuantumVariable`.

    ::

        from qrisp import QuantumVariable
        qv = QuantumVariable(2)
        res = H.get_measurement(qv)
        print(res)
        #Yields 2.0
    
    """

    def __init__(self):
        pass

    #
    # Define abstract methods
    #

    @abstractmethod
    def _repr_latex_(self):
        """

        """
        pass

    @abstractmethod
    def __str__(self):
        """
        Returns a string representing the Hamiltonian.

        Returns
        -------
        str
            A string representing the Hamiltonian.

        """
        pass
    
    @abstractmethod
    def __add__(self, other):
        """
        Returns the sum of the operator self and other.

        Parameters
        ----------
        other : int, float, complex or Hamiltonian
            A scalar or a Hamiltonian to add to the operator self.

        Returns
        -------
        result : Hamiltonian
            The sum of the operator self and other.

        """
        pass

    @abstractmethod
    def __sub__(self, other):
        """
        Returns the difference of the operator self and other.

        Parameters
        ----------
        other : int, float, complex or Hamiltonian
            A scalar or a Hamiltonian to substract from the operator self.

        Returns
        -------
        result : Hamiltonian
            The difference of the operator self and other.

        """
        pass

    @abstractmethod
    def __mul__(self, other):
        """
        Returns the product of the operator self and other.

        Parameters
        ----------
        other : int, float, complex or Hamiltonian
            A scalar or a Hamiltonian to multiply with the operator self.

        Returns
        -------
        result : Hamiltonian
            The product of the operator self and other.

        """
        pass

    @abstractmethod
    def __iadd__(self, other):
        """
        Adds other to the operator self.

        Parameters
        ----------
        other : int, float, complex or Hamiltonian
            A scalar or a Hamiltonian to add to the operator self.

        """
        pass

    @abstractmethod
    def __isub__(self, other):
        """
        Substracts other from the operator self.

        Parameters
        ----------
        other : int, float, complex or Hamiltonian
            A scalar or a Hamiltonian to substract from the operator self.

        """
        pass

    @abstractmethod
    def __imul__(self,other):
        """
        Multiplys other to the operator self.

        Parameters
        ----------
        other : int, float, complex or PauliOperator
            A scalar or a Hamiltonian to multiply with the operator self.
        
        """
        pass

    @abstractmethod
    def apply_threshold(self, threshold):
        """
        Removes all terms with coefficient absolute value below the specified threshold.

        Parameters
        ----------
        threshold : float
            The threshold for the coefficients of the terms.

        """
        pass

    @abstractmethod
    def ground_state_energy(self):
        """
        Calculates the ground state energy (i.e., the minimum eigenvalue) of the Hamiltonian classically.
    
        Returns
        -------
        E : float
            The ground state energy. 

        """
        pass

    @abstractmethod
    def trotterization(self):
        r"""
        Returns a function for performing Hamiltonian simulation, i.e., approximately implementing the unitary operator $e^{itH}$ via Trotterization.

        Returns
        -------
        U : function 
            A Python function that implements the first order Suzuki-Trotter formula.
            Given a Hamiltonian $H=H_1+\dotsb +H_m$ the unitary evolution $e^{itH}$ is 
            approximated by 
            
            .. math::

                e^{itH}\approx U_1(t,N)=\left(e^{iH_1t/N}\dotsb e^{iH_mt/N}\right)^N

            This function recieves the following arguments:

            * qarg : QuantumVariable or QuantumArray
                The quantum argument.
            * t : float, optional
                The evolution time $t$. The default is 1.
            * steps : int, optional
                The number of Trotter steps $N$. The default is 1.
            * iter : int, optional 
                The number of iterations the unitary $U_1(t,N)$ is applied. The default is 1.
        
        """
        pass

    #
    # Evaluate expected value
    #
        
    def get_measurement(
        self,
        qarg,
        precision=0.01,
        method='QWC',
        backend=None,
        shots=100000,
        compile=True,
        compilation_kwargs={},
        subs_dic={},
        circuit_preprocessor=None,
        precompiled_qc = None,
        _measurement = None
    ):
        r"""
        This method returns the expected value of a Hamiltonian for the state of a quantum argument.

        Parameters
        ----------
        qarg : QuantumVariable, QuantumArray or list[QuantumVariable]
            The quantum argument to evaluate the Hamiltonian on.
        method : string, optional
            The method for evaluating the expected value of the Hamiltonian.
            Available is ``QWC``: Pauli terms are grouped based on qubit-wise commutativity.
            The default is ``QWC``.
        precision: float, optional
            The precision with which the expectation of the Hamiltonian is to be evaluated.
            The default is 0.01.
        backend : BackendClient, optional
            The backend on which to evaluate the quantum circuit. The default can be
            specified in the file default_backend.py.
        shots : integer, optional
            The maximum amount of shots to evaluate the expectation of the Hamiltonian. 
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

        We define a Hamiltonian, and measure its expected value for the state of a :ref:`QuantumVariable`.

        ::

            from qrisp import QuantumVariable, h
            from qrisp.operators import X,Y,Z
            qv = QuantumVariable(2)
            h(qv)
            H = Z(0)*Z(1)
            res = H.get_measurement(qv)
            print(res)
            #Yields 0.0

        We define a Hamiltonian, and measure its expected value for the state of a :ref:`QuantumArray`.

        ::

            from qrisp import QuantumVariable, QuantumArray, h
            from qrisp.operators import X,Y,Z
            qtype = QuantumVariable(2)
            q_array = QuantumArray(qtype, shape=(2))
            h(q_array)
            H = Z(0)*Z(1) + X(2)*X(3)
            res = H.get_measurement(q_array)
            print(res)
            #Yields 1.0

        """

        from qrisp import QuantumSession, merge

        if isinstance(qarg,QuantumVariable):
            if qarg.is_deleted():
                raise Exception("Tried to get measurement from deleted QuantumVariable")
            qs = qarg.qs
            
        elif isinstance(qarg,QuantumArray):
            for qv in qarg.flatten():
                if qv.is_deleted():
                    raise Exception(
                        "Tried to measure QuantumArray containing deleted QuantumVariables"
                    )
            qs = qarg.qs
        elif isinstance(qarg,list):
            qs = QuantumSession()
            for qv in qarg:
                if qv.is_deleted():
                    raise Exception(
                        "Tried to measure QuantumArray containing deleted QuantumVariables"
                    ) 
                merge(qs,qv.qs)

        if backend is None:
            if qs.backend is None:
                from qrisp.default_backend import def_backend

                backend = def_backend
            else:
                backend = qarg.qs.backend

        if len(qs.env_stack) != 0:
            raise Exception("Tried to get measurement within open environment")


        # Copy circuit in over to prevent modification
        if precompiled_qc is None:        
            if compile:
                qc = qompiler(
                    qs, **compilation_kwargs
                )
            else:
                qc = qs.copy()
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

        if _measurement is None:
            pauli_measurement = self.pauli_measurement()
        else:
            pauli_measurement = _measurement

        meas_circs = pauli_measurement.circuits
        meas_qubits = pauli_measurement.qubits
        meas_ops = pauli_measurement.operators_int
        meas_coeffs = pauli_measurement.coefficients
        meas_shots = pauli_measurement.shots

        meas_shots = [round(x/precision**2) for x in meas_shots]
        tot_shots = sum(x for x in meas_shots)
        if tot_shots>shots:
            meas_shots = [round(x*shots/tot_shots) for x in meas_shots]
            print("Warning: The total number of shots required " + str(tot_shots) +" for the target precision exceeds the allowed maxium.")

        N = len(meas_circs)

        expectation = 0

        for k in range(N):

            curr = qc.copy()
            curr.append(meas_circs[k].to_gate(), meas_qubits[k])

            res = get_measurement_from_qc(curr, meas_qubits[k], backend, meas_shots[k])
            
            # Groupings
            M = len(meas_ops[k])
            for l in range(M):
                for outcome,probability in res.items():
                    expectation += probability*evaluate_observable(meas_ops[k][l],outcome)*meas_coeffs[k][l]

        return expectation

    
    