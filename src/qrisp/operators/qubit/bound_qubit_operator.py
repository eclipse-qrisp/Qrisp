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
from qrisp.operators.qubit.bound_qubit_term import BoundQubitTerm
from qrisp.operators.qubit.measurement import get_measurement
from qrisp import h, sx, IterationEnvironment, conjugate, merge

import sympy as sp

threshold = 1e-9

#
# BoundQubitOperator
#


class BoundQubitOperator(Hamiltonian):
    r"""
    This class provides an efficient implementation of Pauli Hamiltonians acting on QuantumVariables, i.e.,
    Hamiltonians of the form

    .. math::

        H=\sum\limits_{j}\alpha_jP_j

    where $P_j=\prod_i\sigma_i^j$ is a Pauli product,
    and $\sigma_i^j\in\{I,X,Y,Z\}$ is the Pauli operator acting on qubit $i$.

    Parameters
    ----------
    terms_dict : dict, optional
        A dictionary representing a BoundQubitOperator.

    Examples
    --------

    A BoundQubitOperator can be specified conveniently in terms of ``X``, ``Y``, ``Z`` operators:

    ::

        from qrisp import QuantumVariable
        from qrisp.operators import BoundQubitOperator, X,Y,Z

        qv = QuantumVariable(2)
        H = 1+2*X(qv[0])+3*X(qv[0])*Y(qv[1])

    Yields $1+2X(qv.0)+3X(qv.0)Y(qv.1)$.

    """

    def __init__(self, terms_dict={}):
        self.terms_dict = terms_dict
        pass

    def len(self):
        return len(self.terms_dict)

    #
    # Printing
    #

    def _repr_latex_(self):
        # Convert the sympy expression to LaTeX and return it
        expr = self.to_expr()
        return f"${sp.latex(expr)}$"

    def __str__(self):
        # Convert the sympy expression to a string and return it
        expr = self.to_expr()
        return str(expr)

    def to_expr(self):
        """
        Returns a SymPy expression representing the operator.

        Returns
        -------
        expr : sympy.expr
            A SymPy expression representing the operator.

        """

        expr = 0
        for pauli, coeff in self.terms_dict.items():
            expr += coeff * pauli.to_expr()
        return expr

    #
    # Arithmetic
    #

    def __pow__(self, e):
        if self.len() == 1:
            if isinstance(e, int) and e >= 0:
                if e % 2 == 0:
                    return BoundQubitOperator({BoundQubitTerm(): 1})
                else:
                    return self
            else:
                raise TypeError(
                    "Unsupported operand type(s) for ** or pow(): "
                    + str(type(self))
                    + " and "
                    + str(type(e))
                )
        else:
            raise TypeError(
                "Unsupported operand type(s) for ** or pow(): "
                + str(type(self))
                + " and "
                + str(type(e))
            )

    def __add__(self, other):
        """
        Returns the sum of the operator self and other.

        Parameters
        ----------
        other : int, float, complex or BoundQubitOperator
            A scalar or a BoundQubitOperator to add to the operator self.

        Returns
        -------
        result : BoundQubitOperator
            The sum of the operator self and other.

        """

        if isinstance(other, (int, float, complex)):
            other = BoundQubitOperator({BoundQubitTerm(): other})
        if not isinstance(other, BoundQubitOperator):
            raise TypeError("Cannot add BoundQubitOperator and " + str(type(other)))

        res_terms_dict = {}

        for pauli, coeff in self.terms_dict.items():
            res_terms_dict[pauli] = res_terms_dict.get(pauli, 0) + coeff
            if abs(res_terms_dict[pauli]) < threshold:
                del res_terms_dict[pauli]

        for pauli, coeff in other.terms_dict.items():
            res_terms_dict[pauli] = res_terms_dict.get(pauli, 0) + coeff
            if abs(res_terms_dict[pauli]) < threshold:
                del res_terms_dict[pauli]

        result = BoundQubitOperator(res_terms_dict)
        return result

    def __sub__(self, other):
        """
        Returns the difference of the operator self and other.

        Parameters
        ----------
        other : int, float, complex or BoundQubitOperator
            A scalar or a BoundQubitOperator to substract from the operator self.

        Returns
        -------
        result : BoundQubitOperator
            The difference of the operator self and other.

        """

        if isinstance(other, (int, float, complex)):
            other = BoundQubitOperator({BoundQubitTerm(): other})
        if not isinstance(other, BoundQubitOperator):
            raise TypeError(
                "Cannot substract BoundQubitOperator and " + str(type(other))
            )

        res_terms_dict = {}

        for pauli, coeff in self.terms_dict.items():
            res_terms_dict[pauli] = res_terms_dict.get(pauli, 0) + coeff
            if abs(res_terms_dict[pauli]) < threshold:
                del res_terms_dict[pauli]

        for pauli, coeff in other.terms_dict.items():
            res_terms_dict[pauli] = res_terms_dict.get(pauli, 0) - coeff
            if abs(res_terms_dict[pauli]) < threshold:
                del res_terms_dict[pauli]

        result = BoundQubitOperator(res_terms_dict)
        return result

    def __rsub__(self, other):
        """
        Returns the difference of the operator other and self.

        Parameters
        ----------
        other : int, float, complex or BoundQubitOperator
            A scalar or a BoundQubitOperator to substract the operator self from.

        Returns
        -------
        result : BoundQubitOperator
            The difference of the operator other and self.

        """

        if isinstance(other, (int, float, complex)):
            other = BoundQubitOperator({BoundQubitTerm(): other})
        if not isinstance(other, BoundQubitOperator):
            raise TypeError(
                "Cannot substract BoundQubitOperator and " + str(type(other))
            )

        res_terms_dict = {}

        for pauli, coeff in self.terms_dict.items():
            res_terms_dict[pauli] = res_terms_dict.get(pauli, 0) - coeff
            if abs(res_terms_dict[pauli]) < threshold:
                del res_terms_dict[pauli]

        for pauli, coeff in other.terms_dict.items():
            res_terms_dict[pauli] = res_terms_dict.get(pauli, 0) + coeff
            if abs(res_terms_dict[pauli]) < threshold:
                del res_terms_dict[pauli]

        result = BoundQubitOperator(res_terms_dict)
        return result

    def __mul__(self, other):
        """
        Returns the product of the operator self and other.

        Parameters
        ----------
        other : int, float, complex or BoundQubitOperator
            A scalar or a BoundQubitOperator to multiply with the operator self.

        Returns
        -------
        result : BoundQubitOperator
            The product of the operator self and other.

        """

        if isinstance(other, (int, float, complex)):
            other = BoundQubitOperator({BoundQubitTerm(): other})
        if not isinstance(other, BoundQubitOperator):
            raise TypeError(
                "Cannot multipliy BoundQubitOperator and " + str(type(other))
            )

        res_terms_dict = {}

        for pauli1, coeff1 in self.terms_dict.items():
            for pauli2, coeff2 in other.terms_dict.items():
                curr_pauli, curr_coeff = pauli1 * pauli2
                res_terms_dict[curr_pauli] = (
                    res_terms_dict.get(curr_pauli, 0) + curr_coeff * coeff1 * coeff2
                )

        result = BoundQubitOperator(res_terms_dict)
        return result

    __radd__ = __add__
    __rmul__ = __mul__

    #
    # Inplace arithmetic
    #

    def __iadd__(self, other):
        """
        Adds other to the operator self.

        Parameters
        ----------
        other : int, float, complex or BoundQubitOperator
            A scalar or a BoundQubitOperator to add to the operator self.

        """

        if isinstance(other, (int, float, complex)):
            self.terms_dict[BoundQubitTerm()] = (
                self.terms_dict.get(BoundQubitTerm(), 0) + other
            )
            return self
        if not isinstance(other, BoundQubitOperator):
            raise TypeError("Cannot add BoundQubitOperator and " + str(type(other)))

        for pauli, coeff in other.terms_dict.items():
            self.terms_dict[pauli] = self.terms_dict.get(pauli, 0) + coeff
            if abs(self.terms_dict[pauli]) < threshold:
                del self.terms_dict[pauli]
        return self

    def __isub__(self, other):
        """
        Substracts other from the operator self.

        Parameters
        ----------
        other : int, float, complex or BoundQubitOperator
            A scalar or a BoundQubitOperator to substract from the operator self.

        """

        if isinstance(other, (int, float, complex)):
            self.terms_dict[BoundQubitTerm()] = (
                self.terms_dict.get(BoundQubitTerm(), 0) - other
            )
            return self
        if not isinstance(other, BoundQubitOperator):
            raise TypeError("Cannot add BoundQubitOperator and " + str(type(other)))

        for pauli, coeff in other.terms_dict.items():
            self.terms_dict[pauli] = self.terms_dict.get(pauli, 0) - coeff
            if abs(self.terms_dict[pauli]) < threshold:
                del self.terms_dict[pauli]
        return self

    def __imul__(self, other):
        """
        Multiplys other to the operator self.

        Parameters
        ----------
        other : int, float, complex or BoundQubitOperator
            A scalar or a BoundQubitOperator to multiply with the operator self.

        """

        if isinstance(other, (int, float, complex)):
            # other = BoundQubitOperator({BoundQubitTerm():other})
            for term in self.terms_dict:
                self.terms_dict[term] *= other
            return self

        if not isinstance(other, BoundQubitOperator):
            raise TypeError(
                "Cannot multipliy BoundQubitOperator and " + str(type(other))
            )

        res_terms_dict = {}

        for pauli1, coeff1 in self.terms_dict.items():
            for pauli2, coeff2 in other.terms_dict.items():
                curr_pauli, curr_coeff = pauli1 * pauli2
                res_terms_dict[curr_pauli] = (
                    res_terms_dict.get(curr_pauli, 0) + curr_coeff * coeff1 * coeff2
                )

        self.terms_dict = res_terms_dict
        return self

    #
    # Substitution
    #

    def subs(self, subs_dict):
        """

        Parameters
        ----------
        subs_dict : dict
            A dictionary with indices (int) as keys and numbers (int, float, complex) as values.

        Returns
        -------
        result : BoundQubitOperator
            The resulting BoundQubitOperator.

        """

        res_terms_dict = {}

        for pauli, coeff in self.terms_dict.items():
            curr_pauli, curr_coeff = pauli.subs(subs_dict)
            res_terms_dict[curr_pauli] = (
                res_terms_dict.get(curr_pauli, 0) + curr_coeff * coeff
            )

        result = BoundQubitOperator(res_terms_dict)
        return result

    #
    # Miscellaneous
    #

    def apply_threshold(self, threshold):
        """
        Removes all Pauli terms with coefficient absolute value below the specified threshold.

        Parameters
        ----------
        threshold : float
            The threshold for the coefficients of the Pauli terms.

        """

        delete_list = []
        for pauli, coeff in self.terms_dict.items():
            if abs(coeff) < threshold:
                delete_list.append(pauli)
        for pauli in delete_list:
            del self.terms_dict[pauli]

    def to_sparse_matrix(self):
        """
        Returns a matrix representing the operator.

        Returns
        -------
        M : scipy.sparse.csr_matrix
            A sparse matrix representing the operator.

        """

        import scipy.sparse as sp
        from scipy.sparse import kron as TP, csr_matrix

        I = csr_matrix([[1, 0], [0, 1]])

        def get_matrix(P):
            if P == "I":
                return csr_matrix([[1, 0], [0, 1]])
            if P == "X":
                return csr_matrix([[0, 1], [1, 0]])
            if P == "Y":
                return csr_matrix([[0, -1j], [1j, 0]])
            else:
                return csr_matrix([[1, 0], [0, -1]])

        def recursive_TP(keys, factor_dict):
            if len(keys) == 1:
                return get_matrix(factor_dict.get(keys[0], "I"))
            return TP(
                get_matrix(factor_dict.get(keys.pop(0), "I")),
                recursive_TP(keys, factor_dict),
            )

        factor_dicts = []
        coeffs = []

        keys = set()
        for pauli, coeff in self.terms_dict.items():
            curr_dict = pauli.factor_dict
            keys.update(set(curr_dict.keys()))
            factor_dicts.append(curr_dict)
            coeffs.append(coeff)

        keys = set()
        for item in factor_dicts:
            keys.update(set(item.keys()))
        keys = sorted(keys)
        dim = len(keys)

        m = len(coeffs)
        M = sp.csr_matrix((2**dim, 2**dim))
        for k in range(m):
            M += complex(coeffs[k]) * recursive_TP(keys.copy(), factor_dicts[k])

        return M

    def ground_state_energy(self):
        """
        Calculates the ground state energy (i.e., the minimum eigenvalue) of the operator classically.

        Returns
        -------
        E : float
            The ground state energy.

        """

        from scipy.sparse.linalg import eigsh

        M = self.to_sparse_matrix()
        # Compute the smallest eigenvalue
        eigenvalues, _ = eigsh(M, k=1, which="SA")  # 'SA' stands for smallest algebraic
        E = eigenvalues[0]

        return E

    #
    # Partitions
    #

    # Commutativity: Partitions the BoundQubitOperator into BoundQubitOperators with pairwise commuting BoundQubitTerms
    def commuting_groups(self):
        r"""
        Partitions the QubitOperator into QubitOperators with pairwise commuting terms. That is,

        .. math::

            H = \sum_{i=1}^mH_i

        where the terms in each $H_i$ are mutually commuting.

        Returns
        -------
        groups : list[QubitOperator]
            The partition of the Hamiltonian.

        """

        groups = []  # Groups of commuting BoundQubitTerms

        # Sorted insertion heuristic https://quantum-journal.org/papers/q-2021-01-20-385/pdf/
        sorted_terms = sorted(
            self.terms_dict.items(), key=lambda item: abs(item[1]), reverse=True
        )

        for pauli, coeff in sorted_terms:

            commute_bool = False
            if len(groups) > 0:
                for group in groups:
                    for pauli_, coeff_ in group.terms_dict.items():
                        commute_bool = pauli_.commute(pauli)
                        if not commute_bool:
                            break
                    if commute_bool:
                        group.terms_dict[pauli] = coeff
                        break
            if len(groups) == 0 or not commute_bool:
                groups.append(BoundQubitOperator({pauli: coeff}))

        return groups

    # Qubit-wise commutativity: Partitions the BoundQubitOperator into BoundQubitOperators with pairwise qubit-wise commuting BoundQubitTerms
    def commuting_qw_groups(self, show_bases=False):
        r"""
        Partitions the BoundQubitOperator into BoundQubitOperators with pairwise qubit-wise commuting terms. That is,

        .. math::

            H = \sum_{i=1}^mH_i

        where the terms in each $H_i$ are mutually qubit-wise commuting.

        Returns
        -------
        groups : list[BoundQubitOperator]
            The partition of the Hamiltonian.

        """

        groups = []  # Groups of qubit-wise commuting BoundQubitTerms
        bases = []  # Bases as BoundQubitTerms

        # Sorted insertion heuristic https://quantum-journal.org/papers/q-2021-01-20-385/pdf/
        sorted_terms = sorted(
            self.terms_dict.items(), key=lambda item: abs(item[1]), reverse=True
        )

        for pauli, coeff in sorted_terms:

            commute_bool = False
            if len(groups) > 0:
                n = len(groups)
                for i in range(n):
                    commute_bool = bases[i].commute_qw(pauli)
                    if commute_bool:
                        bases[i].update(pauli.factor_dict)
                        groups[i].terms_dict[pauli] = coeff
                        break
            if len(groups) == 0 or not commute_bool:
                groups.append(BoundQubitOperator({pauli: coeff}))
                bases.append(pauli.copy())

        if show_bases:
            return groups, bases
        else:
            return groups

    #
    # Measurement settings
    #

    def get_measurement(
        self,
        qarg,
        precision=0.01,
        backend=None,
        shots=1000000,
        compile=True,
        compilation_kwargs={},
        subs_dic={},
        precompiled_qc=None,
        measurement_data=None,  # measurement settings
    ):
        r"""
        This method returns the expected value of a Hamiltonian for the state of a quantum argument.

        Parameters
        ----------
        qarg : QuantumVariable, QuantumArray or list[QuantumVariable]
            The quantum argument to evaluate the Hamiltonian on.
        precision: float, optional
            The precision with which the expectation of the Hamiltonian is to be evaluated.
            The default is 0.01. The number of shots scales quadratically with the inverse precision.
        backend : BackendClient, optional
            The backend on which to evaluate the quantum circuit. The default can be
            specified in the file default_backend.py.
        shots : integer, optional
            The maximum amount of shots to evaluate the expectation of the Hamiltonian.
            The default is 1000000.
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
        precompiled_qc : QuantumCircuit, optional
            A precompiled quantum circuit.

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

        unbound_operator, qarg = self.unbind()

        return unbound_operator.get_measurement(
            qarg,
            precision=precision,
            backend=backend,
            compile=compile,
            compilation_kwargs=compilation_kwargs,
            subs_dic=subs_dic,
            precompiled_qc=precompiled_qc,
            measurement_data=measurement_data,
        )

    def unbind(self):
        from qrisp.operators import QubitTerm, QubitOperator

        participating_qubits = []
        for term, coeff in self.terms_dict.items():
            for qb in term.factor_dict.keys():
                if qb not in participating_qubits:
                    participating_qubits.append(qb)

        index_inv = {
            participating_qubits[i]: i for i in range(len(participating_qubits))
        }

        unbound_operator = 0

        for term, coeff in self.terms_dict.items():
            new_term_factor_dict = {}
            for qb, factor in term.factor_dict.items():
                new_term_factor_dict[index_inv[qb]] = factor

            unbound_operator += QubitOperator({QubitTerm(new_term_factor_dict): coeff})

        return unbound_operator, participating_qubits

    #
    # Trotterization
    #

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

            This function receives the following arguments:

            * qarg : QuantumVariable
                The quantum argument.
            * t : float, optional
                The evolution time $t$. The default is 1.
            * steps : int, optional
                The number of Trotter steps $N$. The default is 1.
            * iter : int, optional
                The number of iterations the unitary $U_1(t,N)$ is applied. The default is 1.

        """

        def change_of_basis(factor_dict):
            for qubit, axis in factor_dict.items():
                if axis == "X":
                    h(qubit)
                if axis == "Y":
                    sx(qubit)

        groups, bases = self.commuting_qw_groups(show_bases=True)

        def trotter_step(qarg, t, steps):

            # qubit to apply gphase for identity term
            if isinstance(qarg, list):
                qubit = qarg[0][0]
            else:
                qubit = qarg[0]

            for index, basis in enumerate(bases):
                with conjugate(change_of_basis)(basis.factor_dict):
                    for term, coeff in groups[index].terms_dict.items():
                        term.simulate(coeff * t / steps, qubit)

        def U(qarg, t=1, steps=1, iter=1):
            merge([qarg])
            with IterationEnvironment(qarg.qs, iter * steps):
                trotter_step(qarg, t, steps)

        return U
