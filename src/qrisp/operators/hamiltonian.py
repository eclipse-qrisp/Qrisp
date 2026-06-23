"""********************************************************************************
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
********************************************************************************
"""

from abc import ABC, abstractmethod


class Hamiltonian(ABC):
    r"""Central structure to facilitate treatment of quantum Hamiltonians.

    For example, Hamiltonians of the form

    .. math::

        H=\sum\limits_{j}\alpha_jP_j,

    where each $P_j=\prod_i\sigma_i^j$ is a Pauli product,
    and $\sigma_i^j\in\{I,X,Y,Z\}$ is the Pauli operator acting on qubit $i$,
    are specified in terms of Pauli ``X``, ``Y``, ``Z`` operators.

    Examples
    --------
    We define a Hamiltonian:

    ::

        from qrisp.operators.qubit import X,Y,Z
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
        """ """
        pass

    @abstractmethod
    def __str__(self):
        """Returns a string representing the Hamiltonian.

        Returns
        -------
        str
            A string representing the Hamiltonian.

        """
        pass

    @abstractmethod
    def __add__(self, other):
        """Returns the sum of the operator self and other.

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
        """Returns the difference of the operator self and other.

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
    def __rsub__(self, other):
        """Returns the difference of the operator other and self.

        Parameters
        ----------
        other : int, float, complex or Hamiltonian
            A scalar or a Hamiltonian to substract the operator self from.

        Returns
        -------
        result : Hamiltonian
            The difference of the operator other and self.

        """
        pass

    @abstractmethod
    def __mul__(self, other):
        """Returns the product of the operator self and other.

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
        """Adds other to the operator self.

        Parameters
        ----------
        other : int, float, complex or Hamiltonian
            A scalar or a Hamiltonian to add to the operator self.

        """
        pass

    @abstractmethod
    def __isub__(self, other):
        """Substracts other from the operator self.

        Parameters
        ----------
        other : int, float, complex or Hamiltonian
            A scalar or a Hamiltonian to substract from the operator self.

        """
        pass

    @abstractmethod
    def __imul__(self, other):
        """Multiplys other to the operator self.

        Parameters
        ----------
        other : int, float, complex or PauliOperator
            A scalar or a Hamiltonian to multiply with the operator self.

        """
        pass

    @abstractmethod
    def apply_threshold(self, threshold):
        """Removes all terms with coefficient absolute value below the specified threshold.

        Parameters
        ----------
        threshold : float
            The threshold for the coefficients of the terms.

        """
        pass

    @abstractmethod
    def ground_state_energy(self):
        """Calculates the ground state energy (i.e., the minimum eigenvalue) of the Hamiltonian classically.

        Returns
        -------
        E : float
            The ground state energy.

        """
        pass
