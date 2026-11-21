# ********************************************************************************
# * Copyright (c) 2024 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Defines factory functions X, Y, Z, A, C, P0, P1, I for single-qubit QubitOperator terms."""

from qrisp.operators.qubit.qubit_operator import QubitOperator
from qrisp.operators.qubit.qubit_term import QubitTerm


def X(arg: int) -> QubitOperator:
    r"""Returns a QubitOperator representing the Pauli $X$ operator acting on qubit ``arg``.

    Parameters
    ----------
    arg : int
        The index of the qubit the operator acts on.

    Returns
    -------
    QubitOperator
        The operator $X_{\text{arg}}$.

    """
    if isinstance(arg, int):
        return QubitOperator({QubitTerm({arg: "X"}): 1}, num_qubits=arg + 1)
    raise TypeError("Cannot initialize operator from type " + str(type(arg)))


def Y(arg: int) -> QubitOperator:
    r"""Returns a QubitOperator representing the Pauli $Y$ operator acting on qubit ``arg``.

    Parameters
    ----------
    arg : int
        The index of the qubit the operator acts on.

    Returns
    -------
    QubitOperator
        The operator $Y_{\text{arg}}$.

    """
    if isinstance(arg, int):
        return QubitOperator({QubitTerm({arg: "Y"}): 1}, num_qubits=arg + 1)
    raise TypeError("Cannot initialize operator from type " + str(type(arg)))


def Z(arg: int) -> QubitOperator:
    r"""Returns a QubitOperator representing the Pauli $Z$ operator acting on qubit ``arg``.

    Parameters
    ----------
    arg : int
        The index of the qubit the operator acts on.

    Returns
    -------
    QubitOperator
        The operator $Z_{\text{arg}}$.

    """
    if isinstance(arg, int):
        return QubitOperator({QubitTerm({arg: "Z"}): 1}, num_qubits=arg + 1)
    raise TypeError("Cannot initialize operator from type " + str(type(arg)))


def A(arg: int) -> QubitOperator:
    r"""Returns a QubitOperator representing the lowering (ladder) operator acting on qubit ``arg``.

    Parameters
    ----------
    arg : int
        The index of the qubit the operator acts on.

    Returns
    -------
    QubitOperator
        The operator $A_{\text{arg}}$.

    """
    return QubitOperator({QubitTerm({arg: "A"}): 1}, num_qubits=arg + 1)


def C(arg: int) -> QubitOperator:
    r"""Returns a QubitOperator representing the raising (ladder) operator acting on qubit ``arg``.

    Parameters
    ----------
    arg : int
        The index of the qubit the operator acts on.

    Returns
    -------
    QubitOperator
        The operator $C_{\text{arg}}$.

    """
    return QubitOperator({QubitTerm({arg: "C"}): 1}, num_qubits=arg + 1)


def P0(arg: int) -> QubitOperator:
    r"""Returns a QubitOperator representing the projector onto $\ket{0}$ on qubit ``arg``.

    Parameters
    ----------
    arg : int
        The index of the qubit the operator acts on.

    Returns
    -------
    QubitOperator
        The operator $P^0_{\text{arg}}$.

    """
    return QubitOperator({QubitTerm({arg: "P0"}): 1}, num_qubits=arg + 1)


def P1(arg: int) -> QubitOperator:
    r"""Returns a QubitOperator representing the projector onto $\ket{1}$ on qubit ``arg``.

    Parameters
    ----------
    arg : int
        The index of the qubit the operator acts on.

    Returns
    -------
    QubitOperator
        The operator $P^1_{\text{arg}}$.

    """
    return QubitOperator({QubitTerm({arg: "P1"}): 1}, num_qubits=arg + 1)


# Named like the other factor functions, despite the ambiguity E743 warns about.
def I(arg: int) -> QubitOperator:  # noqa: E743
    r"""Returns a QubitOperator representing the identity operator on qubit ``arg``.

    As an operator this is the identity, so multiplying by it changes nothing
    but :attr:`~QubitOperator.num_qubits`, which becomes at least ``arg + 1``.
    That specifies the amount of qubits an operator acts on beyond the ones
    its other factors touch.

    Parameters
    ----------
    arg : int
        The index of the qubit the operator acts on.

    Returns
    -------
    QubitOperator
        The operator $I_{\text{arg}}$.

    Examples
    --------
    A Pauli string on ten qubits, which acts non-trivially on qubits 3 and 4:

    >>> from qrisp.operators import I, X
    >>> P = X(3)*X(4)*I(9)
    >>> P
    X(3)*X(4)
    >>> P.num_qubits
    10

    """
    if isinstance(arg, int):
        return QubitOperator({QubitTerm(): 1}, num_qubits=arg + 1)
    raise TypeError("Cannot initialize operator from type " + str(type(arg)))
