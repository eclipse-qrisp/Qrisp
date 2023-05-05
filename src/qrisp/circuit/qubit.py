"""
/*********************************************************************
* Copyright (c) 2023 the Qrisp Authors
*
* This program and the accompanying materials are made
* available under the terms of the Eclipse Public License 2.0
* which is available at https://www.eclipse.org/legal/epl-2.0/
*
* SPDX-License-Identifier: EPL-2.0
**********************************************************************/
"""


class Qubit:
    """
    This class describes qubits. Qubits are created by supplying the identifier string.

    Attributes
    ----------
    identifier : str
        A string to identify the Qubit.

    Examples
    --------

    We create a Qubit and add it to a :ref:`QuantumCircuit`:

    >>> from qrisp import QuantumCircuit, Qubit
    >>> qb = Qubit("alphonse")
    >>> qc = QuantumCircuit()
    >>> qc.add_qubit(qb)
    >>> qc.x(qb)
    >>> print(qc)
              ┌───┐
    alphonse: ┤ X ├
              └───┘


    """

    def __init__(self, identifier):
        if identifier == "qb_[127]":
            raise
        self.identifier = identifier
        self.hash_value = id(self)
        self.lock = False
        self.perm_lock = False

    def __str__(self):
        return self.identifier

    def __repr__(self):
        return "Qubit(" + self.identifier + ")"

    def __hash__(self):
        return self.hash_value

    def __eq__(self, other):
        return self.hash_value == other.hash_value
