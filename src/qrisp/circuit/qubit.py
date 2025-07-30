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

import numpy as np

qubit_hash = np.zeros(1)

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

    .. code-block:: none

                  ┌───┐
        alphonse: ┤ X ├
                  └───┘


    """

    __slots__ = [
        "hash_value",
        "qs",
        "identifier",
        "allocated",
        "recompute",
        "lock",
        "perm_lock",
        "bit_type"
    ]

    def __init__(self, identifier):
        self.identifier = identifier
        self.hash_value = int(qubit_hash[0])
        qubit_hash[0] += 1
        self.bit_type = 0
        self.lock = False
        self.perm_lock = False

    def __str__(self):
        return self.identifier

    def __repr__(self):
        return "Qubit(" + self.identifier + ")"

    def __hash__(self):
        return self.hash_value

    def __eq__(self, other):
        return self.hash_value == other.hash_value and self.bit_type == other.bit_type

    def __add__(self, other):
        if not isinstance(other, list):
            raise Exception(
                f"Tried to add Qubit to type {type(other)} (only list ist possible)"
            )
        return [self] + other

    def __radd__(self, other):
        if not isinstance(other, list):
            raise Exception(
                f"Tried to add Qubit to type {type(other)} (only list ist possible)"
            )
        return other + [self]
