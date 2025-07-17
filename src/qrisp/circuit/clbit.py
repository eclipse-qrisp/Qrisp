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


class Clbit:
    """
    This class represents classical bits. Classical bits are created by supplying the
    identifier string.

    Examples
    --------

    We create a Clbit and add it to a QuantumCircuit. After applying an Hadamard-Gate,
    we measure into the Clbit.

    >>> from qrisp import Clbit, QuantumCircuit
    >>> qc = QuantumCircuit(1)
    >>> qc.h(0)
    >>> mes_result = Clbit("mes_result")
    >>> qc.add_clbit(mes_result)
    >>> qc.measure(0, mes_result)
    >>> qc.run()
    {'0': 5000, '1': 5000}

    """

    dtype = np.dtype("bool")
    clbit_hash = np.zeros(1)

    def __init__(self, identifier):
        self.identifier = identifier
        self.hash_value = int(self.clbit_hash[0])
        self.bit_type = 1
        self.clbit_hash += 1

    def __str__(self):
        return self.identifier

    def __repr__(self):
        return "Clbit(" + self.identifier + ")"

    def __hash__(self):
        return self.hash_value

    def __eq__(self, other):
        return self.hash_value == other.hash_value and self.bit_type == other.bit_type
