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

    def __init__(self, identifier):
        self.identifier = identifier

    def __str__(self):
        return self.identifier

    def __repr__(self):
        return self.identifier

    def __hash__(self):
        return hash(self.identifier)
