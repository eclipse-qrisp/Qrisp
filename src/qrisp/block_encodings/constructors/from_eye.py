"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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

from qrisp.block_encodings.block_encoding_base import BlockEncoding
from qrisp.qtypes import QuantumBool


def build_from_eye(
    cls: BlockEncoding,
    diagonal_index: int = 0,
) -> BlockEncoding:
    r"""
    Constructs a BlockEncoding of a 2-D array with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    diagonal_index : int
        Index of the diagonal to set to one: 0 (the default) refers to the main diagonal,
        a positive value refers to an upper diagonal, and a negative value to a lower diagonal.

    Returns
    -------
    BlockEncoding
        A BlockEncoding representing an array where all elements are equal to zero,
        except for the $k$-th diagonal, whose values are equal to one.

    Examples
    --------

    ::

        from qrisp import *
        from qrisp.block_encodings import BlockEncoding

        # diagonal_index = 0: ones on the main diagonal
        BE1 = BlockEncoding.from_eye(diagonal_index=0)

        # diagonal_index = -2: ones on the second lower subdiagonal
        # (non-cyclic) shift |x> -> |x+2>
        BE2 = BlockEncoding.from_eye(diagonal_index=-2)

        BE3 = BE1.kron(BE2)

        def operand_prep():
            operand1 = QuantumFloat(3)
            operand2 = QuantumFloat(3)
            h(operand1)
            cx(operand1, operand2)
            return operand1, operand2

        @terminal_sampling
        def main():
            operand1, operand2 = BE3.apply_rus(operand_prep)()
            return operand1, operand2

        main()
        # {(0.0, 2.0): 0.16666666666666666,
        # (1.0, 3.0): 0.16666666666666666,
        # (2.0, 4.0): 0.16666666666666666,
        # (3.0, 5.0): 0.16666666666666666,
        # (4.0, 6.0): 0.16666666666666666,
        # (5.0, 7.0): 0.16666666666666666}

    """

    if diagonal_index == 0:
        return cls(1, [], lambda operand: None, is_hermitian=True)

    if diagonal_index > 0:
        # Shift |x> -> |x - diagonal_index> can be implemented as cyclic shift |x> -> |x - diagonal_index mod N>
        # followed by a comparator checking if x >= 2**n - diagonal_index.

        def unitary(*args):
            anc = args[0]
            operand = args[1]
            operand -= diagonal_index
            n = operand.size

            if diagonal_index == 1:
                # Comparator for x >= 2**n - 1 is equivalent to comparator for x == 2**n - 1.

                def comp(a, b):
                    return a == b

                injected_comp = anc << comp
                injected_comp(operand, 2**n - 1)

            else:

                def comp(a, b):
                    return a >= b

                injected_comp = anc << comp
                injected_comp(operand, 2**n - diagonal_index)

        return cls(1, [QuantumBool().template()], unitary)

    if diagonal_index < 0:
        # Shift |x> -> |x - diagonal_index> can be implemented as cyclic shift |x> -> |x - diagonal_index mod N>
        # followed by a comparator checking if x < -diagonal_index.

        def unitary(*args):
            anc = args[0]
            operand = args[1]
            operand -= diagonal_index

            if diagonal_index == -1:
                # Comparator for x < 1 is equivalent to comparator for x == 0.

                def comp(a, b):
                    return a == b

                injected_comp = anc << comp
                injected_comp(operand, 0)

            else:

                def comp(a, b):
                    return a < b

                injected_comp = anc << comp
                injected_comp(operand, -diagonal_index)

        return cls(1, [QuantumBool().template()], unitary)
