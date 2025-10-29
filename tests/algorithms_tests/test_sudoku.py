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

from qrisp import *


def test_sudoku_checker():

    sudoku_board = np.array([[0, -1, 2, 3], [2, 3, 0, -1], [1, 0, 3, 2], [3, -1, 1, 0]])

    def element_distinctness(iterable):

        n = len(iterable)

        comparison_list = []

        for i in range(n):
            for j in range(i + 1, n):

                # If both elements are classical and agree, return a QuantumBool with False
                if not isinstance(iterable[i], QuantumVariable) and not isinstance(
                    iterable[j], QuantumVariable
                ):
                    if iterable[i] == iterable[j]:
                        res = QuantumBool()
                        res[:] = False
                        return res
                    else:
                        continue

                # If atleast one of the elements is quantum, do a comparison
                comparison_list.append(iterable[i] != iterable[j])

        if len(comparison_list) == 0:
            return None

        res = QuantumBool()

        mcx(comparison_list, res)

        # Using recompute here reduces the qubit count dramatically
        # More information here https://qrisp.eu/reference/Core/Uncomputation.html#recomputation
        for qbl in comparison_list:
            qbl.uncompute(recompute=False)

        return res

    # Receives a quantum array with the values for the empty fields and
    # returns a QuantumBool, that is True if the Sudoku solution is valid
    @auto_uncompute
    def check_sudoku_board(empty_field_values: QuantumArray):

        # Create a quantum array, that contains a mix of the classical and quantum values
        shape = sudoku_board.shape
        quantum_sudoku_board = np.zeros(shape=sudoku_board.shape, dtype="object")

        quantum_value_list = list(empty_field_values)

        # Fill the board
        for i in range(shape[0]):
            for j in range(shape[1]):
                if sudoku_board[i, j] == -1:
                    quantum_sudoku_board[i, j] = quantum_value_list.pop(0)
                else:
                    quantum_sudoku_board[i, j] = int(sudoku_board[i, j])

        # Go through the conditions that need to be satisfied
        check_values = []
        for i in range(4):

            # Rows
            check_values.append(element_distinctness(quantum_sudoku_board[i, :]))

            # Columns
            check_values.append(element_distinctness(quantum_sudoku_board[j, :]))

        # Squares
        top_left_square = quantum_sudoku_board[:2, :2].flatten()
        top_right_square = quantum_sudoku_board[2:, :2].flatten()
        bot_left_square = quantum_sudoku_board[:2, 2:].flatten()
        bot_right_square = quantum_sudoku_board[2:, 2:].flatten()

        check_values.append(element_distinctness(top_left_square))
        check_values.append(element_distinctness(top_right_square))
        check_values.append(element_distinctness(bot_left_square))
        check_values.append(element_distinctness(bot_right_square))

        # element_distinctness returns None if only classical values have been compared
        # Filter these out
        i = 0
        while i < len(check_values):
            if check_values[i] is None:
                check_values.pop(i)
                continue
            i += 1

        # Compute the result
        res = QuantumBool()
        mcx(check_values, res)

        return res

    #################
    # Test function #
    #################

    empty_field_values = QuantumArray(qtype=QuantumFloat(2), shape=(3))
    # Not a valid solution
    empty_field_values[:] = [1, 1, 1]

    test = check_sudoku_board(empty_field_values)
    assert test.get_measurement() == {False: 1.0}
    # {False: 1.0}

    empty_field_values = QuantumArray(qtype=QuantumFloat(2), shape=(3))
    # Valid solution
    empty_field_values[:] = [1, 1, 2]

    test = check_sudoku_board(empty_field_values)
    assert test.get_measurement() == {True: 1.0}
