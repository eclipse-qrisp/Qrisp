# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
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

"""Tests the LCU function with Pauli Hamiltonian unitaries and coefficients."""

import numpy as np

from qrisp import LCU, QuantumVariable, prepare, terminal_sampling
from qrisp.operators import A, C, X, Y, Z


def test_operators_LCU():

    @terminal_sampling
    def main():

        H = 2 * X(0) * X(1) - Z(0) * Z(1)

        unitaries, coeffs = H.unitaries()

        def operand_prep():
            return QuantumVariable(2)

        def state_prep(case):
            prepare(case, np.sqrt(coeffs))

        qv = LCU(operand_prep, state_prep, unitaries)
        return qv

    res_dict = main()

    for k, v in res_dict.items():
        res_dict[k] = v**0.5
    for k, v in res_dict.items():
        res_dict[k] = v / res_dict[0]

    # print(res_dict)
    # {3: 2.0000002235174685, 0: 1.0}

    assert (np.abs(res_dict[0] - 1)) < 1e-4
    assert (np.abs(res_dict[3] - 2)) < 1e-4


def test_operators_LCU_non_hermitian():
    """Verify QubitOperator.unitaries() encodes non-Hermitian operators without hermitizing."""
    op = (1 + 2j) * X(0) * X(1) - 1j * Z(0) * Z(1)
    assert not op.is_hermitian()
    assert (A(0) + C(0)).is_hermitian()
    assert (A(0) - 0.5j * Y(0)).is_hermitian()
    assert not A(0).is_hermitian()

    @terminal_sampling
    def main():
        unitaries, coeffs = op.unitaries()

        def operand_prep():
            return QuantumVariable(2)

        def state_prep(case):
            prepare(case, np.sqrt(coeffs))

        return LCU(operand_prep, state_prep, unitaries)

    res_dict = main()
    # op |00> = (1 + 2j)|11> - 1j|00>; probabilities proportional to |1+2j|^2 = 5 and |-1j|^2 = 1
    assert np.isclose(res_dict[3], 5 / 6, atol=1e-4)
    assert np.isclose(res_dict[0], 1 / 6, atol=1e-4)
