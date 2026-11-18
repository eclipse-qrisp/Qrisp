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

"""Tests for QubitOperator."""

import numpy as np
import pytest

from qrisp import QuantumVariable
from qrisp.operators.qubit import X, Y, Z


def test_qubit_operator_mul_by_zero():
    """Test that multiplying a QubitOperator by zero results in an operator with an empty terms dict,
    rather than a dict with a single term with zero coefficient.
    """
    op = X(1) * 0
    assert op.terms_dict == {}
    assert not op.find_minimal_qubit_amount()

    op = 0 * X(1)
    assert op.terms_dict == {}
    assert not op.find_minimal_qubit_amount()

    op = X(1)
    op *= 0
    assert op.terms_dict == {}
    assert not op.find_minimal_qubit_amount()


@pytest.mark.parametrize(
    "operator",
    [Y(0), X(0) * Y(1), Y(0) * Z(1) + 0.5 * Y(0), Y(0) * Y(1) * X(2)],
)
def test_change_of_basis_reconstructs_operator(operator):
    """The gates U applied by change_of_basis and the returned diagonal operator D
    satisfy U^dg D U = operator, so measuring D after U measures the operator.
    """
    n = operator.find_minimal_qubit_amount()
    qv = QuantumVariable(n)
    diagonal_operator = operator.change_of_basis(qv)
    U = qv.qs.compile().get_unitary()
    assert np.allclose(U.conj().T @ diagonal_operator.to_array(n) @ U, operator.to_array(n))
