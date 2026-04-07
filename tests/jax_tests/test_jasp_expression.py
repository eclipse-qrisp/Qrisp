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

from qrisp import *
from qrisp.jasp import *
import pytest


@pytest.mark.parametrize(
    "static_argnums", [(0, 1), [0, 1], (0,), [0], (1,), [1], (), [], 0, 1]
)
def test_jasp_static_argnums(static_argnums):
    """Test that the static_argnums argument of make_jaspr works correctly."""

    def main(num_qubits1, num_qubits2):

        qv1 = QuantumVariable(num_qubits1)
        qv2 = QuantumVariable(num_qubits2)

        x(qv1[0])
        x(qv2[0])

    jaspr = make_jaspr(main, static_argnums=static_argnums)(2, 3)
    assert len(jaspr.eqns) == 6
