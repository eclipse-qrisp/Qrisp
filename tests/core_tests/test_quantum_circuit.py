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
import pytest

from qrisp import QuantumCircuit

# define the circuit
qc = QuantumCircuit(4)
qc.cx(0, range(1,4))

@pytest.mark.parametrize("method, args, expected_ops, expected_counts", [
    ("mcz", ([0, 1, 3]), 'mcz', {"cx": 3, "mcz": 1}),
    ("mcp", (0.3, [0, 1, 3]), 'mcp', {"cx": 3, "mcp": 1}),
    ("crz", (0.3, [0, 1], 3), 'mcrz', {"cx": 3, "mcrz": 2}),
])
def test_qc_gate_methods(method, args, expected_ops, expected_counts):
    """Check if gate is applied to the circuit."""
    getattr(qc, method)(*args)
    num_ops = qc.count_ops()
    assert num_ops[expected_ops] == expected_counts[expected_ops]
    assert num_ops['cx'] == expected_counts['cx']

@pytest.mark.parametrize("input_method", [("mcp"), ("crz")])
def test_invalid_phi(input_method):
    """Check if an error is raised when the input parameter is not of expected type."""
    with pytest.raises(ValueError, match = "Input parameter phi must be of type float or sympy.Symbol."):
       getattr(qc, input_method)(2, [0, 1], 3)