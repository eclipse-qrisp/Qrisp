"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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
********************************************************************************/
"""


from qrisp.arithmetic import QuantumFloat
from qrisp.iterators.qrange import qRange
from qrisp import h, merge


def test_loop():
    from qrisp import QuantumFloat, qRange, h

    # Create some QuantumFloats
    n = QuantumFloat(3)
    qf = QuantumFloat(5)
    # Initialize the value 4
    n[:] = 4

    # Apply H-gate to 0-th qubit
    h(n[0])

    assert n.get_measurement() == {4.0: 0.5, 5.0: 0.5}

    # Perform successive addition of increasing numbers
    for i in qRange(n):
        qf += i

    assert qf.get_measurement() == {6.0: 0.5, 10.0: 0.5}
