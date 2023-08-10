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

# Created by ann81984 at 28.06.2022
from qrisp import (
    QuantumDictionary,
    QuantumFloat,
    custom_qv,
    h,
    multi_measurement,
    QuantumVariable,
)

# This example demonstrates the use of a QuantumDictionary with flexible types


def test_quantum_dictionary():
    # Create a QuantumDictionary
    test_qd = QuantumDictionary()

    # Fill in some entries (works precisely like normal Python dictionaries)
    test_qd["hello"] = "hallo"
    test_qd["world"] = "welt"

    # Create Key QuantumVariable (the constructor takes a
    # list of values, which the QuantumVariable is supposed to represent)
    key_qv = custom_qv(["hello", "world"])

    # Bring into superposition
    h(key_qv)

    print(key_qv)
    assert str(key_qv) == "{'hello': 0.5, 'world': 0.5}"
    # Yields {'hello': 0.5, 'world': 0.5}

    # Use QuantumDictionary to "dereference" the values into a new QuantumVariable
    value_qv = test_qd[key_qv]

    # Evaluate values
    print(value_qv)
    assert str(value_qv) == "{'hallo': 0.5, 'welt': 0.5}"
    # Yields {'hallo': 0.5, 'welt': 0.5}

    # Now we demonstrate how a QuantumDictionary with a specified type can be created
    # The return values from dereferencing this QuantumDictionary are now of the type QuantumFloat(4, -1)
    float_qd = QuantumDictionary(return_type=QuantumFloat(4, -1))

    # Encode some values (performed as with usual Python dictionaries)
    float_qd.update({"hallo": 1.5, "welt": 0.5})

    # Dereference
    value_qf = float_qd[value_qv]

    # Results are indeed QuantumFloat
    print(type(value_qf))
    assert type(value_qf).__name__ == "QuantumFloat"

    # Usual arithmetic is possible
    value_qf += 2.5

    # Evaluate results
    assert value_qf.get_measurement() == {3.0: 0.5, 4.0: 0.5}
    # Yields {4.0: 0.5, 3.0: 0.5}

    test_qd[(1, 2)] = 73
    test_qd[(0, 2)] = 37
    qf1 = QuantumFloat(1)
    qf2 = QuantumFloat(2)
    h(qf1)
    qf2[:] = 2
    res = test_qd[(qf1, qf2)]
    assert multi_measurement([qf1, qf2, res]) == {(0, 2, 37): 0.5, (1, 2, 73): 0.5}

    qtype = QuantumFloat(4, -2, signed=True)

    float_qd = QuantumDictionary(return_type=qtype)

    float_qd["hello"] = 0.5
    float_qd["world"] = -1
    key_qv = QuantumVariable.custom([1, 42, "hello", "world"])
    key_qv[:] = {"hello": 1, "world": 1}

    qf = qtype.duplicate()
    float_qd.load(key_qv, qf)

    qf2 = qtype.duplicate()
    float_qd.load(key_qv, qf2, synth_method="pprm")

    assert qf2.get_measurement() == {0.5: 0.5, -1.0: 0.5}
