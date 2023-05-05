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


from qrisp import QuantumDictionary, QuantumFloat, QuantumVariable, h

# This example demonstrates the use of a QuantumDictionary with flexible types

# Create a QuantumDictionary
test_qd = QuantumDictionary()

# Fill in some entries (works precisely like normal Python dictionaries)
test_qd["hello"] = "hallo"
test_qd["world"] = "welt"

key_qv = QuantumVariable.custom(["hello", "world"])

# Bring into superposition
h(key_qv)

print(key_qv)
# Yields {'hello': 0.5, 'world': 0.5}

# Use QuantumDictionary to "dereference" the values into a new QuantumVariable
value_qv = test_qd[key_qv]

# Evaluate values
print(value_qv)
# Yields {'hallo': 0.5, 'welt': 0.5}

# Now we demonstrate how a QuantumDictionary with a specified type can be created.
# The return values from dereferencing this QuantumDictionary are now of the type
# QuantumFloat(4, -1)
float_qd = QuantumDictionary(return_type=QuantumFloat(4, -1))

# Encode some values (performed as with usual Python dictionaries)
float_qd.update({"hallo": 1.5, "welt": 0.5})

# Dereference
value_qf = float_qd[value_qv]
# Results are indeed QuantumFloat
print(type(value_qf))

# Usual arithmetic is possible
value_qf += 2.5

# Evaluate results
print(value_qf)
# Yields {4.0: 0.5, 3.0: 0.5}
