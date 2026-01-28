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

from qrisp import inpl_mult, h, QuantumFloat, q_switch, multi_measurement, x

# classical indexed switch tests

def test_jasp_q_switch_classical_index():

    def main(index_val):

        def f0(x): pass
        def f1(x): x += 1
        def f2(x): x += 2
        def f3(x): x += 3
        branches = [f0, f1, f2, f3]
        operand = QuantumFloat(2)

        q_switch(index_val, branches, operand)
        return operand.get_measurement()

    for index_val in range(4):
        res = main(index_val)
        assert res[index_val] == 1.0


# quantum indexed switch tests

def test_jasp_q_switch_quantum_index():
    from qrisp import QuantumFloat, q_switch

    def main(index_val):

        def f0(x): pass
        def f1(x): x += 1
        def f2(x): x += 2
        def f3(x): x += 3
        index = QuantumFloat(2)
        index[:] = index_val
        branches = [f0, f1, f2, f3]
        operand = QuantumFloat(2)

        q_switch(index, branches, operand)
        return operand.get_measurement()

    for index_val in range(4):
        res = main(index_val)
        assert res[index_val] == 1.0


def test_q_switch_branches_list():

    # Some sample index functions
    def f0(x): x += 1
    def f1(x): inpl_mult(x, 3, treat_overflow = False)
    def f2(x): pass
    def f3(x): h(x[1])
    branches_list = [f0, f1, f2, f3]
    
    # Create operand and index variable
    operand = QuantumFloat(4)
    operand[:] = 1
    index = QuantumFloat(2)
    h(index)
    
    # Execute switch_index function
    q_switch(index, branches_list, operand, method = "sequential")
    
    # Simulate
    assert multi_measurement([index, operand]) == {(0, 2): 0.25, (1, 3): 0.25, (2, 1): 0.25, (3, 1): 0.125, (3, 3): 0.125}
    # Yields {(0, 2): 0.25, (1, 3): 0.25, (2, 1): 0.25, (3, 1): 0.125, (3, 3): 0.125}
    
    # Create operand and index variable
    operand = QuantumFloat(4)
    operand[:] = 1
    index = QuantumFloat(2)
    h(index)
    
    # Execute switch_index function
    q_switch(index, branches_list, operand, method = "parallel")
    
    # Simulate
    assert multi_measurement([index, operand]) == {(0, 2): 0.25, (1, 3): 0.25, (2, 1): 0.25, (3, 1): 0.125, (3, 3): 0.125}
    # Yields {(0, 2): 0.25, (1, 3): 0.25, (2, 1): 0.25, (3, 1): 0.125, (3, 3): 0.125}

    # Create operand and index variable
    operand = QuantumFloat(4)
    operand[:] = 1
    index = QuantumFloat(2)
    h(index)
    
    # Execute switch_index function
    q_switch(index, branches_list, operand, method = "tree")
    
    # Simulate
    assert multi_measurement([index, operand]) == {(0, 2): 0.25, (1, 3): 0.25, (2, 1): 0.25, (3, 1): 0.125, (3, 3): 0.125}
    # Yields {(0, 2): 0.25, (1, 3): 0.25, (2, 1): 0.25, (3, 1): 0.125, (3, 3): 0.125}


def test_q_switch_function():

    # Some sample index functions
    def f0(x): x += 1
    def f1(x): inpl_mult(x, 3, treat_overflow = False)
    def f2(x): pass
    def f3(x): h(x[1])
    def branches(i, x):
        if i == 0:
            f0(x)
        if i == 1:
            f1(x)
        if i == 2:
            f2(x)
        if i == 3:
            f3(x)
    
    # Create operand and index variable
    operand = QuantumFloat(4)
    operand[:] = 1
    index = QuantumFloat(2)
    h(index)
    
    # Execute switch_index function
    q_switch(index, branches, operand, method = "sequential")
    
    # Simulate
    assert multi_measurement([index, operand]) == {(0, 2): 0.25, (1, 3): 0.25, (2, 1): 0.25, (3, 1): 0.125, (3, 3): 0.125}
    # Yields {(0, 2): 0.25, (1, 3): 0.25, (2, 1): 0.25, (3, 1): 0.125, (3, 3): 0.125}
    
    # Create operand and index variable
    operand = QuantumFloat(4)
    operand[:] = 1
    index = QuantumFloat(2)
    h(index)
    
    # Execute switch_index function
    q_switch(index, branches, operand, method = "parallel")
    
    # Simulate
    assert multi_measurement([index, operand]) == {(0, 2): 0.25, (1, 3): 0.25, (2, 1): 0.25, (3, 1): 0.125, (3, 3): 0.125}
    # Yields {(0, 2): 0.25, (1, 3): 0.25, (2, 1): 0.25, (3, 1): 0.125, (3, 3): 0.125}

    # Create operand and index variable
    operand = QuantumFloat(4)
    operand[:] = 1
    index = QuantumFloat(2)
    h(index)
    
    # Execute switch_index function
    q_switch(index, branches, operand, method = "tree")
    
    # Simulate
    assert multi_measurement([index, operand]) == {(0, 2): 0.25, (1, 3): 0.25, (2, 1): 0.25, (3, 1): 0.125, (3, 3): 0.125}
    # Yields {(0, 2): 0.25, (1, 3): 0.25, (2, 1): 0.25, (3, 1): 0.125, (3, 3): 0.125}


def test_q_switch_list_cutoff(): 
        
    def branches(i, arg):
        for j, b in enumerate(bin(i)[:1:-1]):
            if b== "1":
                x(arg[j])

    branches_list = [lambda arg: branches(0,arg),
         lambda arg: branches(1,arg),
         lambda arg: branches(2,arg),
         lambda arg: branches(3,arg),
         lambda arg: branches(4,arg)]

    # Execute switch_index function
    for mode, r in zip([1,4, None], [1,4,5]):
        operand = QuantumFloat(4)
        index = QuantumFloat(4)
        h(index)
        q_switch(index, branches_list, operand, method = "tree", branch_amount=mode)
        res = multi_measurement([index, operand])

        for i in range(16):
            if r <= i:
                assert res[(i,0)] == 0.0625
            else:
                assert res[(i,i)] == 0.0625

    for mode, r in zip([1,4, None], [1,4,5]):
        operand = QuantumFloat(3)
        index = QuantumFloat(3)
        h(index)
        q_switch(index, branches_list, operand, method = "parallel", branch_amount=r)
        res = multi_measurement([index, operand])

        for i in range(8):
            if r <= i:
                assert res[(i,0)] == 0.0625*2
            else:
                assert res[(i,i)] == 0.0625*2


def test_q_switch_function_cutoff(): 
        
    def branches(i, arg):
        for j, b in enumerate(bin(i)[:1:-1]):
            if b== "1":
                x(arg[j])

    # Execute switch_index function
    for r in [4,7,8]:
        operand = QuantumFloat(4)
        index = QuantumFloat(4)
        h(index)
        q_switch(index, branches, operand, method = "tree", branch_amount=r)
        res = multi_measurement([index, operand])

        for i in range(16):
            if r <= i:
                assert res[(i,0)] == 0.0625
            else:
                assert res[(i,i)] == 0.0625

    for r in [3,5,6]:
        operand = QuantumFloat(3)
        index = QuantumFloat(3)
        h(index)
        q_switch(index, branches, operand, method = "parallel", branch_amount=r)
        res = multi_measurement([index, operand])

        for i in range(8):
            if r <= i:
                assert res[(i,0)] == 0.0625*2
            else:
                assert res[(i,i)] == 0.0625*2