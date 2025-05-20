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

from qrisp import inpl_mult, h, QuantumFloat, qswitch, multi_measurement, x
def test_qswitch_case_list():

    # Some sample case functions
    def f0(x): x += 1
    def f1(x): inpl_mult(x, 3, treat_overflow = False)
    def f2(x): pass
    def f3(x): h(x[1])
    case_function_list = [f0, f1, f2, f3]
    
    # Create operand and case variable
    operand = QuantumFloat(4)
    operand[:] = 1
    case = QuantumFloat(2)
    h(case)
    
    # Execute switch_case function
    qswitch(operand, case, case_function_list)
    
    # Simulate
    assert multi_measurement([case, operand]) == {(0, 2): 0.25, (1, 3): 0.25, (2, 1): 0.25, (3, 1): 0.125, (3, 3): 0.125}
    # Yields {(0, 2): 0.25, (1, 3): 0.25, (2, 1): 0.25, (3, 1): 0.125, (3, 3): 0.125}
    
    # Create operand and case variable
    operand = QuantumFloat(4)
    operand[:] = 1
    case = QuantumFloat(2)
    h(case)
    
    # Execute switch_case function
    qswitch(operand, case, case_function_list, method = "parallel")
    
    # Simulate
    assert multi_measurement([case, operand]) == {(0, 2): 0.25, (1, 3): 0.25, (2, 1): 0.25, (3, 1): 0.125, (3, 3): 0.125}
    # Yields {(0, 2): 0.25, (1, 3): 0.25, (2, 1): 0.25, (3, 1): 0.125, (3, 3): 0.125}

    # Create operand and case variable
    operand = QuantumFloat(4)
    operand[:] = 1
    case = QuantumFloat(2)
    h(case)
    
    # Execute switch_case function
    qswitch(operand, case, case_function_list, method = "tree")
    
    # Simulate
    assert multi_measurement([case, operand]) == {(0, 2): 0.25, (1, 3): 0.25, (2, 1): 0.25, (3, 1): 0.125, (3, 3): 0.125}
    # Yields {(0, 2): 0.25, (1, 3): 0.25, (2, 1): 0.25, (3, 1): 0.125, (3, 3): 0.125}

def test_qswitch_case_function():

    # Some sample case functions
    def f0(x): x += 1
    def f1(x): inpl_mult(x, 3, treat_overflow = False)
    def f2(x): pass
    def f3(x): h(x[1])
    def case_function_list(i, x):
        if i == 0:
            f0(x)
        if i == 1:
            f1(x)
        if i == 2:
            f2(x)
        if i == 3:
            f3(x)
    
    # Create operand and case variable
    operand = QuantumFloat(4)
    operand[:] = 1
    case = QuantumFloat(2)
    h(case)
    
    # Execute switch_case function
    qswitch(operand, case, case_function_list)
    
    # Simulate
    assert multi_measurement([case, operand]) == {(0, 2): 0.25, (1, 3): 0.25, (2, 1): 0.25, (3, 1): 0.125, (3, 3): 0.125}
    # Yields {(0, 2): 0.25, (1, 3): 0.25, (2, 1): 0.25, (3, 1): 0.125, (3, 3): 0.125}
    
    # Create operand and case variable
    operand = QuantumFloat(4)
    operand[:] = 1
    case = QuantumFloat(2)
    h(case)
    
    # Execute switch_case function
    qswitch(operand, case, case_function_list, method = "parallel")
    
    # Simulate
    assert multi_measurement([case, operand]) == {(0, 2): 0.25, (1, 3): 0.25, (2, 1): 0.25, (3, 1): 0.125, (3, 3): 0.125}
    # Yields {(0, 2): 0.25, (1, 3): 0.25, (2, 1): 0.25, (3, 1): 0.125, (3, 3): 0.125}

    # Create operand and case variable
    operand = QuantumFloat(4)
    operand[:] = 1
    case = QuantumFloat(2)
    h(case)
    
    # Execute switch_case function
    qswitch(operand, case, case_function_list, method = "tree")
    
    # Simulate
    assert multi_measurement([case, operand]) == {(0, 2): 0.25, (1, 3): 0.25, (2, 1): 0.25, (3, 1): 0.125, (3, 3): 0.125}
    # Yields {(0, 2): 0.25, (1, 3): 0.25, (2, 1): 0.25, (3, 1): 0.125, (3, 3): 0.125}



def test_qswitch_case_list_cutoff(): 
        
    def case_function_list(i, arg):
        for j, b in enumerate(bin(i)[:1:-1]):
            if b== "1":
                x(arg[j])

    l = [lambda arg: case_function_list(0,arg),
         lambda arg: case_function_list(1,arg),
         lambda arg: case_function_list(2,arg),
         lambda arg: case_function_list(3,arg),
         lambda arg: case_function_list(4,arg)]

    # Execute switch_case function
    for mode, r in zip([1,4, "auto"], [1,4,5]):
        operand = QuantumFloat(4)
        case = QuantumFloat(4)
        h(case)
        qswitch(operand, case, l, method = "tree", number=mode)
        res = multi_measurement([case, operand])

        for i in range(16):
            if r <= i:
                assert res[(i,0)] == 0.0625
            else:
                assert res[(i,i)] == 0.0625

def test_qswitch_case_function_cutoff(): 
        
    def case_function_list(i, arg):
        for j, b in enumerate(bin(i)[:1:-1]):
            if b== "1":
                x(arg[j])

    # Execute switch_case function
    for r in [4,7,8]:
        operand = QuantumFloat(4)
        case = QuantumFloat(4)
        h(case)
        qswitch(operand, case, case_function_list, method = "tree", number=r)
        res = multi_measurement([case, operand])

        for i in range(16):
            if r <= i:
                assert res[(i,0)] == 0.0625
            else:
                assert res[(i,i)] == 0.0625