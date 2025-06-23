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

def test_jasp_qswitch_case_list():
    from qrisp import QuantumFloat, h, qswitch, terminal_sampling
    import numpy as np
    
    # Some sample case functions
    def f0(x): x += 1
    def f1(x): x += 2
    def f2(x): x += 3
    def f3(x): x += 4 
    case_function_list = [f0, f1, f2, f3]
    
    @terminal_sampling
    def main():
        # Create operand and case variable
        operand = QuantumFloat(4)
        operand[:] = 1
        case = QuantumFloat(2)
        h(case)

        # Execute switch_case function
        qswitch(operand, case, case_function_list, "sequential")

        return operand
    
    meas_res = main()
    # {2.0: 0.25, 3.0: 0.25, 4.0: 0.25, 5.0: 0.25}
    
    for i in [2,3,4,5]:
        assert np.round(meas_res[i],2) == 0.25

    @terminal_sampling
    def main():
        # Create operand and case variable
        operand = QuantumFloat(4)
        operand[:] = 1
        case = QuantumFloat(2)
        h(case)

        # Execute switch_case function
        qswitch(operand, case, case_function_list, "tree")

        return operand
    
    meas_res = main()
    # {2.0: 0.25, 3.0: 0.25, 4.0: 0.25, 5.0: 0.25}
    
    for i in [2,3,4,5]:
        assert np.round(meas_res[i],2) == 0.25


def test_jasp_qswitch_case_function():
    from qrisp import QuantumFloat, h, qswitch, terminal_sampling, control
    import numpy as np
    
    # Some sample case function 
    def case_function(i, x):
        with control(i == 0):
            x += 1
        with control(i == 1):
            x += 2
        with control(i == 2):
            x += 3
        with control(i == 3):
            x += 4
    
    @terminal_sampling
    def main():
        # Create operand and case variable
        operand = QuantumFloat(4)
        operand[:] = 1
        case = QuantumFloat(2)
        h(case)

        # Execute switch_case function
        qswitch(operand, case, case_function, "sequential")

        return operand
    
    meas_res = main()
    # {2.0: 0.25, 3.0: 0.25, 4.0: 0.25, 5.0: 0.25}
    
    for i in [2,3,4,5]:
        assert np.round(meas_res[i],2) == 0.25

    @terminal_sampling
    def main():
        # Create operand and case variable
        operand = QuantumFloat(4)
        operand[:] = 1
        case = QuantumFloat(2)
        h(case)

        # Execute switch_case function
        qswitch(operand, case, case_function, "tree")

        return operand
    
    meas_res = main()
    # {2.0: 0.25, 3.0: 0.25, 4.0: 0.25, 5.0: 0.25}
    
    for i in [2,3,4,5]:
        assert np.round(meas_res[i],2) == 0.25
    

def test_jasp_qswitch_case_hamiltonian_simulation():
    from qrisp import QuantumFloat, h, qswitch, terminal_sampling
    import numpy as np
    from qrisp.operators import X,Y,Z

    H1 = Z(0)*Z(1)
    H2 = Y(0)+Y(1)
    
    # Some sample case functions
    def f0(x): H1.trotterization()(x)
    def f1(x): H2.trotterization()(x, t=np.pi/4)
    case_function_list = [f0, f1]
    
    @terminal_sampling
    def main():
        # Create operand and case variable
        operand = QuantumFloat(2)
        case = QuantumFloat(1)
        h(case)

        # Execute switch_case function
        qswitch(operand, case, case_function_list)

        return case, operand
    
    meas_res = main()
    
    assert np.round(meas_res[0,0],2) == 0.5
    for i in [0,1,2,3]:
        assert np.round(meas_res[1,i],3) == 0.125
        
def test_jasp_qswitch_inversion():
    
    from qrisp import QuantumFloat, qswitch, jaspify, measure
    import jax.numpy as jnp
    from jax import jit
    
    @jit
    def extract_boolean_digit(integer, digit):
        return (integer >> digit) & 1

    def fake_inversion(qf, precision, res_qf=None):
        if res_qf is None:
            res_qf = QuantumFloat(2*precision+1,-precision)
        def case_function(i, operand):
            curr = 2**(2*precision)//(jnp.maximum(i,1))
            operand[:] = curr/2**(2*precision)
        qswitch(res_qf, qf, case_function)
        return res_qf

    @jaspify
    def main():
        
        qf = QuantumFloat(5)
        qf[:] = 5
        inv = fake_inversion(qf, 5)
        return measure(inv)
        
    assert main() == 0.1875