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

from qrisp import *
from qrisp.jasp import *
from jax import make_jaxpr

def test_rus():
    
    @RUS
    def rus_trial_function():
        qf = QuantumFloat(5)
        h(qf[0])

        for i in range(1, 5):
            cx(qf[0], qf[i])

        cancelation_bool = measure(qf[0])
        return cancelation_bool, qf

    def call_RUS_example():

        qf = rus_trial_function()

        return measure(qf)

    jaspr = make_jaspr(call_RUS_example)()
    assert jaspr() == 31
    # Yields, 31 which is the decimal version of 11111
    
    # More complicated example
    
    def test_function():
        
        @RUS
        def trial_function():
            a = QuantumFloat(5)
            b = QuantumFloat(5)
            qbl = QuantumBool()
            a[:] = 10
            h(qbl[0])
            
            with control(qbl[0]):
                jasp_mod_adder(a, b, 7, inpl_adder = jasp_fourier_adder)
            
            return measure(qbl[0]), b
        
        
        res = trial_function()
        jasp_fourier_adder(5, res)
        
        return measure(res)
    jaspr = make_jaspr(test_function)()
    
    assert jaspr() == 8
    
    # Test LCU feature
    
    def case_function_0(x):
        pass

    def case_function_1(x):
        x += 1

    def case_function_2(x):
        x += 2

    def case_function_3(x):
        x += 3

    def case_function_4(x):
        x += 4

    case_function_list = [case_function_0, case_function_1, case_function_2, case_function_3]

    def state_preparation(qv):
        h(qv)


    # Encodes |3> + |4> + |5> + |6>
    def block_encoding():
        
        qf = QuantumFloat(3)
        qf[:] = 3
        
        case_indicator = QuantumFloat(2)
        case_indicator_qubits = [case_indicator[i] for i in range(2)]
        
        with conjugate(state_preparation)(case_indicator):
            for i in range(len(case_function_list)):
                with control(case_indicator_qubits, ctrl_state = i):
                    case_function_list[i](qf)
        
        return measure(case_indicator) == 0, qf


    @jaspify
    def main():
        
        qf = RUS(block_encoding)()
        
        return measure(qf)

    assert main() in [3,4,5,6]
    
    # Test static arguments
    
    def case_function_0(x):
        x += 3

    def case_function_1(x):
        x += 4

    def case_function_2(x):
        x += 5

    def case_function_3(x):
        x += 6

    case_functions = (case_function_0, 
                      case_function_1, 
                      case_function_2, 
                      case_function_3)

    def state_prep_full(qv):
        h(qv[0])
        h(qv[1])

    def state_prep_half(qv):
        h(qv[0])

    # Specify the corresponding arguments of the block encoding as "static",
    # i.e. compile time constants.

    @RUS
    def block_encoding(return_size, state_preparation, case_functions):
        
        # This QuantumFloat will be returned
        qf = QuantumFloat(return_size)
        
        # Specify the QuantumVariable that indicates, which
        # case to execute
        n = int(np.ceil(np.log2(len(case_functions))))
        case_indicator = QuantumFloat(n)
        
        # Turn into a list of qubits
        case_indicator_qubits = [case_indicator[i] for i in range(n)]
        
        # Perform the LCU protocoll
        with conjugate(state_preparation)(case_indicator):
            for i in range(len(case_functions)):
                with control(case_indicator_qubits, ctrl_state = i):
                    case_functions[i](qf)
        
        # Compute the success condition
        success_bool = (measure(case_indicator) == 0)
        
        return success_bool, qf

    @terminal_sampling
    def main():
        return block_encoding(4, state_prep_full, case_functions)
        
    assert main() == {3.0: 0.25, 4.0: 0.25, 5.0: 0.25, 6.0: 0.25}


    



