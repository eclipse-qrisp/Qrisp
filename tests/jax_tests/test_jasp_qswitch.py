"""
\********************************************************************************
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
********************************************************************************/
"""

def test_jasp_qswitch():
    from qrisp import QuantumFloat, h, qswitch, terminal_sampling
    
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
        qswitch(operand, case, case_function_list)

        return operand
    
    assert main() == {2.0: 0.25, 3.0: 0.25, 4.0: 0.25, 5.0: 0.25}