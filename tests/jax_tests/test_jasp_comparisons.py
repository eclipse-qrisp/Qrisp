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

from qrisp import *

def test_comparisons():
    
    def lt_op(a, b):
        return a < b

    def gt_op(a, b):
        return a > b

    def le_op(a, b):
        return a <= b

    def ge_op(a, b):
        return a >= b
    
    def eq_op(a, b):
        return a == b
    
    def neq_op(a, b):
        return a != b
    
    
    op_list = [lt_op, gt_op, le_op, ge_op, eq_op, neq_op]
    
    for op in op_list:
    
        @terminal_sampling
        def main(i, j):
            
            a = QuantumFloat(i)
            b = QuantumFloat(j)
            
            h(a)
            h(b)
            
            return a, b, op(a, b)
        
        for i in range(1, 5):
            for j in range(1, 5):
                sampling_dict = main(i, j)
                
                for k in sampling_dict.keys():
                    
                    a, b, comparison_value = k
                    
                    assert op(a,b) == comparison_value
                    
        @terminal_sampling
        def main(i, j):
            
            a = QuantumFloat(i)
            h(a)
            
            return a, op(a, j)
        
        for i in range(1, 5):
            for j in range(1, 5):
                sampling_dict = main(i, j)
                
                for k in sampling_dict.keys():
                    a, comparison_value = k
                    
                    assert op(a,j) == comparison_value
                
        