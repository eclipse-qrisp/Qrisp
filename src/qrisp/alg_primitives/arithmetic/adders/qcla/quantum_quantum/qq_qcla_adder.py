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

import numpy as np

from qrisp.alg_primitives import x, cx
from qrisp.alg_primitives.arithmetic.adders.qcla.quantum_quantum.qq_carry_path import qq_calc_carry
from qrisp.alg_primitives.arithmetic.adders.qcla.quantum_quantum.qq_sum_path import qq_sum_path_direct_uncomputation, qq_sum_path
from qrisp.alg_primitives.arithmetic.adders.gidney import gidney_adder
from qrisp.environments import QuantumEnvironment, invert
from qrisp.core import QuantumVariable
from qrisp.circuit import fast_append
from qrisp.misc.utility import redirect_qfunction

verify_manual_uncomputations = np.zeros(1)


# This function performs the in-place addition 
# b += a
# based on the higher radix qcla
# The overall radix can be specified as an exponential of the form
# R = radix_base**radix_exponent
def qq_qcla(a, b, radix_base = 2, radix_exponent = 1, t_depth_reduction = True):
    
    if len(a) > len(b):
        raise Exception("Tried to add QuantumFloat of higher precision onto QuantumFloat of lower precision")
    
    with fast_append():
    # if True:
        R = radix_base**radix_exponent
        
        # The case that a only has a single qubit is simple.
        if len(b) == 1:
            cx(a[0],b[0])
            return
        elif len(b) <= R:
            qcla_anc = QuantumVariable(len(b) - len(a), name = "qcla_anc*", qs = b[0].qs())
            gidney_adder(list(a) + list(qcla_anc), b)
            qcla_anc.delete(verify = bool(verify_manual_uncomputations[0]))
            return
        
        # Calculate the carry
        # Executing within a QuantumEnvironemnt accelerates the uncomputation algorithm
        # because it doesn't have to consider the operations appended outside of this function
        with QuantumEnvironment():
            c = qq_calc_carry(a, b, radix_base, radix_exponent)
        
        if t_depth_reduction:
            qq_sum_path_direct_uncomputation(a,b,c,R)
        else:
            qq_sum_path(a,b,c,R)
            
            # To uncompute the carry we use Drapers strategy
            # CARRY(A,B) = CARRY(A, NOT(A+B))
            # We therefore bitflip the sum
            x(b)
            
            # Contrary to Draper's adder we don't need to uncompute every carry digit.
            # Because of the above equivalence, the carries agree on every digit, so especially
            # on the digits representing the output of the calc_carry function. We can therefore
            # uncompute using calc_carry (even with higher radix) by inverting calc_carry.
            
            with invert():
                #We use the redirect_qfunction decorator to steer the function onto c
                redirect_qfunction(qq_calc_carry)(a, b, radix_base, radix_exponent, target = c)
            
            # Flip the sum back
            x(b)
        
        # In the case R = 1 we can use automatic uncomputation which seems to perform better
        if R == 1:
            c.uncompute()
            return
        # Delete c
        c.delete(verify = bool(verify_manual_uncomputations[0]))
            
        
        