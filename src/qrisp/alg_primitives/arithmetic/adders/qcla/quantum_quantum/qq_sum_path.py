# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:22:37 2023

@author: sea
"""

import numpy as np

from qrisp.core.gate_application_functions import x, cx
from qrisp.qtypes import QuantumVariable, QuantumBool
from qrisp.environments import invert
from qrisp.misc import redirect_qfunction
from qrisp.alg_primitives.arithmetic.adders.gidney import gidney_adder
from qrisp.alg_primitives.arithmetic.adders.qcla.quantum_quantum.qq_carry_path import qq_calc_carry
from qrisp.alg_primitives.arithmetic.adders.incrementation import lin_incr


def qq_sum_path(a, b, c, R):
    
    if R == 1:
        # If R = 1, we are in the case of Drapers QCLA:
        # https://arxiv.org/abs/quant-ph/0406142   
        # We can use the formula
        # S = A (+) B (+) C
        cx(c[:-1], b[1:])
        cx(a, b[:len(a)])
    else:
        i = 0
        # Execute addition using the corresponding carry values
        for i in range(len(c)+1):
            
            # Determine the radix qubits to perform the addition on
            a_block = a[R*i:R*(i+1)]
            b_block = b[R*i:R*(i+1)]
            
            # If a_block is empty we break the loop.
            # This doesn't necessarily imply that b_block is empty because
            # len(b) could be larger then len(a). In this case we execute an
            # increment block after this loop
            if not len(a_block):
                i = i-1
                break
            
            # If a_block and b_block have different sizes, we create a padding
            # variable to fill up the remaining entries
            if not len(a_block) == len(b_block):
                padding_var = QuantumVariable(len(b_block) - len(a_block))
                a_block = a_block + list(padding_var)
            # Perform Cuccarro addition
            if i == 0:
                # cuccaro_procedure(a.qs, a_block, b_block)
                
                gidney_adder(a_block, b_block)
            else:
                
                # cuccaro_procedure(a.qs, a_block, b_block, carry_in = c[i-1])
                gidney_adder(a_block, b_block, c[i-1])
            
            # Delete carry var
            try:
                padding_var.delete(verify = bool(verify_manual_uncomputations[0]))
            except NameError:
                pass
        
        
        # This loop treats the case that len(a) != len(b)
        # In this case we need to perform the increment function on the remaining
        # qubits of b (incase the carry is True)
        #We start at the index the last loop finished at
        for j in range(i+1, len(b)//R+1)[::-1]:
            
            b_block = b[R*j:R*(j+1)]
            
            if not len(b_block):
                continue
            
            #Perform incrementation function
            if R*j == 0:
                lin_incr(b_block, c_out = c[j])
            else:
                lin_incr(b_block, c_in = c[j-1], c_out = c[j])
            
# This function is a version of the sum path that requires no uncomputation after
# This can be achieved because the the carry_out of the i-th Gidney-Adder
# is used to uncompute the carry_in values of the i+1-th Gidney-adder.

# This however creates another problem: The i+1-th Gidney-adder requires it's carry-in
# at the end of it's computation. Therefore the i-th Gidney adder has to wait until
# the i+1-th finished. This basically reintroduces linear depth.
# BUT
# Not linear T-depth. This is because the right "wing" of the V-shaped part of the
# Gidney-Adder has T-depth 0 regardless of the input size. Since it is only these
# parts that need to wait, the overall T-depth is still logarithmic.

# To understand this function it can be helpfull to first understand the regular
# qq_sum_path. A lot of this function is executed here in reversed order.
# This is because the i-th Gidney-adder needs to wait until the i+1-th finished.
# Therefore if we append the instructions of the i-th after the i+1-th, this 
# "waiting time" is realized.
def qq_sum_path_direct_uncomputation(a, b, c, R):
    
    if R == 1:
        # If R = 1, we are in the case of Drapers QCLA:
        # https://arxiv.org/abs/quant-ph/0406142   
        # We can use the formula
        # S = A (+) B (+) C
        for i in range(len(c)):
            cx(c[i], b[i+1])
        for i in range(len(a)):
            cx(a[i], b[i])
            
        # For this case, c will be uncomputed in qq_qcla.
        return
    
    # This loop treats the case that len(a) != len(b)
    # In this case we need to perform the increment function on the remaining
    # qubits of b (incase the carry is True)
    # #We start at the index the last loop finished at
    if len(a) != len(b):
        
        # Perform the loop in reverse
        for j in range(int(np.ceil(len(a)/R)), len(b)//R+2)[::-1]:
            
            b_block = b[R*j:R*(j+1)]
            
            if not len(b_block):
                continue
            
            #Perform incrementation function
            if R*j == 0:
                lin_incr(b_block, c_out = c[j])
            else:
                if j < len(c):
                    # We use the c_out of the incrementor to uncompute the carry of
                    # the previous iteration
                    lin_incr(b_block, c[j-1], c_out = c[j])
                else:
                    lin_incr(b_block, c[j-1])
                    

    # Execute addition using the corresponding carry values
    for i in range(len(a)//R+1)[::-1]:
        # Determine the radix qubits to perform the addition on
        a_block = a[R*i:R*(i+1)]
        b_block = b[R*i:R*(i+1)]
        
        # If a_block is empty we break the loop.
        # This doesn't necessarily imply that b_block is empty because
        # len(b) could be larger then len(a). In this case we execute an
        # increment block after this loop
        if not len(a_block):
            i = i-1
            continue
        
        
        # If a_block and b_block have different sizes, we create a padding
        # variable to fill up the remaining entries
        if not len(a_block) == len(b_block):
            padding_var = QuantumVariable(len(b_block) - len(a_block))
            a_block = a_block + list(padding_var)
        
        
        # Perform Gidney addition
        if i == 0:
            
            # Use the c_out keyword of the Gidney-Adder to uncompute the carry of
            # the previous iteration
            if len(c):
                gidney_adder(a_block, b_block, c_out = c[i])
            else:
                gidney_adder(a_block, b_block)
        
        
        elif i < len(c):
            gidney_adder(a_block, b_block, c_in = c[i-1], c_out = c[i])
        else:
            gidney_adder(a_block, b_block, c_in = c[i-1])
            
        
        # Delete carry var
        try:
            padding_var.delete(verify = bool(verify_manual_uncomputations[0]))
        except NameError:
            pass
    