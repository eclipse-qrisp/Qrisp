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

from qrisp.qtypes import QuantumFloat, QuantumModulus
from qrisp.alg_primitives.arithmetic.comparisons import less_than
from qrisp.alg_primitives.arithmetic.modular_arithmetic.mod_tools import modinv, montgomery_encoder
from qrisp.environments import custom_control
from qrisp.alg_primitives import cx, swap, mcx
from qrisp.misc.utility import bin_rep, redirect_qfunction
from qrisp.environments import control, invert
from qrisp.circuit import fast_append
from qrisp.permeability import auto_uncompute
# This file implements the techniques described in this paper: https://arxiv.org/abs/1801.01081
# The goal is to have performant modular multiplication. To this end, instead of taking the
# much explored path of creating a modular adder, that is used within the multiplication,
# the paper instead describes multiple techniques to perform the modulo reduction
# after the (non-modular) multiplication has concluded.

# In this file we implement the described techniques for Montgomery reduction.

def cl_montgomery_reduction(t, N, m):
    """
    This function serves as an example, how classical Montgomery reduction works.
    
    Note, that the return value is not t%N but (t*2**-m)%N. This is called Montgomery
    form and doesn't really create in any problems because it is just a matter of
    encoding.

    Parameters
    ----------
    t : integer
        The value to be reduced.
    N : integer
        The modulus.
    m : integer
        The Montgomery shift. Needs to satisfy t < N 2**n.

    Returns
    -------
    S : integer
        The result of the reduction: S = (t*2**-m)%N
    u : integer
        A garbage variable with the value u = (-tN**-1)%(2**m)

    """
    
    S = t
    u = int(0)
    for k in range(m):
        temp = S%2
        S -= temp*N
        S = S//2
        
        u = u | temp<<k
    
    if S < 0:
        S += N
    
    return S, u



# This function transfer the least significant bit from one QuantumFloat to another one
def transfer_lsb(from_qv, to_qv):
    
    # Get the qubit in question
    lsb = from_qv.reg.pop(0)
    
    # Adjust the relevant QuantumFloat attributes
    from_qv.size -= 1
    from_qv.msize -= 1
    from_qv.mshape[0] += 1
    from_qv.exponent += 1

    # Insert the qubit in the target
    to_qv.reg.insert(len(to_qv), lsb)
    
    # Adjust the relevant attributes
    to_qv.size += 1
    to_qv.msize += 1
    to_qv.mshape[1] += 1


# This function realizes the first step of the Montgomery reduction

def QREDC(t, N, m):
    # Create the variable holding u
    u = QuantumFloat(0, qs = t.qs, name = "u*")
    
    # Set an alias for S
    S = t
    
    # Perform the loop of the Montgomery reduction
    for k in range(m):
        
        # Set the alias similar to the classical version
        temp = S[0]
        
        # Transfer the least significant bit to u
        transfer_lsb(S, u)
        
        
        # Perform the step S -= temp*N
        
        # The subtraction only happens if S is odd. Since N
        # is odd by design, the new least significant qubit would always
        # ends up to be zero (this is why we can do the S = S/2 step)
        # Since we already transfered the least significant qubit to u,
        # we would have to create a new qubit to perform the subtraction
        # where we however know that this qubit will end up to be zero and
        # it would also immediately phased out again by the S = S/2 step.
        # Therefore instead of subtracting N, we subtract (N-1) on 
        # "truncated" S. 
        
        # To accomodate for the fact, that S is "truncated" ie. currently it's
        # least significant qubit has significance 2**1 = 2, we additionally divide
        # (N-1) by two.
        with control(temp):
            t.inpl_adder((-(N-1)//2), S)
        
        # Perform the S = S/2 step.        
        S.exp_shift(-1)
    
    
    
    # Perform the correction step
    # if S < 0:
    #     S += N
    with control(S.sign()):
        t.inpl_adder(N, S[:-1])
        
    
    # The following steps are not part of the original Montgomery reduction algorithm
    # but as described in the paper, they help with the uncomputation of u
    cx(S[0], S.sign())
    
    # Transfer the sign to u
    sgn = S.reg.pop(-1)
    S.size -= 1
    S.signed = False
    
    u.reg.insert(len(u), sgn)
    u.size += 1
    u.mshape[1] += 1
    
    # Adjust the m attribute, which indicates the current Montgomery shift of this
    # QuantumModulus
    S.m = S.m - m
    
    return S, u


# This function performs Montgomery reduction and subsequently uncomputes the u value
def montgomery_red(t, a, b, N, m, permeable_if_zero = False):
    
    # Perform Montgomery reduction
    t, u = QREDC(t, N, m)
    
    # Perform the uncomputation as described in the paper
    for k in range(len(a)):
        with control(a[k]):
            t.inpl_adder(-((2**k*b))*modinv(N, 2**(m+1)), u)
    
    if permeable_if_zero:    
        # cx(t[0], u[-1])
        pass
        # mcx([a[-1]] + [t[0]], u[-1], ctrl_state = "01")
        mcx(list(a) + [t[0]],  
            u[-1], ctrl_state = "0"*len(a) + "1",
            method = "balauca")
    
    # Delete the uncomputed value
    u.delete(verify = False)
    return t


# This function searches for a suitable Montgomery shift m
# On one hand, we want to have m as small as possible (because it requires extra
# qubits and iterations in the Montgomery reduction) but on the other hand we
# are limited by the requirement that a*b = t < N*2**m
def find_best_montgomery_shift(b, N):
    # The idea is now to try out increasingly large m, compute the corresponding
    # Montgomery form of the (classical) multiplication value b and check, if it
    # satisfies the above condition
    
    m = 0
    while True:
        b_trial = montgomery_encoder(int(b), 1<<m, N)%N
        # Since a is reduced, in the worst case, it is equal to N-1
        if (N-1)*b_trial < N*2**m:
            return m
        m += 1


# This function performs semi classical out of place multiplication follow by a
# Montgomery reduction
def montgomery_mod_semi_mul(a, b, output_qg = None, permeable_if_zero = False):
    
    # Set some aliases
    N = a.modulus
    
    # Reduce b
    b = b%N
    
    # Check some special cases that can be treated efficiently
    if b == 0:
        if output_qg is None:
            return a.duplicate()
        else:
            return output_qg
    if b == 1:
        if output_qg is None:
            return a.duplicate(init = True)
        else:
            a.inpl_adder(a, output_qg)
            return output_qg
    
    # Determine a suitable Montgomery shift
    m_shift = find_best_montgomery_shift(b, N)
    
    # Encode the Montgomery shift
    b_encoded = montgomery_encoder(b, 2**m_shift, N)%N
    
    # The idea here is to multiply the encoded b instead of original b
    # After Montgomery reduction, we have
    # (a*b_encoded*2**-m_shift)%N
    # = (a*b*2**m_shift*2**-m_shift)%N
    # = (a*b)%N
    
    # If no output value is given, create it
    if output_qg is None:
        t = QuantumFloat(a.size+m_shift, signed = True, qs = a.qs)
    else:
        if output_qg.modulus != N:
            raise Exception("Output QuantumModulus has incompatible modulus")
        
        # Extend the output to hold the overflow of the multiplication
        output_qg.extend(m_shift, 0)
        
        # Add the sign
        output_qg.add_sign()
        
        # Set the alias
        t = output_qg
        
    
    
    # Perform the non-modular multiplication
    
    n = int(np.ceil(np.log2(N)))
    b_encoded_string = bin_rep(b_encoded, n)[::-1]
    for i in range(n):
        if b_encoded_string[i] == "1":
            a.inpl_adder(a, t[i:])
    
    # Typecast to QuantumModulus and increase the Montgomery shift    
    from qrisp import QuantumModulus
    t.__class__ = QuantumModulus
    t.modulus = a.modulus
    t.m = a.m + m_shift
    t.inpl_adder = a.inpl_adder
    
    # Perform Montgomery reduction and return
    return montgomery_red(t, a, b_encoded, N, m_shift, permeable_if_zero = permeable_if_zero)


@custom_control
def ft_swap(a, b, ctrl = None):
    
    if ctrl is None:
        swap(a,b)
    else:
        cx(a, b)
        mcx([ctrl, b], a, method = "jones")
        cx(a, b)


# This function perform semi-classical in-place multiplication
@custom_control
def semi_cl_inpl_mult(a, X, ctrl = None, treat_invalid = False):
    
    # The idea here is to perform two semi-classical out of place multiplications,
    # where the second one uncomputes the input value.
    # After that, the variables are swapped, such that the input variable holds the
    # result and the temporary variable holds |0> and thus can be deleted.
    
    # An additional feature of this function is that it can be efficiently controlled.
    # We achieve this by controlling the swaps. If the control argument contains a 
    # QuantumBool that is set to True, another swap will be executed between the temporary
    # register and the input register.
    # This results in the fact that the first out of place multiplication will have 
    # no effect, because the input value of this multiplication is 0.
    # Before the second out of place multiplication happens, we perform another swap,
    # such that the input value of this multiplication is 0 as well.
    # Finally, we swap again, to restore the original input value into the input register.
    
    # Perform a reduction
    X = X%a.modulus
    
    
    # Check some special cases
    if X == 0:
        raise Exception("Tried to perform in-place multiplication with 0 (not invertible)")
    if X == 1:
        return a

    
    with fast_append(2):
        
        # Create the temporary value    
        tmp = a.duplicate(qs = a.qs)
        
        # If treat_invalid is set to True, the function should leave the invalid values
        # invariant. That is, values that are bigger than the modulus N.
        
        # We achieve this by computing a QuantumBool, which indicates wether the value is
        # invalid and the controlling the multiplication on this QuantumBool.
        if treat_invalid:
            a.__class__ = QuantumFloat
            reduced = a < a.modulus
            a.__class__ = QuantumModulus
            ctrl = [ctrl, reduced[0]]
        
        # If the controlled version of this function is required, we perform the swapping
        # strategy from above.
                
        if ctrl is not None:
            with control(ctrl, invert = True):
                for i in range(len(a)):
                    # swap(tmp[i], a[i])
                    ft_swap(tmp[i], a[i])


        
        # Perform the out of place multiplication
        tmp = montgomery_mod_semi_mul(a, 
                                      X, 
                                      output_qg = tmp,
                                      permeable_if_zero = ctrl is not None)
        
        # Perform the intermediate swap
        if ctrl is not None:
            with control(ctrl, invert = True):
                for i in range(len(a)):
                    # swap(tmp[i], a[i])
                    ft_swap(tmp[i], a[i])
        
        # The second out of place multiplication is the quantum inverted version
        # of the multiplication with the modular inverse
        # Why does this uncompute the input value?
        # Before the first out of place multiplication, we are in the state
        # |a>|0>
        # After the multiplication, we have
        # |a>|(a*X)%N>
        # If we rewrite this with b = (a*X)%N, we have
        # |(b*X**-1)%N>|b>
        # This could also be seen as the result of an out of place multiplication
        # from the tmp variable into the input variable by the number X**-1.
        # We can use this insight to uncompute the input variable by simply reverting
        # precisely this operation.
        
        with invert():
            a = montgomery_mod_semi_mul(tmp, 
                                        modinv(X, a.modulus), 
                                        output_qg = a, 
                                        permeable_if_zero = ctrl is not None)
        
        # Perform the corresponding swaps
        if ctrl is not None:
            with control(ctrl, invert = False):
                for i in range(len(a)):
                    # swap(tmp[i], a[i])
                    ft_swap(tmp[i], a[i])
        else:
            for i in range(len(a)):
                swap(tmp[i], a[i])
        
        # Delete the temporary variable
        tmp.delete(verify = False)
        
        # Uncompute the reduced bool    
        if treat_invalid:
            a.__class__ = QuantumFloat
            redirect_qfunction(less_than)(a,a.modulus,target= reduced)
            a.__class__ = QuantumModulus
            reduced.delete(verify = True)
        
        return a