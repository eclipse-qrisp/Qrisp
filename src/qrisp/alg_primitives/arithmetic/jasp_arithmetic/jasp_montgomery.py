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

from qrisp.alg_primitives.arithmetic.adders.gidney import gidney_adder
from qrisp.qtypes import QuantumFloat
from qrisp.jasp import jrange, check_for_tracing_mode
from qrisp.environments import control, invert
from qrisp.alg_primitives.arithmetic.modular_arithmetic.mod_tools import modinv
from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger, bi_modinv
from qrisp.core import swap, cx
import jax.numpy as jnp
import jax
import numpy as np

def compute_aux_radix_exponent(modulos, n):
    """
    Compute the exponent of the auxiliary radix for the montgomery reduction.

    Parameters
    ----------
    modulos : int
        The original modulos
    n : int
        The qubit size of the QuantumFloat
        
    Returns
    ----------
    The exponent of the $m$ of the auxiliary radix $R=2^m$.

    """
    #if check_for_tracing_mode():
    #    return (jnp.ceil(jnp.log2((modulos - 1) ** 2) + 1) - n).astype(jnp.int64)
    #else:
    #    return int(np.ceil(np.log2((modulos - 1) ** 2) + 1) - n)

    if check_for_tracing_mode():
        return (jnp.ceil(jnp.log2(modulos))).astype(jnp.int64)
    else:
        return int(np.ceil(np.log2(modulos)))

def q_montgomery_reduction(qf: QuantumFloat, N: int, m: int, inpl_adder=gidney_adder):
    """
    Perform the montgomery reduction of a QuantumFloat in-place

    ::

        qf = qf*2**(-m) mod N

    Parameters
    ----------
    qf : QuantumFloat
        The QuantumFloat of which to compute the montgomery reduction.
    N : int
        Original modulos of the montgomery reduction.
    m : int
        Exponent $m$ of the auxiliary radix $R=2^m$, see ``compute_aux_radix_exponent``.
    inpl_adder : Callable
        In-place adder to use during computation, defaults to ``gidney_adder``.
        
    """
      
    if check_for_tracing_mode():
        xrange = jrange
        get_size = lambda arg: arg.size
    else:
        xrange = range
        def get_size(arg) -> int:
            if isinstance(arg, list):
                return len(arg)
            else:
                return arg.size

    #Estimation stage
    for j in xrange(m):
        with control(qf[j]):
            inpl_adder(-(N-1)//2, qf[j+1:])


    #Correction stage
    with control(qf[-1]):
        inpl_adder(N, qf[m:-1])
    
    for i in xrange(get_size(qf)-m-+1):
        swap(qf[-1-i], qf[-1-i-1])
    cx(qf[m+1], qf[m])

def q_montgomery_reduction_bi(qf: QuantumFloat, N: BigInteger, m: int, inpl_adder=gidney_adder):
    """
    Perform the montgomery reduction of a QuantumFloat in-place

    ::

        qf = qf*2**(-m) mod N

    Parameters
    ----------
    qf : QuantumFloat
        The QuantumFloat of which to compute the montgomery reduction.
    N : int
        Original modulos of the montgomery reduction.
    m : int
        Exponent $m$ of the auxiliary radix $R=2^m$, see ``compute_aux_radix_exponent``.
    inpl_adder : Callable
        In-place adder to use during computation, defaults to ``gidney_adder``.
        
    """
      
    if check_for_tracing_mode():
        xrange = jrange
        get_size = lambda arg: arg.size
    else:
        xrange = range
        def get_size(arg) -> int:
            if isinstance(arg, list):
                return len(arg)
            else:
                return arg.size

    #Estimation stage
    for j in xrange(m):
        with control(qf[j]):
            with invert():
                inpl_adder((N-1) >> 1, qf[j+1:])


    #Correction stage
    with control(qf[-1]):
        inpl_adder(N, qf[m:-1])
    
    for i in xrange(get_size(qf)-m-+1):
        swap(qf[-1-i], qf[-1-i-1])
    cx(qf[m+1], qf[m])

def cq_montgomery_multiply_inplace_bi(X: BigInteger, y: QuantumFloat, N: BigInteger, m: int, inpl_adder=gidney_adder):
    """
    Perform the montgomery product of an integer and a QuantumFloat in-place

    ::

        X*y*2**(-m) mod N

    Parameters
    ----------
    X : int
        Integer factor of the montgomery product.
    y : QuantumFloat
        Quantum factor of the montgomery product.
    N : int
        Original modulos of the montgomery reduction.
    m : int
        Exponent $m$ of the auxiliary radix $R=2^m$, see ``compute_aux_radix_exponent``.
    inpl_adder : Callable
        In-place adder to use during computation
    """

    if check_for_tracing_mode():
        xrange = jrange
        get_size = lambda arg: arg.size
    else:
        xrange = range
        def get_size(arg) -> int:
            if isinstance(arg, list):
                return len(arg)
            else:
                return arg.size
    n = get_size(y)
    res = QuantumFloat(n)
    aux = QuantumFloat(m+1)
    wqf = aux[:] + res[:]


    # Multiplication

    @jax.jit
    def compute_val(X, j, N):
        return (X << j)%N
    
    for i in xrange(n):
        j = n - i - 1
        with control(y[j]):
            #inpl_adder((2**j*X)%N, wqf)
            #inpl_adder((X << j)%N, wqf)
            inpl_adder(compute_val(X, j, N), wqf)

    # Reduction
    q_montgomery_reduction_bi(wqf, N, m, inpl_adder=inpl_adder)

    # Uncomputation

    @jax.jit
    def compute_val(N, m):
        return bi_modinv(N, BigInteger.from_int(1, N.digits.shape[0]) << m+1)
    N1 = compute_val(N, m)

    @jax.jit
    def compute_val(X, j, N, N1):
        return ((X << j)%N)*N1

    for i in xrange(n):
        j = n - i - 1
        with control(y[j]):
            #inpl_adder(-((2**j*X)%N)*N1, aux[:])
            with invert():
                #inpl_adder(((X << j)%N)*N1, aux[:])
                inpl_adder(compute_val(X, j, N, N1), aux[:])


    # Swap res and y and multiply with X^{-1}
    for i in xrange(n):
        swap(y[i], res[i])

    # Inverse multiplication with inverse of X
    @jax.jit
    def compute_val(X, N):
        return (bi_modinv(X, N))
    X1 = compute_val(X, N)


    with invert():
        
        @jax.jit
        def compute_val(X1, j, N):
            return (X1 << j)%N
        # Multiplication
        for i in xrange(n):
            j = n - i - 1
            with control(y[j]):
                #inpl_adder((X1 << j)%N, wqf)
                inpl_adder(compute_val(X1, j, N), wqf)
    
        # Reduction
        q_montgomery_reduction_bi(wqf, N, m, inpl_adder=inpl_adder)
    
        @jax.jit
        def compute_val(X1, j, N, N1):
            return ((X1 << j)%N)*N1

        # Uncomputation
        for i in xrange(n):
            j = n - i - 1
            with control(y[j]):
                #inpl_adder(-((2**j*X1)%N)*N1, aux[:])
                with invert():
                    #inpl_adder(((X1 << j)%N)*N1, aux[:])
                    inpl_adder(compute_val(X1, j, N, N1), aux[:])
    
    aux.delete()
    res.delete()


    
def cq_montgomery_multiply(X: int, y: QuantumFloat, N: int, m: int, inpl_adder=gidney_adder):
    """
    Perform the montgomery product of an integer and a QuantumFloat

    ::

        X*y*2**(-m) mod N

    Parameters
    ----------
    X : int
        Integer factor of the montgomery product.
    y : QuantumFloat
        Quantum factor of the montgomery product.
    N : int
        Original modulos of the montgomery reduction.
    m : int
        Exponent $m$ of the auxiliary radix $R=2^m$, see ``compute_aux_radix_exponent``.
    inpl_adder : Callable
        In-place adder to use during computation
        
    Returns
    ----------
    QuantumFloat
        The mongomery product of the inputs.
    """

    if check_for_tracing_mode():
        xrange = jrange
        get_size = lambda arg: arg.size
    else:
        xrange = range
        def get_size(arg) -> int:
            if isinstance(arg, list):
                return len(arg)
            else:
                return arg.size
    n = get_size(y)
    res = QuantumFloat(n)
    aux = QuantumFloat(m+1)
    wqf = aux[:] + res[:]


    # Multiplication
    for i in xrange(n):
        j = n - i - 1
        with control(y[j]):
            inpl_adder((2**j*X)%N, wqf)

    # Reduction
    q_montgomery_reduction(wqf, N, m, inpl_adder=inpl_adder)

    # Uncomputation
    N1 = modinv(N, 2**(m+1))

    for i in xrange(n):
        j = n - i - 1
        with control(y[j]):
            inpl_adder(-((2**j*X)%N)*N1, aux[:])

    aux.delete()

    return res

def cq_montgomery_multiply_inplace(X: int, y: QuantumFloat, N: int, m: int, inpl_adder=gidney_adder):
    """
    Perform the montgomery product of an integer and a QuantumFloat in-place

    ::

        X*y*2**(-m) mod N

    Parameters
    ----------
    X : int
        Integer factor of the montgomery product.
    y : QuantumFloat
        Quantum factor of the montgomery product.
    N : int
        Original modulos of the montgomery reduction.
    m : int
        Exponent $m$ of the auxiliary radix $R=2^m$, see ``compute_aux_radix_exponent``.
    inpl_adder : Callable
        In-place adder to use during computation
    """

    if check_for_tracing_mode():
        xrange = jrange
        get_size = lambda arg: arg.size
    else:
        xrange = range
        def get_size(arg) -> int:
            if isinstance(arg, list):
                return len(arg)
            else:
                return arg.size
    n = get_size(y)
    res = QuantumFloat(n)
    aux = QuantumFloat(m+1)
    wqf = aux[:] + res[:]


    # Multiplication
    for i in xrange(n):
        j = n - i - 1
        with control(y[j]):
            inpl_adder((2**j*X)%N, wqf)

    # Reduction
    q_montgomery_reduction(wqf, N, m, inpl_adder=inpl_adder)

    # Uncomputation
    N1 = modinv(N, 2**(m+1))

    for i in xrange(n):
        j = n - i - 1
        with control(y[j]):
            inpl_adder(-((2**j*X)%N)*N1, aux[:])


    # Swap res and y and multiply with X^{-1}
    for i in xrange(n):
        swap(y[i], res[i])

    # Inverse multiplication with inverse of X
    X1 = (modinv(X, N))

    with invert():
        # Multiplication
        for i in xrange(n):
            j = n - i - 1
            with control(y[j]):
                inpl_adder((2**j*X1)%N, wqf)
    
        # Reduction
        q_montgomery_reduction(wqf, N, m, inpl_adder=inpl_adder)
    
        # Uncomputation
        for i in xrange(n):
            j = n - i - 1
            with control(y[j]):
                inpl_adder(-((2**j*X1)%N)*N1, aux[:])
    
    aux.delete()
    res.delete()

def qq_montgomery_multiply(x: QuantumFloat, y: QuantumFloat, N: int, m: int, inpl_adder=gidney_adder):
    """
    Perform the montgomery product of two QuantumFloats

    ::

        x*y*2**(-m) mod N

    Parameters
    ----------
    x : QuantumFloat
        First factor of the montgomery product.
    y : QuantumFloat
        Second factor of the montgomery product.
    N : int
        Original modulos of the montgomery reduction.
    m : int
        Exponent $m$ of the auxiliary radix $R=2^m$, see ``compute_aux_radix_exponent``.
    inpl_adder : Callable
        In-place adder to use during computation
        
    Returns
    ----------
    QuantumFloat
        The mongomery product of the inputs.
    """

    if check_for_tracing_mode():
        xrange = jrange
        get_size = lambda arg: arg.size
    else:
        xrange = range
        def get_size(arg) -> int:
            if isinstance(arg, list):
                return len(arg)
            else:
                return arg.size
            
    def qq_mul(ox: QuantumFloat, oy: QuantumFloat, ores: QuantumFloat):
        for i in xrange(get_size(oy)):
            with control(oy[i]):
                inpl_adder(ox[:], ores[i:])

    def qc_mul_inplace(operand, cl_int):
        size = get_size(operand)
        for i in xrange(size - 1):
            with control(operand[size - 2 - i]):
                inpl_adder(cl_int // 2, operand[size - 1 - i :])

            
    n = get_size(y)
    res = QuantumFloat(n)
    aux = QuantumFloat(m + 1)
    wqf = aux[:] + res[:]
    
    qq_mul(x, y, wqf[:-1])
    q_montgomery_reduction(wqf, N, m, inpl_adder=inpl_adder)
    qc_mul_inplace(aux, N)
    with invert():
        qq_mul(x, y, aux[:])
    aux.delete()

    return res

def montgomery_product(x, y, N, m, inpl_adder=gidney_adder):
    """
    Perform the montgomery product of the two inputs

    ::

        x*y*2**(-m) mod N

    Parameters
    ----------
    x : QuantumFloat or QuantumModulos or integer
        First factor of the montgomery product.
    y : QuantumFloat or QuantumModulos or integer
        Second factor of the montgomery product.
    N : int
        Original modulos of the montgomery reduction.
    m : int
        Exponent $m$ of the auxiliary radix $R=2^m$, see ``compute_aux_radix_exponent``.
    inpl_adder : Callable
        In-place quantum adder to use during computation.
        
    Returns
    ----------
    QuantumFloat
        The mongomery product of the inputs.
    """

    if isinstance(x, [int, float]) and isinstance(x, [int, float]):
        raise TypeError("Provide at least one Quantum input")
    elif isinstance(x, [int, float]):
        return cq_montgomery_multiply(x, y, N, m, inpl_adder=inpl_adder)
    elif isinstance(y, [int, float]):
        return cq_montgomery_multiply(y, x, N, m, inpl_adder=inpl_adder)
    else:
        return qq_montgomery_multiply(x, y, N, m, inpl_adder=inpl_adder)