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

import numpy as np
from qrisp import (
    QuantumArray,
    QuantumVariable,
    QuantumBool,
    h,
    u3,
    z,
    control,
    invert,
    gphase,
)
from qrisp.jasp import qache, jrange
import jax
import jax.numpy as jnp
import optax


def compute_gqsp_polynomial(p, num_iterations=10000, learning_rate=0.01):
    r"""
    Find the second GQSP polynomial $q$ using Optax optimization within a JAX JIT loop.

    This function solves the optimization problem

    .. math::

        \text{argmin}_{q}\|p\star\text{reversed}(p)^* + q\star\text{reversed}(q)^* - \delta\|^2

    where $\delta=(0,\dotsc,0,1,0,\dotsc,0)$ is a vector of length $2d+1$ with $1$ at the center and $d$ zeros on each side, 
    and $\star$ is the convolution operator.

    The polynomial $p$ must satisfy $|p(e^{ix})|^2\leq 1$ for all $x\in\mathbb R$.

    This function is JAX-traceable.

    Parameters
    ----------
    p : ndarray
        A polynomial $p\in\mathbb C[x]$ represended as a vector of its coefficients, 
        i.e., $p=(p_0,p_1,\dotsc,p_d)$ corresponds to $p_0+p_1x+\dotsb+p_dx^d$.
    num_iterations : int
        Number of optimization steps.
    learning_rate : float
        Optimizer learning rate.

    Returns
    -------
        ndarray
            The optimized vector $q$.
    """

    d = len(p)
    delta = jnp.zeros(2*d-1)
    delta = delta.at[d-1].set(1)
    c_target = delta - jnp.convolve(p, jnp.conjugate(p[::-1]), mode='full') 

    def objective_function(q, c_target):
        c_actual = jnp.convolve(q, jnp.conjugate(q[::-1]), mode='full')
        diff = c_actual - c_target
        return jnp.linalg.norm(diff)**2

    # JIT-compile the value and gradient calculation
    loss_and_grad = jax.jit(jax.value_and_grad(objective_function))

    # Initialize optimizer (e.g., Adam or Gradient Descent)
    optimizer = optax.adam(learning_rate)
    
    # Initialize parameters (b) and optimizer state
    key = jax.random.PRNGKey(0)
    # Start with a random guess
    b_params = jax.random.normal(key, shape=(d,)) 
    opt_state = optimizer.init(b_params)

    # Define a single optimization step function
    @jax.jit
    def update_step(params, opt_state, target_c_static):
        loss_val, grads = loss_and_grad(params, target_c_static)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    # Run the optimization loop (this loop itself is Python, but the step inside is JIT)
    for i in range(num_iterations):
        b_params, opt_state, loss_val = update_step(b_params, opt_state, c_target)

    return b_params


def compute_gqsp_phase_factors(p, q):
    r"""
    Computes the phase factors for GQSP.

    Given two polynomials such that

    * $p,q\in\mathbb C[x]$, $\deg p, \deg q \leq d$,
    * for all $x\in\mathbb R$, $|p(e^{ix})|^2+|q(e^{ix})|^2=1$,

    this method computes the phase factors $\theta,\phi\in\mathbb R^{d+1}$, $\lambda\in\mathbb R$.

    This function is JAX-traceable.

    Parameters
    ----------
    p : ndarray
        A polynomial $p\in\mathbb C[x]$ represented as a vector of its coefficients, 
        i.e., $p=(p_0,p_1,\dotsc,p_d)$ corresponds to $p_0+p_1x+\dotsb+p_dx^d$.
    q : ndarray
        A polynomial $q\in\mathbb C[x]$ represented as a vector of its coefficients. 

    Returns
    -------
    theta_arr : ndarray
        The phase factors $(\theta_0,\dotsc,\theta_d)$.
    phi_arr : ndarray
        The phase factors $(\phi_0,\dotsc,\phi_d)$.
    lambda : float
        The phase factor $\lambda$.

    """

    d = len(p) - 1
    theta_arr = jnp.zeros(d + 1)
    phi_arr = jnp.zeros(d + 1)

    S = jnp.vstack([p, q])

    # Add a small perturbation to denominators to avoid division by zero when computing phase factors
    eps = 1e-10
    theta_arr = theta_arr.at[d].set(jnp.arctan(jnp.abs(S[1][d] / (S[0][d] + eps))))
    phi_arr = phi_arr.at[d].set(jnp.angle(S[0][d] / (S[1][d] + eps)))

    def cond_fun(vals):
        d, S, theta_arr, phi_arr = vals
        return d > 0

    def body_fun(vals):
        d, S, theta_arr, phi_arr = vals
    
        theta = theta_arr[d]
        phi = phi_arr[d]
        # R(theta, phi, 0)^dagger
        R = jnp.array([[jnp.exp(-phi*1j) * jnp.cos(theta), jnp.sin(theta)],[jnp.exp(-phi*1j) * jnp.sin(theta), -jnp.cos(theta)]])
        S = R @ S
        S = jnp.vstack([S[0][1:d+1],S[1][0:d]])
        
        d = d-1
        theta_arr = theta_arr.at[d].set(jnp.arctan(jnp.abs(S[1][d] / (S[0][d] + eps))))
        phi_arr = phi_arr.at[d].set(jnp.angle(S[0][d] / (S[1][d] + eps)))

        return d, S, theta_arr, phi_arr
    
    #d, S, theta_arr, phi_arr = jax.lax.while_loop(cond_fun, body_fun, (d, S, theta_arr, phi_arr))
    vals = (d, S, theta_arr, phi_arr)
    while(cond_fun(vals)):
        vals = body_fun(vals)

    d, S, theta_arr, phi_arr = vals
    lambda_ = jnp.angle(S[1][0])

    return theta_arr, phi_arr, lambda_


# https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.020368
def GQSP(qargs, U, p, q=None, k=0):
    r"""
    Performs `Generalized Quantum Signal Processing <https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.020368>`_.

    Given two polynomials such that

    * $p,q\in\mathbb C[x]$, $\deg p, \deg q\leq d$
    * For all $x\in\mathbb R$, $|p(e^{ix})|^2+|q(e^{ix})|^2=1$,

    this method implements the unitary

    .. math::

        \begin{pmatrix} 
        p(U) & * \\ 
        q(U) & *
        \end{pmatrix}
        =\left(\prod_{j=1}^dR(\theta_j,\phi_j,0)A\right)R(\theta_0,\phi_0,\lambda)

    where the angles $\theta,\phi\in\mathbb R^{d+1}$, $\lambda\in\mathbb R$ are calculated from the polynomials $p,q$, 
    $A=\begin{pmatrix}I & 0\\ 0 & U\end{pmatrix}$ is the signal operator.
    If $q$ is not specified, it is computed numerically form $p$. The polynomial $p$ must satisfy $|p(e^{ix})|^2\leq 1$ for all $x\in\mathbb R$.

    Parameters
    ----------
    qargs : QuantumVariable | QuantumArray | list[QuantumVariable | QuantumArray]
        The (list of) QuantumVariables representing the state to apply the GQSP on.
    U : function
        A function appying a unitary to the variables in ``qargs``.
        Typically, $U=e^{iH}$ for a Hermitian operator $H$ and GQSP applies a function of $H$.
    p : ndarray
        A polynomial $p\in\mathbb C[x]$ represented as a vector of its coefficients, 
        i.e., $p=(p_0,p_1,\dotsc,p_d)$ corresponds to $p_0+p_1x+\dotsb+p_dx^d$.
    q : ndarray, optional
        A polynomial $q\in\mathbb C[x]$ represented as a vector of its coefficients. 
        If not specified, the polynomial is computed numerically from $p$.
    k : int, optional
        If specified, the Laurent polynomials $p'(x)=x^{-k}P(x)$, $q'(x)=x^{-k}q(x)$ are applied.
        The default is 0.

    Returns
    -------
    QuantumBool
        Auxiliary variable after applying the GQSP protocol. 
        Must be measuered in state $\ket{0}$ for the GQSP protocol to be successful.
        
    Examples
    --------

    **Example 1: Applying a transformation in Fourier basis**

    We apply the operator

    .. math::

        \cos(H) = \frac{e^{iH}+e^{-iH}}{2}

    for some :ref:`Hermitian operator <operators>` $H$ to the input state $\ket{\psi}=\ket{0}$.

    First, we define an operator $H$ and the unitary performing the Hamiltonian evolution $e^{iH}$.
    (In this case, Trotterization will perform Hamiltonian evolution exactly since the individual terms commute.)

    ::

        from qrisp import *
        from qrisp.gqsp import *
        from qrisp.operators import X,Y,Z
        import jax.numpy as jnp

        H = Z(0)*Z(1) + X(0)*X(1)

        def U(operand):
            H.trotterization(forward_evolution=False)(operand)


    Next, we define the ``operand_prep`` function that prepares a QuantumVariable is state $\ket{\psi}=\ket{0}$.

    ::

        def operand_prep():
            operand = QuantumVariable(2)
            return operand

    The transformation $\cos(H)$ is achieved by applying $p'(x)=0.5x^{-1} + 0.5x^1$ to the unitary $e^{iH}$.
    This corresponds to the polynomial $p(x)=0.5+0.5x^2$ (i.e., ``p=[0.5,0,0.5]``) and ``k=1``. 
    A suitable second polynomial is $q(x)=-0.5+0.5x^2$ (i.e., ``q=[-0.5,0,0.5]``) which corresponds to $q'(x)=-0.5x^{-1}+0.5x$.

    Finally, we apply QSP within a :ref:`RUS` protocol.

    ::

        @RUS
        def inner():

            p = jnp.array([0.5,0,0.5])
            q = jnp.array([-0.5,0,0.5])

            operand = operand_prep()
            qbl = GQSP(operand, U, p, q, k=1)

            success_bool = measure(qbl) == 0
            return success_bool, operand


        @terminal_sampling
        def main(): 

            qv = inner()
            return qv

    and simulate

    >>> main()
    {3: 0.85471756539818, 0: 0.14528243460182003}

    Let's compare to the classically calculated result:

    >>> A = H.to_array()
    >>> from scipy.linalg import cosm
    >>> print(cosm(A))
    [[ 0.29192658+0.j  0.        +0.j  0.        +0.j -0.70807342+0.j]
    [ 0.        +0.j  0.29192658+0.j  0.70807342+0.j  0.        +0.j]
    [ 0.        +0.j  0.70807342+0.j  0.29192658+0.j  0.        +0.j]
    [-0.70807342+0.j  0.        +0.j  0.        +0.j  0.29192658+0.j]]

    That is, starting in state $\ket{\psi}=\ket{0}=(1,0,0,0)$, we obtain

    >>> result = cosm(A)@(np.array([1,0,0,0]).transpose())
    >>> result = result/np.linalg.norm(result) # normalise
    >>> result = result**2 # compute measurement probabilities
    >>> print(result)
    [0.1452825+0.j 0.       +0.j 0.       +0.j 0.8547175-0.j]

    which are exactly the probabilities we observed in the quantum simulation.


    **Example 2: Applying a transformation in Chebyshev basis**

    An example for filtered state preparation with GQSP is shown in the :ref:`tutorial`.

    """

    # Convert qargs into a list
    if isinstance(qargs, (QuantumVariable, QuantumArray)):
        qargs = [qargs]

    d = len(p) - 1

    if q == None:
        q = compute_gqsp_polynomial(p, num_iterations=10000)

    theta, phi, lambda_ = compute_gqsp_phase_factors(p, q)

    qbl = QuantumBool()

    # Define R gate application function based on formula (4) in https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.020368
    def R(theta, phi, kappa, qubit):
        z(qubit)
        u3(2 * theta, -phi, -kappa, qubit)
        gphase(phi + kappa, qubit)


    R(theta[0], phi[0], lambda_, qbl)

    for i in jrange(d-k):
        with control(qbl, ctrl_state=0):
            U(*qargs)   
        R(theta[i+1], phi[i+1], 0, qbl)

    for i in jrange(k):
        with control(qbl, ctrl_state=1):
            with invert():
                U(*qargs)
        R(theta[d-k+i+1], phi[d-k+i+1], 0, qbl)

    return qbl


@jax.jit
def polynomial_to_chebyshev(coeffs):
    """
    Converts polynomial coefficients in the power basis to coefficients 
    in the Chebyshev basis of the first kind $(T_n(x))$.

    This function is JAX-traceable.
    
    Parameters
    ----------
    coeffs : ndarray
        A polynomial $p\in\mathbb C[x]$ represented as a vector of its coefficients, 
        i.e., $p=(p_0,p_1,\dotsc,p_d)$ corresponds to $p_0+p_1x+\dotsb+p_dx^d$.

    Returns
    -------
    ndarray
        A polynomial $p\in\mathbb C[x]$ represented as a vector of its coefficients in Chebyshev basis, 
        i.e., $(t_0,t_1,\dotsc,t_d)$ corresponds to $t_0T_0(x)+t_1T_1(x)+\dotsb+t_dT_d(x)$
        where $T_n(x)$ is the $n$-th Chebyshev polynomial of first kind.

    """
    N = len(coeffs)
    
    # Build the transformation matrix C such that P_power = C @ P_cheb
    # This matrix contains the power-basis coefficients of T_n(x)
    C = jnp.zeros((N, N), dtype=coeffs.dtype)
    C = C.at[0, 0].set(1)
    if N > 1:
        C = C.at[1, 1].set(1)
        for n in range(2, N):
            prev = C[n-1]
            prev_shifted = jnp.roll(prev, 1) * 2
            # Handle the roll boundary condition manually to match 2*x*T_{n-1}
            prev_shifted = prev_shifted.at[0].set(0) 
            C = C.at[n, :].set(prev_shifted - C[n-2, :])
            
    # Solve the linear system for the Chebyshev coefficients
    # The matrix C is triangular/well-behaved, making the solve stable
    cheb_coeffs = jnp.linalg.solve(C.T, coeffs)
    
    return cheb_coeffs