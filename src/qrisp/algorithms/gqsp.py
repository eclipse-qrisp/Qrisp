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
    u3,
    control,
    invert,
)
from qrisp.jasp import qache, jrange
import jax
import jax.numpy as jnp
import optax


def compute_gqsp_polynomial(p, num_iterations=1000, learning_rate=0.01):
    r"""
    Finds a vector b using Optax optimization within a JAX JIT loop.

    .. math::

        \text{argmin}_{q}\|p\star\text{reversed}(p) + q\star\text{reversed}(q) - \delta\|^2

    Args:
        c_target (np.ndarray): The target convolution result c.
        b_len_guess (int): The required length of vector b.
        num_iterations (int): Number of optimization steps.
        learning_rate (float): Optimizer learning rate.

    Returns:
        np.ndarray: The optimized vector b.
    """

    d = len(p)
    delta = jnp.zeros(2*d-1)
    delta = delta.at[d-1].set(1)
    c_target = delta - jnp.convolve(p,p[::-1]) 

    def objective_function(q, c_target):
        c_actual = jnp.convolve(q, q[::-1], mode='full')
        diff = c_actual - c_target
        return jnp.sum(diff**2)

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
        #if i % 100 == 0:
        #    print(f"Step {i}, Loss: {loss_val.item()}")

    #print(f"Final Loss: {loss_val.item()}")
    return b_params


def compute_gqsp_phase_factors(p, q):
    """
    Computes the phase factors for Generalized Quantum Signal Processing.

    Given two polynomials such that

    * $p,q\in\mathbb C[x]$, $\deg p, deg q\leq d$
    * For all $x\in\mathbb R$, $|p(e^{ix})|^2+|q(e^{ix})|^2=1$,

    this method computes the phase factors $\tetha,\phi\in\mathbb R^{d+1}$, $\lambda\in\mathbb R$.
    
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
    kappa = jnp.angle(S[1][0])

    return theta_arr, phi_arr, kappa


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
    p : numpy.ndarray
        A polynomial $p\in\mathbb C[x]$ represended as a vector of its coeffcients, 
        i.e., $p=(p_0,p_1,\dotsc,p_d)$ corresponds to $p_0+p_1x+\dotsb+p_dx^d$.
    q : numpy.ndarray, optional
        A polynomial $q\in\mathbb C[x]$ represended as a vector of its coeffcients. 
        If not specified, the polynomial is computed numerically from $p$.
    k : int, optional
        If specified, the Laurent polynomials $p'(x)=x^{-k}P(x)$, $q'(x)=x^{-k}q(x)$ are applied.
        The default is 0.

    Returns
    -------
    QuantumBool
        Auxiliary variable after applying the GQSP protocol. 
        Must be measuered in state $\ket{0}$ for the GQSP protocoll to be successful.
        
    Examples
    --------

    """

    # Convert qargs into a list
    if isinstance(qargs, (QuantumVariable, QuantumArray)):
        qargs = [qargs]

    d = len(p)

    if q == None:
        q = compute_gqsp_polynomial(p, num_iterations=5000)

    theta, phi, kappa = compute_gqsp_phase_factors(p, q)

    qbl = QuantumBool()

    u3(theta[0], phi[0], kappa, qbl)

    for i in jrange(d-k):
        with control(qbl, ctrl_state=0):
            U(*qargs)   
        u3(theta[i+1], phi[i+1], 0, qbl)

    for i in jrange(k):
        with control(qbl, ctrl_state=0):
            with invert():
                U(*qargs)
        u3(theta[d-k+i+1], phi[d-k+i+1], 0, qbl)

    return qbl
