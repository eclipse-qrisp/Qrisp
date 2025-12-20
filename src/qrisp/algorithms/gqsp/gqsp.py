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

from qrisp import (
    QuantumArray,
    QuantumVariable,
    QuantumBool,
    u3,
    z,
    control,
    invert,
    gphase,
)
from qrisp.algorithms.gqsp.gqsp_angles import _gqsp_angles
from qrisp.jasp import jrange


# https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.020368
def GQSP(qargs, U, p, q=None, angles=None, k=0):
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
    p : ndarray, optional
        A polynomial $p\in\mathbb C[x]$ represented as a vector of its coefficients, 
        i.e., $p=(p_0,p_1,\dotsc,p_d)$ corresponds to $p_0+p_1x+\dotsb+p_dx^d$.
        Either the polynomial ``p`` or ``angles`` must be specified.
    q : ndarray, optional
        A polynomial $q\in\mathbb C[x]$ represented as a vector of its coefficients. 
        If not specified, the polynomial is computed numerically from $p$, and $p$ is rescaled to ensure $|p(e^{ix})|\leq 1$ for all $x\in\mathbb R$.
    angles : tuple(ndarray, ndarray, float), optional
        A tuple of angles $(\theta,\phi,\lambda)$ for $\theta,\phi\in\mathbb R^{d+1}$, $\lambda\in\mathbb R$.
    k : int, optional
        If specified, the Laurent polynomials $\tilde p(x)=x^{-k}p(x)$, $\tilde q(x)=x^{-k}q(x)$ are applied.
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

    The transformation $\cos(H)$ is achieved by applying $\tilde p(x)=0.5x^{-1} + 0.5x^1$ to the unitary $e^{iH}$.
    This corresponds to the polynomial $p(x)=0.5+0.5x^2$ (i.e., ``p=[0.5,0,0.5]``) and ``k=1``. 
    A suitable second polynomial is $q(x)=-0.5+0.5x^2$ (i.e., ``q=[-0.5,0,0.5]``) which corresponds to $\tilde q(x)=-0.5x^{-1}+0.5x$.

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

    if angles != None:
        theta, phi, lambda_ = angles
        d = len(theta) - 1
    else:
        d = len(p) - 1
        theta, phi, lambda_ = _gqsp_angles(p)

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