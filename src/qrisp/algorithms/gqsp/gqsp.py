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

#from __future__ import annotations
#from jax.typing import ArrayLike
from qrisp import (
    QuantumArray,
    QuantumVariable,
    QuantumBool,
    control,
    invert,
    rx,
    rz,
)
from qrisp.algorithms.gqsp.gqsp_angles import gqsp_angles
from qrisp.jasp import jrange
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from jax.typing import ArrayLike


# https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.020368
def GQSP(
    anc: QuantumBool, 
    *qargs: QuantumVariable, 
    unitary: Callable[..., None], 
    p: Optional["ArrayLike"] = None, 
    angles: Optional[Tuple["ArrayLike", "ArrayLike", "ArrayLike"]] = None, 
    k: int = 0, 
    kwargs: Dict[str, Any] = {}
) -> None:
    r"""
    Performs `Generalized Quantum Signal Processing <https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.020368>`_.

    Generalized Quantum Signal Processing was introduced by `Motlagh and Wiebe <https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.020368>`_. Its equivalence to
    the non-linear Fourier transform (NLFT) was subsequently established by `Laneve <https://arxiv.org/pdf/2503.03026>`_. 
    By adopting the conventions from Laneve, we treat Generalized Quantum Signal Processing (GQSP) as the overarching framework for single-qubit signal processing. 
    In this setting, previously known QSP variants—such as those constrained to $X$ or $Y$ rotations—emerge naturally as special cases where the underlying complex sequences of the NLFT are restricted to purely imaginary or real values, respectively.

    GQSP is described as follows:

    * Given a unitary $U$, 
    * and a complex degree $d$ polynomial $p(z)\in\mathbb C[z]$ such that $|p(e^{ix})|^2\leq 1$ for all $x\in\mathbb R$,

    this algorithm implements the unitary

    .. math::

        \begin{pmatrix} 
        p(U) & * \\ 
        * & *
        \end{pmatrix}
        &=R(\theta_0,\phi_0,\lambda)\left(\prod_{j=1}^{d}AR(\theta_j,\phi_j,0)\right)\\
        &=R(\theta_0,\phi_0,\lambda)AR(\theta_1,\phi_1,0)A\dotsc AR(\theta_d,\phi_d,0)

    where

    .. math::

        R(\theta,\phi,\lambda) = e^{i\lambda Z}e^{i\phi X}e^{i\theta Z} \in SU(2)

    and the angles $\theta,\phi\in\mathbb R^{d+1}$, $\lambda\in\mathbb R$ are calculated from the polynomial $p$, 
    $A=\begin{pmatrix}U & 0\\ 0 & I\end{pmatrix}$ is the signal operator.

    Parameters
    ----------
    anc : QuantumBool
        Auxiliary variable in state $\ket{0}$ for applying the GQSP protocol.
        Must be measured in state $\ket{0}$ for the GQSP protocol to be successful.
    *qargs : QuantumVariable
        QuantumVariables serving as operands for the unitary.
    unitary : Callable
        A function applying a unitary to the variables ``*qargs``.
        Typically, $U=e^{iH}$ for a Hermitian operator $H$ and GQSP applies a function of $H$.
    p : ArrayLike, optional
        1-D array containing the polynomial coefficients, ordered from lowest order term to highest.
        Either the polynomial ``p`` or ``angles`` must be specified.
    angles : tuple(ArrayLike, ArrayLike, ArrayLike), optional
        A tuple of angles $(\theta,\phi,\lambda)$ where $\theta,\phi\in\mathbb R^{d+1}$ are 1-D arrays
        and $\lambda\in\mathbb R$ is a scalar.
    k : int
        If specified, the Laurent polynomial $\tilde p(x)=x^{-k}p(x)$ is applied.
        The default is 0.
    kwargs : dict
        A dictionary of keyword arguments to pass to ``unitary``. The default is {}.

    Notes
    -----
    - The polynomial $p$ is rescaled automatically to satisfy $|p(e^{ix})|^2\leq 1$ for all $x\in\mathbb R$.
        
    Examples
    --------

    **Applying a transformation in Fourier basis**

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

    Finally, we apply QSP within a :ref:`RUS` protocol.

    ::

        @RUS
        def inner():

            p = jnp.array([0.5,0,0.5])

            operand = operand_prep()
            anc = QuantumBool()
            GQSP(anc, operand, unitary=U, p=p, k=1)

            success_bool = measure(anc) == 0
            reset(anc)
            anc.delete()
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

    .. note::

        While GQSP allows you to apply arbitrary polynomials to operators, applying
        abitrary polynomials to :ref:`BlockEncodings <BlockEncoding>` requires an 
        additional step. This is because raising the operator

        .. math::

            U = \begin{pmatrix} \frac{A}{\alpha} & * \\ * & * \end{pmatrix}

        to a given power $k$ does not necessarily give you

        .. math::

            \tilde{U} = \begin{pmatrix} \left(\frac{A}{\alpha}\right)^k & * \\ * & * \end{pmatrix}

        In order to still apply polynomials also to them, we need to call the qubitization
        method an transform the polynomial into Chebychev basis. More to that in 
        the GQSP :ref:`tutorial`.

    """

    if angles is not None:
        theta, phi, lambda_ = angles
        d = len(theta) - 1
    elif p is not None:
        d = len(p) - 1
        (theta, phi, lambda_), _ = gqsp_angles(p)

    # Define R gate application function based on Theorem 9 in https://arxiv.org/abs/2503.03026
    def R(theta, phi, qubit):
        rz(-2*theta, qubit)
        rx(-2*phi, qubit)

    theta = theta[::-1]
    phi = phi[::-1]

    for i in jrange(d-k):
        R(theta[i], phi[i], anc)
        with control(anc, ctrl_state=0):
            unitary(*qargs, **kwargs)   

    for i in jrange(k):
        R(theta[d-k+i], phi[d-k+i], anc)
        with control(anc, ctrl_state=1):
            with invert():
                unitary(*qargs, **kwargs)
        
    R(theta[d], phi[d], anc)
    rz(-2*lambda_, anc)