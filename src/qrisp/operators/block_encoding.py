"""
********************************************************************************
* Copyright (c) 2024 the Qrisp authors
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


class BlockEncoding:
    r"""
    Central structure for representing block-encodings.

    Block-encoding is a foundational technique that enables the implementation of non-unitary operations on a quantum computer by embedding them into a larger unitary framework. 
    Given an operator $A$, we embed a scaled version $A/\alpha$ into the upper-left block of a unitary matrix $U_{A}$:

    .. math::

        U_A= 
        \begin{pmatrix}
            A/\alpha & *\\
            * & * 
        \end{pmatrix}

    More formally, a block-encoding of an operator $A$ (not necessarily unitary) acting on a Hilbert space $\mathcal H_{s}$ 
    is a unitary acting on $\mathcal H_a\otimes H_s$ (for some auxiliary Hilbert space $\mathcal H_a$) such that
    
    .. math::

        \|A - \alpha (\bra{0}_a \otimes \mathbb I_s) U_A (\ket{0}_a \otimes \mathbb I_s) \| \leq \epsilon

    where 
    
    - $\alpha\geq \|A\|$ is a subnormalization factor (or scaling factor) that ensures $A/\alpha$ has singular values within the unit disk.
    - $\epsilon\geq 0$ represents the approximation error.

    The block-encoding is termed exact if $\epsilon=0$, meaning the upper-left block of $U_{A}$ is exactly $A/\alpha$.
    
    **Implementation mechanism**

    To apply the operator $A$ to a quantum state $\ket{\psi}$:

    - Prepare the system in state $\ket{0}_a \otimes \ket{\psi}_s$.
    - Apply the unitary $U_A$.
    - Post-select by measuring the ancillas. If the result is $\ket{0}_a$, the remaining state is $\dfrac{A\ket{\psi}}{\|A\ket{\psi}\|}$.
    - The success probability of this operation is given by 

    .. math::

        P_{\text{success}} = \dfrac{\|A\ket{\psi}\|^2}{\alpha^2}

    Parameters
    ----------
    unitary : callable
        A function ``unitary(*ancillas, *operands)`` applying the block-encoding unitary 
        to ancilla and operand QuantumVariables.
    ancillas : list[QuantumVariable]
        A list of QuantumVariables serving as templates for the ancilla variables.
    alpha : float
        The scaling factor.
    is_hermitian : bool
        Indicates whether the block-encoding unitary is Hermitian. The default is ``False``.

    Attributes
    ----------
    ancilla_templates : list[QuantumVariableTemplate]
        Templates for the ancilla variables.

    Examples
    --------

    We define a block-encoding for a symmetric tridiagonal matrix with wrap-around corners.

    ::

        import numpy as np

        N = 8
        I = np.eye(N)
        A = I + np.eye(N, k=1) + np.eye(N, k=-1)
        A[0, N-1] = 1
        A[N-1, 0] = 1

        print(A)
        #[[1. 1. 0. 0. 0. 0. 0. 1.]
        #[1. 1. 1. 0. 0. 0. 0. 0.]
        #[0. 1. 1. 1. 0. 0. 0. 0.]
        #[0. 0. 1. 1. 1. 0. 0. 0.]
        #[0. 0. 0. 1. 1. 1. 0. 0.]
        #[0. 0. 0. 0. 1. 1. 1. 0.]
        #[0. 0. 0. 0. 0. 1. 1. 1.]
        #[1. 0. 0. 0. 0. 0. 1. 1.]]

    This matrix is decomposed as linear combination of three unitaries: the identity $I$, 
    and two shift operators $V\colon\ket{k}\rightarrow\ket{k+1\mod N}$ and $V^{\dagger}\colon\ket{k}\rightarrow\ket{k-1\mod N}$.

    ::

        from qrisp import *
        from qrisp.operators import BlockEncoding

        def I(qv):
            pass

        def V(qv):
            qv += 1

        def V_dg(qv):
            qv -= 1

        unitaries = [I, V, V_dg]

        coeffs = np.array([1.0, 1.0, 1.0, 0])
        alpha = np.sum(coeffs)

        def U(case, operand):
            with conjugate(prepare)(case, np.sqrt(coeffs/alpha)):
                qswitch(operand, case, unitaries)

        block_encoding = BlockEncoding(U, [QuantumVariable(2)], alpha)

    :: 

        @RUS
        def test():

            operand = QuantumFloat(4)
            ancillas = block_encoding.create_ancillas()

            block_encoding.unitary(*ancillas, operand)

            bools = jnp.array([(measure(anc) == 0) for anc in ancillas])
            success_bool = jnp.all(bools)

            # garbage collection
            [reset(anc) for anc in ancillas]
            [anc.delete() for anc in ancillas]

            return success_bool, operand


        @terminal_sampling
        def main():

            qv = test()
            return qv

        main()
        # {15.0: 0.3333334525426193, 0.0: 0.3333333035310118, 1.0: 0.3333332439263688}

    """

    def __init__(
        self,
        unitary,
        ancillas,
        alpha,
        is_hermitian=False,
    ):

        self.unitary = unitary
        self.ancilla_templates = [anc.template() for anc in ancillas]
        self.alpha = alpha
        self.is_hemitian = is_hermitian


    def create_ancillas(self):
        """
        Returns a list of ancilla QuantumVariables for the BlockEncoding.

        Returns
        -------
        list[QuantumVariable]
            A list of ancilla QuantumVariables in state $\ket{0}$.
        
        """
        anc_list = []
        for template in self.ancilla_templates:
            anc_list.append(template.construct())
        return anc_list
  
