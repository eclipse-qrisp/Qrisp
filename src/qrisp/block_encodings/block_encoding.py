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

from __future__ import annotations
import numpy as np
import jax.numpy as jnp
from qrisp.alg_primitives.state_preparation import prepare
from qrisp.core import QuantumVariable
from qrisp.core.gate_application_functions import h, x, z, gphase
from qrisp.environments import conjugate, control, invert
from qrisp.jasp.tracing_logic import QuantumVariableTemplate
from qrisp.qtypes import QuantumBool
from typing import Any, Callable
import inspect


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
    unitary : Callable
        A function ``unitary(*ancillas, *operands)`` applying the block-encoding unitary. 
        It receives the ancilla and operand QuantumVariables as arguments.
    ancillas : list[QuantumVariable | QuantumVariableTemplate]
        A list of QuantumVariables or QuantumVariableTemplates. These serve as 
        templates for the ancilla variables used in the block-encoding.
    alpha : float | int
        The scaling factor.
    is_hermitian : bool, optional
        Indicates whether the block-encoding unitary is Hermitian. The default is False.

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
        unitary: Callable[..., None],
        ancillas: list[QuantumVariable | QuantumVariableTemplate],
        alpha: float | int,
        is_hermitian: bool = False,
    ) -> None:

        self.unitary = unitary
        self.alpha = alpha
        self.is_hermitian = is_hermitian
        self.anc_templates: list[QuantumVariableTemplate] = [
            anc.template() if isinstance(anc, QuantumVariable) else anc 
            for anc in ancillas
        ]

    def create_ancillas(self) -> list[QuantumVariable]:
        r"""
        Returns a list of ancilla QuantumVariables for the BlockEncoding.

        Returns
        -------
        list[QuantumVariable]
            A list of ancilla QuantumVariables in state $\ket{0}$.
        
        """
        anc_list = []
        for template in self.anc_templates:
            anc_list.append(template.construct())
        return anc_list
    
    def apply(self, *operands: QuantumVariable) -> list[QuantumVariable]:
        r"""
        Applies the block-encoding unitary to the given operands.

        Parameters
        ----------
        *operands : QuantumVariable
            QuantumVariables serving as operands for the block-encoding.

        Returns
        -------
        list[QuantumVariable]
            A list of ancilla QuantumVariables used in the application.
            Must be measured to determine success of the block-encoding application.

        Examples
        --------

        Define a block-encoding and apply it using repeat-until-success.

        ::

            import numpy as np
            from qrisp import *
            from qrisp.operators import X, Y, Z

            H = X(0)*X(1) + 0.5*Z(0)*Z(1)
            BE = H.pauli_block_encoding()

            def operand_prep(phi):
                qv = QuantumFloat(2)
                ry(phi, qv[0])
                return qv

            @RUS
            def apply_be(BE, phi):
                qv = operand_prep(phi)

                ancillas = BE.apply(qv)

                bools = jnp.array([(measure(anc) == 0) for anc in ancillas])
                success_bool = jnp.all(bools)

                # garbage collection
                [reset(anc) for anc in ancillas]
                [anc.delete() for anc in ancillas]

                return success_bool, qv

            @terminal_sampling
            def main(BE):
                qv = apply_be(BE, np.pi / 4)
                return qv

            main(BE)
            #{3: 0.6828427278345078, 0: 0.17071065215630213, 2: 0.11715730494804945, 1: 0.02928931506114055}

        For convenience, the :meth:`apply_rus` method directly applies the block-encoding using RUS.    

        Define a block-encoding and apply it using post-selection.

        ::

            import numpy as np
            from qrisp import *
            from qrisp.operators import X, Y, Z

            H = X(0)*X(1) + 0.5*Z(0)*Z(1)
            BE = H.pauli_block_encoding()

            def operand_prep(phi):
                qv = QuantumFloat(2)
                ry(phi, qv[0])
                return qv

            def main(BE):
                operand = operand_prep(np.pi / 4)
                ancillas = BE.apply(operand)
                return operand, ancillas

            operand, ancillas = main(BE)
            res_dict = multi_measurement([operand] + ancillas)

            # Post-selection on ancillas being in |0> state
            new_dict = dict()
            success_prob = 0

            for key, prob in res_dict.items():
                if all(k == 0 for k in key[1:]):
                    new_dict[key[0]] = prob
                    success_prob += prob

            for key in new_dict.keys():
                new_dict[key] = new_dict[key] / success_prob

            new_dict
            #{3: 0.6828427278345078, 0: 0.17071065215630213, 2: 0.11715730494804945, 1: 0.02928931506114055}

        """
        ancillas = self.create_ancillas()
        self.unitary(*ancillas, *operands)
        return ancillas
    
    def apply_rus(self, operand_prep: Callable[..., Any]) -> Callable[..., Any]:
        r"""
        Applies the block-encoding unitary to the prepared operands using Repeat-Until-Success (RUS).

        Parameters
        ----------
        operand_prep : Callable
            A function ``operand_prep(*args)`` that prepares and returns the operand QuantumVariables.

        Returns
        -------
        Callable
            A function ``rus_function(*args, **kwargs)`` with the same signature 
            as ``operand_prep``. It prepares the operands and implements 
            the RUS application of the block-encoding until success is achieved.

        Examples
        --------  

        Define a block-encoding and apply it using RUS. 

        ::

            import numpy as np
            from qrisp import *
            from qrisp.operators import X, Y, Z

            H = X(0)*X(1) + 0.5*Z(0)*Z(1)
            BE = H.pauli_block_encoding()

            def operand_prep(phi):
                qv = QuantumFloat(2)
                ry(phi, qv[0])
                return qv

            @terminal_sampling
            def main(BE):
                qv = BE.apply_rus(operand_prep)(np.pi / 4)
                return qv

            main(BE)
            #{3: 0.6828427278345078, 0: 0.17071065215630213, 2: 0.11715730494804945, 1: 0.02928931506114055}
            
        """
        from qrisp.core.gate_application_functions import measure, reset
        from qrisp.jasp import RUS

        @RUS
        def rus_function(*args):
            operands = operand_prep(*args)
            if not isinstance(operands, tuple):
                operands = (operands,)    

            ancillas = self.create_ancillas()
            self.unitary(*ancillas, *operands)

            bools = jnp.array([(measure(anc) == 0) for anc in ancillas])
            success_bool = jnp.all(bools)

            # garbage collection
            [reset(anc) for anc in ancillas]
            [anc.delete() for anc in ancillas]
            return success_bool, *operands       

        return rus_function

    def __add__(self, other: BlockEncoding) -> BlockEncoding:
        r"""
        Implements addition of two BlockEncodings self and other.

        Parameters
        ----------
        other : BlockEncoding
            Another BlockEncoding to be added.  

        Returns
        -------
        BlockEncoding
            A new BlockEncoding representing the sum of self and other.

        Notes
        -----
        - Can only be used when both BlockEncodings have the same operand structure.
        - The ``+`` operator should be used sparingly, primarily to combine a few block encodings. For larger-scale polynomial transformations, Quantum Signal Processing (QSP) is the superior method.

        Examples
        --------

        Define two block-encodings and add them.

        ::

            from qrisp import *
            from qrisp.operators import X, Y, Z

            H1 = X(0)*X(1) + 0.2*Y(0)*Y(1)
            H2 = Z(0)*Z(1) + X(2)
            H3 = H1 + H2

            BE1 = H1.pauli_block_encoding()
            BE2 = H2.pauli_block_encoding()
            BE3 = H3.pauli_block_encoding()

            BE_add = BE1 + BE2

            def operand_prep():
                qv = QuantumFloat(3)
                return qv

            @terminal_sampling
            def main(BE):
                qv = BE.apply_rus(operand_prep)()
                return qv

            res_be3 = main(BE3)
            res_be_add = main(BE_add)
            print("Result from BE of H1 + H2: ", res_be3)
            print("Result from BE1 + BE2: ", res_be_add)
            # Result from BE of H1 + H2:  {0: 0.37878788804466035, 4: 0.37878788804466035, 3: 0.24242422391067933}
            # Result from BE1 + BE2:  {0: 0.37878789933341894, 4: 0.37878789933341894, 3: 0.24242420133316217}

        """
        if not isinstance(other, BlockEncoding):
            return NotImplemented
        
        alpha = self.alpha
        beta = other.alpha
        m = len(self.anc_templates)
        n = len(other.anc_templates)

        def new_unitary(*args):
            with conjugate(prepare)(args[0], np.array([np.sqrt(alpha / (alpha + beta)), np.sqrt(beta / (alpha + beta))])):
                with control(args[0], ctrl_state=0):
                    self.unitary(*args[1:1 + m], *args[1 + m + n:])

                with control(args[0], ctrl_state=1):
                    other.unitary(*args[1 + m:1 + m + n], *args[1 + m + n:])

        new_anc_templates = [QuantumBool().template()] + self.anc_templates + other.anc_templates
        new_alpha = alpha + beta
        return BlockEncoding(new_unitary, new_anc_templates, new_alpha, is_hermitian=self.is_hermitian and other.is_hermitian)

    def __sub__(self, other: BlockEncoding) -> BlockEncoding:
        r"""
        Implements subtraction of two BlockEncodings self and other.

        Parameters
        ----------
        other : BlockEncoding
            Another BlockEncoding to be subtracted.  

        Returns
        -------
        BlockEncoding
            A new BlockEncoding representing the difference of self and other.

        Notes
        -----
        - Can only be used when both BlockEncodings have the same operand structure.
        - The ``-`` operator should be used sparingly, primarily to combine a few block encodings. For larger-scale polynomial transformations, Quantum Signal Processing (QSP) is the superior method.

        Examples
        --------

        Define two block-encodings and subtract them.

        ::

            from qrisp import *
            from qrisp.operators import X, Y, Z

            H1 = X(0)*X(1) + 0.2*Y(0)*Y(1)
            H2 = Z(0)*Z(1) + X(2)
            H3 = H1 - H2

            BE1 = H1.pauli_block_encoding()
            BE2 = H2.pauli_block_encoding()
            BE3 = H3.pauli_block_encoding()

            BE_sub = BE1 - BE2

            def operand_prep():
                qv = QuantumFloat(3)
                return qv

            @terminal_sampling
            def main(BE):
                qv = BE.apply_rus(operand_prep)()
                return qv

            res_be3 = main(BE3)
            res_be_sub = main(BE_sub)
            print("Result from BE of H1 - H2: ", res_be3)
            print("Result from BE1 - BE2: ", res_be_sub)
            # Result from BE of H1 - H2:  {0: 0.37878788804466035, 4: 0.37878788804466035, 3: 0.24242422391067933}
            # Result from BE1 - BE2:  {0: 0.37878789933341894, 4: 0.37878789933341894, 3: 0.24242420133316217}

        """
        if not isinstance(other, BlockEncoding):
            return NotImplemented
        
        alpha = self.alpha
        beta = other.alpha
        m = len(self.anc_templates)
        n = len(other.anc_templates)

        def new_unitary(*args):
            with conjugate(prepare)(args[0], np.array([np.sqrt(alpha / (alpha + beta)), np.sqrt(beta / (alpha + beta))])):
                z(args[0])  # Apply Z gate to flip the sign for subtraction

                with control(args[0], ctrl_state=0):
                    self.unitary(*args[1:1 + m], *args[1 + m + n:])

                with control(args[0], ctrl_state=1):
                    other.unitary(*args[1 + m:1 + m + n], *args[1 + m + n:])

        new_anc_templates = [QuantumBool().template()] + self.anc_templates + other.anc_templates
        new_alpha = alpha + beta
        return BlockEncoding(new_unitary, new_anc_templates, new_alpha, is_hermitian=self.is_hermitian and other.is_hermitian)

    def __mul__(self, other: BlockEncoding | float | int) -> BlockEncoding:
        r"""
        Implements multiplication of a BlockEncoding by another BlockEncoding or a scalar.

        Parameters
        ----------
        other : BlockEncoding or int or float
            The object to multiply with. If a BlockEncoding is provided, the unitaries are composed. If a scalar is provided, the normalization factor $\alpha$  is scaled. 

        Returns
        -------
        BlockEncoding
            A new BlockEncoding representing the product of self and other.

        Notes
        -----
        - Scalar multiplication: Multiplying by a scalar $c$ results in a new BlockEncoding of $cA$ by updating $\alpha \rightarrow c\alpha$.
        - BlockEncoding multiplication: Can only be used when both BlockEncodings have the same operand structure.
        - The ``*`` operator should be used sparingly, primarily to combine a few block encodings. For larger-scale polynomial transformations, Quantum Signal Processing (QSP) is the superior method.
        - The product of two Hermitian operators A and B is Hermitian if and only if they commute, i.e., AB = BA.

        Examples
        --------

        **Example 1:**

        Define two block-encodings and multiply them.

        ::

            from qrisp import *
            from qrisp.operators import X, Y, Z

            # Commuting operators H1 and H2
            H1 = X(0)*X(1) + 0.2*Y(0)*Y(1)
            H2 = Z(0)*Z(1) + X(2)
            H3 = H1 * H2

            BE1 = H1.pauli_block_encoding()
            BE2 = H2.pauli_block_encoding()
            BE3 = H3.pauli_block_encoding()

            BE_mul = BE1 * BE2

            def operand_prep():
                qv = QuantumFloat(3)
                return qv

            @terminal_sampling
            def main(BE):
                qv = BE.apply_rus(operand_prep)()
                return qv

            res_be3 = main(BE3)
            res_be_mul = main(BE_mul)
            print("Result from BE of H1 * H2: ", res_be3)
            print("Result from BE1 * BE2: ", res_be_mul)
            # Result from BE of H1 * H2:  {3.0: 0.5, 7.0: 0.5}  
            # Result from BE1 * BE2:  {3.0: 0.5, 7.0: 0.5}  

        **Example 2:**

        Define two block-encodings and multiply their linear combination with a scalar.

        ::

            from qrisp import *
            from qrisp.operators import X, Y, Z

            # Commuting operators H1 and H2
            H1 = X(0)*X(1) + 0.2*Y(0)*Y(1)
            H2 = Z(0)*Z(1) + X(2)
            H3 = 2*H1 + H2

            BE1 = H1.pauli_block_encoding()
            BE2 = H2.pauli_block_encoding()
            BE3 = H3.pauli_block_encoding()

            BE_mul = 2*BE1 + BE2
            BE_mul_r = BE1*2 + BE2

            def operand_prep():
                qv = QuantumFloat(3)
                return qv

            @terminal_sampling
            def main(BE):
                qv = BE.apply_rus(operand_prep)()
                return qv

            res_be3 = main(BE3)
            res_be_mul = main(BE_mul)
            res_be_mul_r = main(BE_mul_r)

            print("Result from BE of 2 * H1 + H2: ", res_be3)
            print("Result from 2 * BE1 + BE2: ", res_be_mul)
            print("Result from BE1 * 2 + BE2: ", res_be_mul_r)
            # Result from BE of 2 * H1 + H2:  {3.0: 0.5614033770142979, 0.0: 0.21929831149285103, 4.0: 0.21929831149285103}  
            # Result from 2 * BE1 + BE2:  {3.0: 0.5614033770142979, 0.0: 0.21929831149285103, 4.0: 0.21929831149285103}
            # Result from BE1 * 2 + BE2:  {3.0: 0.5614033770142979, 0.0: 0.21929831149285103, 4.0: 0.21929831149285103}
        """
        if isinstance(other, (int, float)):
            def new_unitary(*args):
                self.unitary(*args)
                if other < 0:
                    gphase(np.pi, args[0][0])
            return BlockEncoding(self.unitary, self.anc_templates, self.alpha * abs(other))

        if isinstance(other, BlockEncoding):
            m = len(self.anc_templates)
            n = len(other.anc_templates)

            def new_unitary(*args):
                other_args = args[m:m + n] + args[m + n:]
                other.unitary(*other_args)
                self_args = args[:m] + args[m + n:]
                self.unitary(*self_args)

            new_anc_templates = self.anc_templates + other.anc_templates
            new_alpha = self.alpha * other.alpha
            return BlockEncoding(new_unitary, new_anc_templates, new_alpha)

        return NotImplemented
    
    __radd__ = __add__
    __rmul__ = __mul__

    def __matmul__(self, other: BlockEncoding) -> BlockEncoding:
        r"""
        Implements the Kronecker product of two BlockEncodings using the ``@`` operator as described in Chapter 10.2 in `Dalzell et al. <https://arxiv.org/abs/2310.03011>`_.

        Parameters
        ----------
        other : BlockEncoding
            Another BlockEncoding to be composed with self.

        Returns
        -------
        BlockEncoding
            A new BlockEncoding representing the Kronecker product of self and other.

        Notes
        -----
        - The ``@`` operator maps the operands of self to the first set of operands and the operands of other to the remaining operands in a single unified unitary.
        - The ``@`` operator should be used sparingly, primarily to combine a few block encodings. For larger-scale polynomial transformations, Quantum Signal Processing (QSP) is the superior method.
        - A more qubit-efficient implementation of the Kronecker product can be found in `this paper <https://arxiv.org/pdf/2509.15779>`_ and will be implemented in future updates.

        Examples
        --------

        **Example 1:**

        Define two block-encodings and perform their Kronecker product.

        ::

            from qrisp import *
            from qrisp.operators import X, Y, Z

            H1 = X(0)*X(1) + 0.2*Y(0)*Y(1)
            H2 = Z(0)*Z(1) + X(2)

            BE1 = H1.pauli_block_encoding()
            BE2 = H2.pauli_block_encoding()

            BE_composed = BE1 @ BE2

            n1 = H1.find_minimal_qubit_amount()
            n2 = H2.find_minimal_qubit_amount()

            def operand_prep():
                qv1 = QuantumVariable(n1)
                qv2 = QuantumVariable(n2)
                return qv1, qv2

            @terminal_sampling
            def main(BE):
                return BE.apply_rus(operand_prep)()

            result = main(BE_composed)
            print("Result from BE1 @ BE2: ", result)

        **Example 2:**

        Perform multiple Kronecker products of block-encodings in sequence.

        ::

            from qrisp import *
            from qrisp.operators import X, Y, Z

            H1 = X(0)*X(1)
            H2 = Z(0)*Z(1)
            H3 = Y(0)*Y(1)

            BE1 = H1.pauli_block_encoding()
            BE2 = H2.pauli_block_encoding()
            BE3 = H3.pauli_block_encoding()

            # Compose BE1 with the composition of BE2 and BE3
            BE_composed = BE1 @ (BE2 @ BE3)

            n1 = H1.find_minimal_qubit_amount()
            n2 = H2.find_minimal_qubit_amount()
            n3 = H3.find_minimal_qubit_amount()

            def operand_prep():
                qv1 = QuantumVariable(n1)
                qv2 = QuantumVariable(n2)
                qv3 = QuantumVariable(n3)
                return qv1, qv2, qv3

            @terminal_sampling
            def main(BE):
                return BE.apply_rus(operand_prep)()

            result = main(BE_composed)
            print("Result from BE1 @ BE2 @ BE3: ", result)

        """
        m = len(self.anc_templates)
        n = len(other.anc_templates)

        sig_self = inspect.signature(self.unitary)
        num_operand_vars_self = len(sig_self.parameters) - m
        
        def new_unitary(*args):
            self_anc = args[:m]
            other_anc = args[m : m + n]
            operands = args[m + n:]

            self.unitary(*self_anc, *operands[:num_operand_vars_self])
            other.unitary(*other_anc, *operands[num_operand_vars_self:])
        
        new_anc_templates = self.anc_templates + other.anc_templates
        new_alpha = self.alpha * other.alpha
        return BlockEncoding(new_unitary, new_anc_templates, new_alpha)

    def __neg__(self) -> BlockEncoding:
        r"""
        Implements negation of the BlockEncoding.

        Returns
        -------
        BlockEncoding
            A new BlockEncoding representing the negation of self.

        Examples
        --------

        Define a block-encoding and negate it.

        ::

            from qrisp import *
            from qrisp.operators import X, Y, Z

            H1 = X(0)*X(1) - 0.2*Y(0)*Y(1)
            H2 = 0.2*Y(0)*Y(1) - X(0)*X(1)

            BE1 = H1.pauli_block_encoding()
            BE2 = H2.pauli_block_encoding()
            BE3 = -BE1

            def operand_prep():
                qv = QuantumFloat(3)
                return qv

            @terminal_sampling
            def main(BE):
                qv = BE.apply_rus(operand_prep)()
                return qv

            res_be2 = main(BE2)
            res_be_neg = main(BE3)

            print("Result from BE of H2 = - H1: ", res_be2)
            print("Result from - BE1: ", res_be_neg)
            # Result from BE of H2 = - H1:  {3.0: 1.0}                                                  
            # Result from - BE1:  {3.0: 1.0}
        """
        def new_unitary(*args):
            self.unitary(*args)
            gphase(np.pi, args[0][0])
        return BlockEncoding(new_unitary, self.anc_templates, self.alpha, is_hermitian=self.is_hermitian)
    
    def dagger(self) -> BlockEncoding:
        r"""
        Returns the Hermitian adjoint (dagger) of the BlockEncoding.

        Returns
        -------
        BlockEncoding
            A new BlockEncoding representing the Hermitian adjoint of self.

        Examples
        --------

        Define a block-encoding and compute its dagger.

        ::

            from qrisp import *
            from qrisp.operators import X, Y, Z

            H = X(0)*Y(1) + 0.5*Z(0)*X(1)
            BE = H.pauli_block_encoding()
            BE_dg = BE.dagger()

        """

        def new_unitary(*args):
            with invert():
                self.unitary(*args)

        return BlockEncoding(new_unitary, self.anc_templates, self.alpha, is_hermitian=self.is_hermitian)
    
    def qubitization(self) -> BlockEncoding:
        r"""
        Returns the qubitization of the BlockEncoding.

        Returns
        -------
        BlockEncoding
            A new BlockEncoding representing the qubitization of self.

        Examples
        --------

        Define a block-encoding and compute its qubitization.

        ::

            from qrisp import *
            from qrisp.operators import X, Y, Z

            H = X(0)*Y(1) + 0.5*Z(0)*X(1)
            BE = H.pauli_block_encoding()
            BE_qubitized = BE.qubitization()

        """
        from qrisp.alg_primitives.reflection import reflection

        m = len(self.anc_templates)

        if self.is_hermitian:
            # W = (2*|0><0| - I) U 
            def new_unitary(*args):
                self.unitary(*args)
                reflection(args[:m])

            return BlockEncoding(new_unitary, self.anc_templates, alpha=self.alpha, is_hermitian=True)
        else:
            # W = (2*|0><0| - I) U_tilde, U_tilde = (|0><1| ⊗ U) + (|1><0| ⊗ U†) is hermitian
            def new_unitary(*args):
                with conjugate(h)(args[0]):
                    with control(args[0], ctrl_state=0):
                        self.unitary(*args[1:])

                    with control(args[0], ctrl_state=1):
                        with invert():
                            self.unitary(*args[1:])

                    x(args[0])

                reflection(args[0:1 + m])

            new_anc_templates = [QuantumBool().template()] + self.anc_templates
            return BlockEncoding(new_unitary, new_anc_templates, alpha=self.alpha, is_hermitian=True)