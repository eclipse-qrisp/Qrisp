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

import numpy as np
import jax.numpy as jnp
from qrisp.alg_primitives.state_preparation import prepare
from qrisp.alg_primitives.switch_case import qswitch
from qrisp.environments import conjugate, control, invert
from qrisp.qtypes import QuantumFloat, QuantumBool
from qrisp.core.gate_application_functions import z
import warnings


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
        anc_templates,
        alpha,
        is_hermitian=False,
    ):

        self.unitary = unitary
        self.anc_templates = anc_templates
        self.alpha = alpha
        self.is_hermitian = is_hermitian


    def create_ancillas(self):
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
    
    def apply(self, *operands):
        r"""
        Applies the block-encoding unitary to the given operands.

        Parameters
        ----------
        operands : list[QuantumVariable]
            A list of QuantumVariables serving as operands for the block-encoding.

        Returns
        -------
        list[QuantumVariable]
            A list of ancilla QuantumVariables used in the application.
            Must be measured to determine success of the block-encoding application.

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
    
    def apply_rus(self, operand_prep):
        r"""
        Applies the block-encoding unitary to the prepared operands using Repeat-Until-Success (RUS).

        Parameters
        ----------
        operand_prep : callable
            A function ``operand_prep(*args)`` preparing and returning the operand QuantumVariables.

        Returns
        -------
        callable
            A function ``rus_function(*args)`` preparing the operand QuantumVariables,
            and implementing the RUS application of the block-encoding.

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

    def __add__(self, other: "BlockEncoding") -> "BlockEncoding":
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
            raise ValueError("Can only add another BlockEncoding")
        
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

    def __sub__(self, other: "BlockEncoding") -> "BlockEncoding":
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
            raise ValueError("Can only add another BlockEncoding")
        
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

    def __mul__(self, other: "BlockEncoding") -> "BlockEncoding":
        r"""
        Implements multiplication of two BlockEncodings self and other.

        Parameters
        ----------
        other : BlockEncoding
            Another BlockEncoding to be multiplied. 

        Returns
        -------
        BlockEncoding
            A new BlockEncoding representing the product of self and other.

        Notes
        -----

        - Can only be used when both BlockEncodings have the same operand structure.
        - The ``*`` operator should be used sparingly, primarily to combine a few block encodings. For larger-scale polynomial transformations, Quantum Signal Processing (QSP) is the superior method
        - The product of two Hermitian operators A and B is Hermitian if and only if they commute, i.e., AB = BA.

        Examples
        --------

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

        """
        if not isinstance(other, BlockEncoding):
            raise ValueError("Can only multiply with another BlockEncoding")

        m = len(self.anc_templates)
        n = len(other.anc_templates)

        def new_unitary(*args):
            self_args = args[:m] + args[m + n:]
            self.unitary(*self_args)
            other_args = args[m:m + n] + args[m + n:]
            other.unitary(*other_args)

        new_anc_templates = self.anc_templates + other.anc_templates
        new_alpha = self.alpha * other.alpha
        return BlockEncoding(new_unitary, new_anc_templates, new_alpha)

    __radd__ = __add__
    __rmul__ = __mul__

    def dagger(self) -> "BlockEncoding":
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
    
    def qubitization(self) -> "BlockEncoding":
        r"""
        Returns the qubitization of the BlockEncoding.

        Returns
        -------
        BlockEncoding
            A new BlockEncoding representing the qubitization of self.

        Notes
        -----
        - Qubitization can only be applied to Hermitian BlockEncodings.

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

        if not self.is_hermitian:
            warnings.warn("Qubitization can only be applied to Hermitian BlockEncodings")

        m = len(self.anc_templates)

        def new_unitary(*args):
            self.unitary(*args)
            reflection(args[:m])

        return BlockEncoding(new_unitary, self.anc_templates, alpha=self.alpha, is_hermitian=self.is_hermitian)