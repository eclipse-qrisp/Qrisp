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
import inspect
from dataclasses import dataclass
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from qrisp.core import QuantumVariable
from qrisp.core.gate_application_functions import gphase, h, ry, x, z
from qrisp.environments import conjugate, control, invert
from qrisp.jasp.tracing_logic import QuantumVariableTemplate
from qrisp.operators import QubitOperator, FermionicOperator
from qrisp.qtypes import QuantumBool
from scipy.sparse import csr_array, csr_matrix
from typing import Any, Callable, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from jax.typing import ArrayLike

MatrixType = Union[npt.NDArray[Any], csr_array, csr_matrix]


@register_pytree_node_class
@dataclass(frozen=False)
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
    alpha : ArrayLike
        The scalar scaling factor.
    ancillas : list[QuantumVariable | QuantumVariableTemplate]
        A list of QuantumVariables or QuantumVariableTemplates. These serve as 
        templates for the ancilla variables used in the block-encoding.
    unitary : Callable
        A function ``unitary(*ancillas, *operands)`` applying the block-encoding unitary. 
        It receives the ancilla and operand QuantumVariables as arguments.
    is_hermitian : bool, optional
        Indicates whether the block-encoding unitary is Hermitian. The default is False.

    Attributes
    ----------
    ancilla_templates : list[QuantumVariableTemplate]
        Templates for the ancilla variables.

    Examples
    --------

    **Example 1: Pauli Block Encoding**

    ::

        from qrisp import *
        from qrisp.operators import X, Y, Z

        H = sum(X(i)*X(i+1) + Y(i)*Y(i+1) + Z(i)*Z(i+1) for i in range(3))
        BE = H.pauli_block_encoding()

        # Apply the operator to an initial system state

        # Prepare initial system state
        def operand_prep():
            operand = QuantumFloat(4)
            h(operand[0])
            return operand

        @terminal_sampling
        def main():
            return BE.apply_rus(operand_prep)()

        main()
        # {0.0: 0.6428571295525347, 2.0: 0.2857142963579722, 1.0: 0.07142857408949305}

    **Example 2: Custom Block Encoding**

    Define a block-encoding for a discrete Laplace operator in one dimension with periodic boundary conditions.

    ::

        import numpy as np

        N = 8
        I = np.eye(N)
        A = 2*I - np.eye(N, k=1) - np.eye(N, k=-1)
        A[0, N-1] = -1
        A[N-1, 0] = -1

        print(A)
        #[[ 2. -1.  0.  0.  0.  0.  0. -1.]
        # [-1.  2. -1.  0.  0.  0.  0.  0.]
        # [ 0. -1.  2. -1.  0.  0.  0.  0.]
        # [ 0.  0. -1.  2. -1.  0.  0.  0.]
        # [ 0.  0.  0. -1.  2. -1.  0.  0.]
        # [ 0.  0.  0.  0. -1.  2. -1.  0.]
        # [ 0.  0.  0.  0.  0. -1.  2. -1.]
        # [-1.  0.  0.  0.  0.  0. -1.  2.]]

    This matrix is decomposed as linear combination of three unitaries: the identity $I$, 
    and two shift operators $V\colon\ket{k}\rightarrow-\ket{k+1\mod N}$ and $V^{\dagger}\colon\ket{k}\rightarrow-\ket{k-1\mod N}$.

    ::

        from qrisp import *
        from qrisp.block_encodings import BlockEncoding

        def I(qv):
            pass

        def V(qv):
            qv += 1
            gphase(np.pi, qv[0])

        def V_dg(qv):
            qv -= 1
            gphase(np.pi, qv[0])

        unitaries = [I, V, V_dg]

        coeffs = np.array([2.0, 1.0, 1.0, 0])
        alpha = np.sum(coeffs)

        def U(case, operand):
            with conjugate(prepare)(case, np.sqrt(coeffs/alpha)):
                qswitch(operand, case, unitaries)

        BE = BlockEncoding(alpha, [QuantumVariable(2)], U)

    Apply the operator to the inital system state $\ket{0}$.

    :: 

        # Prepare initial system state |0>
        def operand_prep():
            return QuantumFloat(3)

        @terminal_sampling
        def main():
            operand = BE.apply_rus(operand_prep)()
            return operand

        main()
        # {0.0: 0.6666666567325588, 7.0: 0.16666667908430155, 1.0: 0.1666666641831397}

    To perform quantum resource estimation (not counting repetitions), 
    replace the ``@terminal_sampling`` decorator with ``@count_ops(meas_behavior="0")``:

    ::

        @count_ops(meas_behavior="0")
        def main():
            operand = BE.apply_rus(operand_prep)()
            return operand

        main()
        # {'s': 4, 'gphase': 2, 'u3': 6, 't': 14, 't_dg': 16, 'x': 5, 'cx': 54, 'p': 2, 'h': 16, 'measure': 10}

    """

    def __init__(
        self,
        alpha: "ArrayLike",
        ancillas: list[QuantumVariable | QuantumVariableTemplate],
        unitary: Callable[..., None],
        is_hermitian: bool = False,
    ) -> None:

        self.alpha = alpha
        self.anc_templates: list[QuantumVariableTemplate] = [
            anc.template() if isinstance(anc, QuantumVariable) else anc 
            for anc in ancillas
        ]
        self.unitary = unitary
        self.is_hermitian = is_hermitian

    def tree_flatten(self):
        """
        PyTree flatten for JAX.

        Returns
        -------
        tuple
            A pair `(children, aux_data)` where `children` is a tuple containing
            the digits array, and `aux_data` is `None`.
        """
        children = (self.alpha, self.anc_templates, )
        aux_data = (self.unitary, self.is_hermitian, )
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        PyTree unflatten for JAX.

        Parameters
        ----------
        aux_data : Any
            Auxiliary data (unused, expected `None`).
        children : tuple
            Tuple containing the digits array.

        Returns
        -------
        BigInteger
            Reconstructed instance.
        """
        return cls(*children, *aux_data)
    
    @classmethod
    def from_operator(cls: "BlockEncoding", O: QubitOperator | FermionicOperator) -> BlockEncoding:
        r"""
        Creates a BlockEncoding from an operator.

        Parameters
        ----------
        O : QubitOperator | FermionicOperator
            The operator to be block-encoded.

        Returns
        -------
        BlockEncoding
            A BlockEncoding representing the Hermitian part $(O+O^{\dagger})/2$.

        Notes
        -----

        - Block encoding based on Pauli decomposition $O=\sum_i\alpha_i P_i$ where $\alpha_i$ are real coefficients and $P_i$ are Pauli strings.

        Examples
        --------

        >>> from qrisp.block_encodings import BlockEncoding
        >>> from qrisp.operators import X, Y, Z
        >>> H = X(0)*X(1) + 0.2*Y(0)*Y(1)
        >>> B = BlockEncoding.from_operator(H)
        
        """
        if isinstance(O, FermionicOperator):
            O = O.to_qubit_operator()
        return O.pauli_block_encoding()
    
    @classmethod
    def from_array(cls: "BlockEncoding", A: MatrixType) -> BlockEncoding:
        r"""
        Creates a BlockEncoding from a 2-D array.

        Parameters
        ----------
        A : ndarray | csr_array | csr_matrix
            2-D array of shape ``(N,N,)`` for a power of two ``N``.

        Returns
        -------
        BlockEncoding
            A BlockEncoding representing the Hermitian part $(A+A^{\dagger})/2$.

        Notes
        -----

        - Block encoding based on Pauli decomposition $O=\sum_i\alpha_i P_i$ where $\alpha_i$ are real coefficients and $P_i$ are Pauli strings.

        Examples
        --------

        >>> import numpy as np
        >>> from qrisp.block_encodings import BlockEncoding
        >>> A = np.array([[0,1,0,1],[1,0,0,0],[0,0,1,0],[1,0,0,0]])
        >>> B = BlockEncoding.from_array(A)
        
        """

        shape = A.shape
        # 1. Check if the array is 2D and square
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(f"Array must be square (N, N), but got {shape}.")
            
        # 2. Check if N is a power of two
        N = shape[0]
        if not (N > 0 and (N & (N - 1)) == 0):
            raise ValueError(f"Size N={N} must be a power of two.")

        return QubitOperator.from_matrix(A, reverse_endianness=True).pauli_block_encoding()

    #
    # Utilities
    #

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

        **Example 1:**

        Define a block-encoding and apply it using **repeat-until-success**.

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

        **Example 2:** 

        Define a block-encoding and apply it using **post-selection**.

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
    
    #
    # Arithmetic
    #

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

        def prep(qb, arr):
            theta = 2 * jnp.arctan(arr[1] / arr[0])
            ry(theta, qb)

        def new_unitary(*args):
            with conjugate(prep)(args[0], jnp.array([jnp.sqrt(alpha / (alpha + beta)), jnp.sqrt(beta / (alpha + beta))])):
                with control(args[0], ctrl_state=0):
                    self.unitary(*args[1:1 + m], *args[1 + m + n:])

                with control(args[0], ctrl_state=1):
                    other.unitary(*args[1 + m:1 + m + n], *args[1 + m + n:])

        new_anc_templates = [QuantumBool().template()] + self.anc_templates + other.anc_templates
        new_alpha = alpha + beta
        return BlockEncoding(new_alpha, new_anc_templates, new_unitary, is_hermitian=self.is_hermitian and other.is_hermitian)

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

        def prep(qb, arr):
            theta = 2 * jnp.arctan(arr[1] / arr[0])
            ry(theta, qb)

        def new_unitary(*args):
            with conjugate(prep)(args[0], jnp.array([jnp.sqrt(alpha / (alpha + beta)), jnp.sqrt(beta / (alpha + beta))])):
                z(args[0])  # Apply Z gate to flip the sign for subtraction

                with control(args[0], ctrl_state=0):
                    self.unitary(*args[1:1 + m], *args[1 + m + n:])

                with control(args[0], ctrl_state=1):
                    other.unitary(*args[1 + m:1 + m + n], *args[1 + m + n:])

        new_anc_templates = [QuantumBool().template()] + self.anc_templates + other.anc_templates
        new_alpha = alpha + beta
        return BlockEncoding(new_alpha, new_anc_templates, new_unitary, is_hermitian=self.is_hermitian and other.is_hermitian)

    def __mul__(self, other: "ArrayLike" | BlockEncoding) -> BlockEncoding:
        r"""
        Implements multiplication of a BlockEncoding by another BlockEncoding or a scalar.

        Parameters
        ----------
        other : ArrayLike | BlockEncoding
            The object to multiply with. If a BlockEncoding is provided, the unitaries are composed. If a scalar is provided, the normalization factor $\alpha$ is scaled. 

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
        from jax.typing import ArrayLike

        if isinstance(other, ArrayLike):
            def new_unitary(*args):
                self.unitary(*args)
                with control(other < 0):
                    gphase(np.pi, args[0][0])
            return BlockEncoding(self.alpha * jnp.abs(other), self.anc_templates, new_unitary, is_hermitian=self.is_hermitian)

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
            return BlockEncoding(new_alpha, new_anc_templates, new_unitary)

        return NotImplemented
    
    __radd__ = __add__
    __rmul__ = __mul__

    def __matmul__(self, other: BlockEncoding) -> BlockEncoding:
        r"""
        Implements the Kronecker product of two BlockEncodings.
        Implementation as described in Chapter 10.2 in `Dalzell et al. <https://arxiv.org/abs/2310.03011>`_.

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
        return BlockEncoding(new_alpha, new_anc_templates, new_unitary)

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
        return BlockEncoding(self.alpha, self.anc_templates, new_unitary, is_hermitian=self.is_hermitian)
    
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

        return BlockEncoding(self.alpha, self.anc_templates, new_unitary, is_hermitian=self.is_hermitian)
    
    #
    # Transformations
    #

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

            return BlockEncoding(self.alpha, self.anc_templates, new_unitary, is_hermitian=True)
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
            return BlockEncoding(self.alpha, new_anc_templates, new_unitary, is_hermitian=True)