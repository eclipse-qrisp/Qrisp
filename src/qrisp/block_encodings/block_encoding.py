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

from __future__ import annotations
import inspect
from dataclasses import dataclass
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from qrisp.core import QuantumVariable
from qrisp.alg_primitives.reflection import reflection
from qrisp.core.gate_application_functions import gphase, h, ry, x, z
from qrisp.environments import conjugate, control, invert
from qrisp.jasp import count_ops, depth, jrange, qache, check_for_tracing_mode as is_tracing
from qrisp.jasp.tracing_logic import QuantumVariableTemplate
from qrisp.operators import QubitOperator, FermionicOperator
from qrisp.qtypes import QuantumBool, QuantumFloat
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

    Block-encoding is a foundational technique that enables the implementation of non-unitary operations on a quantum computer by embedding them into a larger unitary operator. 
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
    num_ops : int
        The number of operand quantum variables. The default is 1.
    is_hermitian : bool
        Indicates whether the block-encoding unitary is Hermitian. The default is False.

    Attributes
    ----------
    alpha : ArrayLike
        The scalar scaling factor.
    unitary : Callable
        A function ``unitary(*ancillas, *operands)`` applying the block-encoding unitary. 
        It receives the ancilla and operand QuantumVariables as arguments.
    num_ops : int
        The number of operand quantum variables.
    num_ancs : int
        The number of ancilla quantum variables.
    is_hermitian : bool
        Indicates whether the block-encoding unitary is Hermitian.

    Notes
    -----
    - The **shape** of the block-encoded operator is determined by the size of the operand variables
      to which the block-encoding is applied. E.g., if a block-encoded $4\times 4$ matrix $A$ is applied to a 
      3-qubit QuantumVariable, then a block-encoding of the $8\times 8$ matrix $\tilde{A}=\mathbb{I}\otimes A$
      is applied. This is consistent with the convention that non-occuring indices in a Pauli string are treated as identities.
      Static-shaped block-encodings may be introduced in a future release.

    Examples
    --------

    **Example 1: Pauli Block Encoding**

    ::

        from qrisp import *
        from qrisp.block_encodings import BlockEncoding
        from qrisp.operators import X, Y, Z

        H = sum(X(i)*X(i+1) + Y(i)*Y(i+1) + Z(i)*Z(i+1) for i in range(3))
        BE = BlockEncoding.from_operator(H)

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

    **Example 2: LCU Block Encoding**

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
        coeffs = np.array([2.0, 1.0, 1.0])
        
        BE = BlockEncoding.from_lcu(coeffs, unitaries)

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
        num_ops: int = 1,
        is_hermitian: bool = False,
    ) -> None:
        self.alpha = alpha
        # Templates for the ancilla variables.
        self._anc_templates: list[QuantumVariableTemplate] = [
            anc.template() if isinstance(anc, QuantumVariable) else anc
            for anc in ancillas
        ]
        self.unitary = unitary
        self.is_hermitian = is_hermitian
        self.num_ancs = len(ancillas)
        # More robust than inferring the number of operands for the unitary via inspect.
        self.num_ops = num_ops

    def tree_flatten(self):
        """
        PyTree flatten for JAX.

        Returns
        -------
        tuple
            A pair `(children, aux_data)` where `children` is a tuple containing
            the digits array, and `aux_data` is `None`.
        """
        children = (
            self.alpha,
            self._anc_templates,
        )
        aux_data = (
            self.unitary,
            self.num_ops,
            self.is_hermitian,
        )
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
    def from_operator(
        cls: "BlockEncoding", O: QubitOperator | FermionicOperator
    ) -> BlockEncoding:
        r"""
        Constructs a BlockEncoding from an operator.

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
        - Block encoding based on Pauli decomposition $O=\sum_i\alpha_i P_i$ where $\alpha_i$ are real positive coefficients
          and $P_i$ are Pauli strings (including the respective sign).
        - **Normalization**: The block-encoding normalization factor is $\alpha = \sum_i \alpha_i$.

        Examples
        --------

        >>> from qrisp.block_encodings import BlockEncoding
        >>> from qrisp.operators import X, Y, Z
        >>> H = X(0)*X(1) + 0.2*Y(0)*Y(1)
        >>> B = BlockEncoding.from_operator(H)

        """
        if isinstance(O, FermionicOperator):
            O = O.to_qubit_operator()

        unitaries, coeffs = O.unitaries()
        return cls.from_lcu(coeffs, unitaries, is_hermitian=True)

    @classmethod
    def from_array(cls: "BlockEncoding", A: MatrixType) -> BlockEncoding:
        r"""
        Constructs a BlockEncoding from a 2-D array.

        Parameters
        ----------
        A : ndarray | csr_array | csr_matrix
            2-D array of shape ``(N,N,)`` for a power of two ``N``.

        Returns
        -------
        BlockEncoding
            A BlockEncoding representing the Hermitian part $(A+A^{\dagger})/2$.

        Raises
        ------
        ValueError 
            If ``A`` is not a 2-D square matrix.
        ValueError
            If the dimension of ``A`` is not a power of two. 

        Notes
        -----
        - Block encoding based on Pauli decomposition $O=\sum_i\alpha_i P_i$ where $\alpha_i$ are real positive coefficients
          and $P_i$ are Pauli strings (including the respective sign).
        - **Normalization**: The block-encoding normalization factor is $\alpha = \sum_i \alpha_i$.

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

        O = QubitOperator.from_matrix(A, reverse_endianness=True)
        return cls.from_operator(O)

    @classmethod
    def from_lcu(
        cls: "BlockEncoding",
        coeffs: npt.NDArray[np.number],
        unitaries: list[Callable[..., Any]],
        num_ops: int = 1,
        is_hermitian: bool = False,
    ) -> BlockEncoding:
        r"""
        Constructs a BlockEncoding using the Linear Combination of Unitaries (LCU) protocol.

        For an LCU block encoding, consider a linear combination of unitaries:

        .. math::

            O = \sum_{i=0}^{M-1} \alpha_i U_i

        where $\alpha_i$ are real non-negative coefficients such that $\sum_i \alpha_i = \alpha$,
        and $U_i$ are unitaries acting on the same operand quantum variables.

        The block encoding unitary is constructed via the LCU protocol:

        .. math::

            U = \text{PREP} \cdot \text{SEL} \cdot \text{PREP}^{\dagger}

        where:

        * **SEL** (Select, in Qrisp: :ref:`q_switch <qswitch>`) applies each unitary $U_i$ conditioned on the auxiliary variable state $\ket{i}_a$:

        .. math::

            \text{SEL} = \sum_{i=0}^{M-1} \ket{i}\bra{i} \otimes U_i

        * **PREP** (Prepare) prepares the state representing the coefficients:

        .. math::

            \text{PREP} \ket{0}_a = \sum_{i=0}^{M-1} \sqrt{\frac{\alpha_i}{\alpha}} \ket{i}_a

        Parameters
        ----------
        coeffs : ArrayLike
            1-D array of non-negative coefficients $\alpha_i$.
        unitaries : list[Callable]
            List of functions, where each ``U(*operands)`` applies a unitary
            transformation in-place to the provided quantum variables.
            All functions must accept the same signature and operate on the
            same set of operands.
        num_ops : int
            The number of operand quantum variables. The default is 1.
        is_hermitian : bool
            Indicates whether the block-encoding unitary is Hermitian. The default is False.
            Set to True, if all provided unitaries are Hermitian.

        Returns
        -------
        BlockEncoding
            A BlockEncoding using LCU.

        Raises
        ------
        ValueError
            If any entry in ``coeffs`` is negative, as the LCU protocol only supports positive coefficients.

        Notes
        -----
        - **Normalization**: The block-encoding normalization factor is $\alpha = \sum_i \alpha_i$.

        Examples
        --------

        ::

            from qrisp import *
            from qrisp.block_encodings import BlockEncoding
            def f0(x): x-=1
            def f1(x): x+=1
            BE = BlockEncoding.from_lcu(np.array([1., 1.]), [f0, f1])

            @terminal_sampling
            def main():
                return BE.apply_rus(lambda : QuantumFloat(2))()

            main()
            # {1.0: 0.5, 3.0: 0.5}

        """
        from qrisp.alg_primitives.state_preparation import prepare
        from qrisp.jasp import q_switch

        m = len(coeffs)
        n = (m - 1).bit_length()  # Number of qubits for index variable
        # Ensure coeffs has size 2 ** n by zero padding
        coeffs = np.pad(coeffs, (0, (1 << n) - m))
        alpha = np.sum(coeffs)

        if np.any(coeffs < 0):
            raise ValueError(f"Negative coefficients detected: {coeffs}. Only positive values are supported.")

        if m == 1:
            return BlockEncoding(
                alpha, [], unitaries[0], num_ops=num_ops, is_hermitian=is_hermitian
            )

        @qache
        def unitary(*args):
            # LCU = PREP SEL PREP_dg
            with conjugate(prepare)(args[0], np.sqrt(coeffs / alpha)):
                q_switch(args[0], unitaries, *args[1:])

        return BlockEncoding(
            alpha,
            [QuantumFloat(n).template()],
            unitary,
            num_ops=num_ops,
            is_hermitian=is_hermitian,
        )

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
        for template in self._anc_templates:
            anc_list.append(template.construct())
        return anc_list

    def apply(self, *operands: QuantumVariable) -> list[QuantumVariable]:
        r"""
        Applies the BlockEncoding unitary to the given operands.

        Parameters
        ----------
        *operands : QuantumVariable
            QuantumVariables serving as operands for the block-encoding.

        Returns
        -------
        list[QuantumVariable]
            A list of ancilla QuantumVariables used in the application.
            Must be measured in $0$ for success of the block-encoding application.

        Raises
        ------
        ValueError
            If the number of provided operands does not match 
            the required number of operands (self.num_ops).

        Examples
        --------

        **Example 1:**

        Define a block-encoding and apply it using :ref:`RUS`.

        ::

            import numpy as np
            from qrisp import *
            from qrisp.block_encodings import BlockEncoding
            from qrisp.operators import X, Y, Z

            H = X(0)*X(1) + 0.5*Z(0)*Z(1)
            BE = BlockEncoding.from_operator(H)

            def operand_prep(phi):
                qv = QuantumFloat(2)
                ry(phi, qv[0])
                return qv

            @RUS
            def apply_be(BE, phi):
                qv = operand_prep(phi)
                ancillas = BE.apply(qv)

                # Alternatively, also use:
                # ancillas = BE.create_ancillas()
                # BE.unitary(*ancillas, qv)

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
            from qrisp.block_encodings import BlockEncoding
            from qrisp.operators import X, Y, Z

            H = X(0)*X(1) + 0.5*Z(0)*Z(1)
            BE = BlockEncoding.from_operator(H)

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
            filtered_dict = {k[0]: p for k, p in res_dict.items() \
                            if all(x == 0 for x in k[1:])}
            success_prob = sum(filtered_dict.values())
            filtered_dict = {k: p / success_prob for k, p in filtered_dict.items()}
            filtered_dict
            #{3: 0.6828427278345078, 0: 0.17071065215630213, 2: 0.11715730494804945, 1: 0.02928931506114055}

        """

        if len(operands) != self.num_ops:
            raise ValueError(
                f"Operation expected {self.num_ops} operands, but got {len(operands)}."
            )
        
        ancillas = self.create_ancillas()
        self.unitary(*ancillas, *operands)
        return ancillas

    def apply_rus(self, operand_prep: Callable[..., Any]) -> Callable[..., Any]:
        r"""
        Applies the BlockEncoding using :ref:`RUS`.

        Parameters
        ----------
        operand_prep : Callable
            A function ``operand_prep(*args)`` that prepares and returns the operand QuantumVariables.

        Returns
        -------
        Callable
            A function ``rus_function(*args)`` with the same signature
            as ``operand_prep``. It prepares the operands and ancillas, and applies
            the block-encoding unitary within a repeat-until-success protocol.
            Returns the transformed operand QuantumVariables.

        Raises
        ------
        TypeError
            If ``operand_prep`` is not a callable object.
        ValueError
            If the number of provided operands does not match 
            the required number of operands (self.num_ops).

        Examples
        --------

        Define a block-encoding and apply it using :ref:`RUS`.

        ::

            import numpy as np
            from qrisp import *
            from qrisp.block_encodings import BlockEncoding
            from qrisp.operators import X, Y, Z

            H = X(0)*X(1) + 0.5*Z(0)*Z(1)
            BE = BlockEncoding.from_operator(H)

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

        if not callable(operand_prep):
            raise TypeError(
                f"Expected 'operand_prep' to be a callable, but got {type(operand_prep).__name__}."
            )

        @RUS
        def rus_function(*args):
            operands = operand_prep(*args)
            if not isinstance(operands, tuple):
                operands = (operands,)

            if len(operands) != self.num_ops:
                raise ValueError(
                    f"Operation expected {self.num_ops} operands, but got {len(operands)}."
                )

            ancillas = self.create_ancillas()
            self.unitary(*ancillas, *operands)

            bools = jnp.array([(measure(anc) == 0) for anc in ancillas])
            success_bool = jnp.all(bools)

            # garbage collection
            [reset(anc) for anc in ancillas]
            [anc.delete() for anc in ancillas]
            return success_bool, *operands

        return rus_function

    def resources(self, *operands: QuantumVariable, meas_behavior: str = "0"):
        r"""
        Estimate the quantum resources required for the BlockEncoding.

        This method uses the ``count_ops`` and ``depth`` decorators to obtain gate counts, circuit depth,
        and (in future release) qubit usage for a single execution of block-encoding ``.unitary``.
        Unlike :meth:`apply_rus`, it does not run the simulator
        and does not include repetitions from the :ref:`RUS` procedure.

        Parameters
        ----------
        *operands : QuantumVariable
            QuantumVariables serving as operands for the block-encoding.
        meas_behavior : str, optional
            Specifies the measurement outcome to assume during the tracing process (e.g., "0", or "1"). The default is "0".

        Returns
        -------
        dict
            A dictionary containing resource metrics with the following structure:

            - "gate counts" : A dictionary of counted quantum operations.
            - "depth": The circuit depth as an integer.

        Raises
        ------
        ValueError
            If the number of provided operands does not match 
            the required number of operands (self.num_ops).

        Examples
        --------

        **Example 1:** Estimate the quantum resources for a block-encoded Pauli operator.

        ::

            from qrisp import *
            from qrisp.block_encodings import BlockEncoding
            from qrisp.operators import X, Z

            H = X(0)*X(1) + 0.5*Z(0)*Z(1)
            BE = BlockEncoding.from_operator(H)

            res_dict = BE.resources(QuantumFloat(2))
            print(res_dict)
            # {'gate counts': {'x': 3, 'cz': 2, 'u3': 2, 'cx': 4, 'gphase': 2},
            # 'depth': 12}

        **Example 2:** Estimate the quantum resources for applying the Quantum Eigenvalue Transform.

        ::

            from qrisp import *
            from qrisp.algorithms.gqsp import QET
            from qrisp.block_encodings import BlockEncoding
            from qrisp.operators import X, Z

            H = X(0)*X(1) + 0.5*Z(0)*Z(1)
            BE = BlockEncoding.from_operator(H)

            # real, fixed parity polynomial
            p = np.array([0, 1, 0, 1])
            BE_QET = QET(BE, p)

            res_dict = BE_QET.resources(QuantumFloat(2))
            print(res_dict)
            # {'gate counts': {'x': 11, 'cz': 8, 'rx': 2, 'u3': 6, 'cx': 16,
            # 'gphase': 6, 'p': 2}, 'depth': 42}

        """

        if len(operands) != self.num_ops:
            raise ValueError(
                f"Operation expected {self.num_ops} operands, but got {len(operands)}."
            )

        ops_templates = [op.template() for op in operands]

        def operand_prep():
            operands = [temp.construct() for temp in ops_templates]
            return operands

        def main():
            operands = operand_prep()
            ancillas = self.create_ancillas()
            self.unitary(*ancillas, *operands)
            return operands

        circuit_depth = depth(meas_behavior=meas_behavior)(main)()
        gate_counts = count_ops(meas_behavior=meas_behavior)(main)()
        return {"gate counts": gate_counts, "depth": circuit_depth}

    def qubitization(self) -> BlockEncoding:
        r"""
        Returns a BlockEncoding representing the qubitization walk operator.

        For a block-encoded operator $A$ with normalization factor $\alpha$,
        this method returns a BlockEncoding of the qubitization walk operator $W$
        satisfying $W^k=T_k(A/\alpha)$ where $T_k$ is the $k$-Chebyshev polynomial of the first kind.

        The action of $W$ partitions the Hilbert space into a direct sum of two-dimensional invariant subspaces giving it the name "qubitization".
        For an eigenstate $\ket{\lambda}$ of $A$ with eigenvalue $\lambda$, the two-dimensional space is spanned by

        - $\ket{\phi_1} = \ket{0}_a\ket{\lambda}$
        - $\ket{\phi_2} = \frac{(W-\lambda/\alpha\mathbb I)\ket{\phi_1}}{\sqrt{1-(\lambda/\alpha)^2}}$

        In this subspace, $W$ implements a Pauli-Y rotaion by angle $\theta=-2\arccos(\lambda/\alpha)$, i.e., $W=e^{i\arccos(\lambda/\alpha)Y}$.

        If the block-encoding unitary $U$ is Hermitian (i.e., $U^2=\mathbb I$), then $W=R U$ where $R = (2\ket{0}_a\bra{0}_a - \mathbb I)$
        is the reflection around the state $\ket{0}_a$ of the ancilla variables.
        Otherwise, $W = R \tilde{U}$ where $\tilde{U} = (\ket{0}\bra{1} \otimes U) + (\ket{1}\bra{0} \otimes U^{\dagger})$
        is a Hermitian block-encoding of $A$ requiring one additional ancilla qubit.

        Returns
        -------
        BlockEncoding
            A new BlockEncoding instance representing the qubitization walk operator.

        Notes
        -----
        - **Normalization**: The resulting block-encoding maintains the same scaling factor $\alpha$ as the original.

        Examples
        --------

        Define a block-encoding and apply the qubitization transformation.

        ::

            from qrisp.block_encodings import BlockEncoding
            from qrisp.operators import X, Y, Z

            H = X(0)*Y(1) + 0.5*Z(0)*X(1)
            BE = BlockEncoding.from_operator(H)
            BE_walk = BE.qubitization()

        """

        m = len(self._anc_templates)
        # W = U
        if m == 0:
            return self

        if self.is_hermitian:
            # W = (2*|0><0| - I) U
            def new_unitary(*args):
                self.unitary(*args)
                reflection(args[:m])

            return BlockEncoding(
                self.alpha,
                self._anc_templates,
                new_unitary,
                num_ops=self.num_ops,
                is_hermitian=True,
            )
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

                reflection(args[0 : 1 + m])

            new_anc_templates = [QuantumBool().template()] + self._anc_templates
            return BlockEncoding(
                self.alpha,
                new_anc_templates,
                new_unitary,
                num_ops=self.num_ops,
                is_hermitian=True,
            )

    def chebyshev(self, k: int, rescale: bool = True) -> BlockEncoding:
        r"""
        Returns a BlockEncoding representing $k$-th Chebyshev polynomial of the first kind applied to the operator.

        For a block-encoded operator $A$ with normalization factor $\alpha$,
        this method returns a BlockEncoding of the rescaled operator $T_k(A)$ if ``rescale=True``,
        or $T_k(A/\alpha)$ if ``rescale=False``.

        Parameters
        ----------
        k : int
            The order of the Chebyshev polynomial. Must be a non-negative integer.
        rescale : bool, optional
            If True (default), the method returns the a block-encoding of $T_k(A)$,
            If False, the method returns a block-encoding of $T_k(A/\alpha)$.

        Returns
        -------
        BlockEncoding
            A new BlockEncoding instance representing the Chebyshev polynomial transformation.

        Notes
        -----
        - The Chebyshev polynomial approach is useful for polynomial approximations and spectral methods.
        - Should be used sparingly, primarily to combine a few block encodings. For larger-scale polynomial transformations, Quantum Signal Processing (QSP) is the superior method (see :meth:`poly`).
        - **Normalization**:
            - ``rescale=True``: The normalization factor is determined by the Quantum Eigenvalue Transform (QET).
            - ``rescale=False``: If $k=1$, the resulting block-encoding maintains the same scaling factor $\alpha$ as the original. Otherwise, the scaling factor is $1$.


        Examples
        --------

        Define a block-encoding and apply the Chebyshev polynomial transformation.

        ::

            from qrisp import *
            from qrisp.block_encodings import BlockEncoding
            from qrisp.operators import X, Y, Z

            H = X(0)*X(1) + 0.5*Z(0)*Z(1)
            BE = BlockEncoding.from_operator(H)

            # Apply Chebyshev polynomial of order 2
            BE_cheb = BE.chebyshev(2)

            def operand_prep():
                qv = QuantumFloat(2)
                return qv

            @terminal_sampling
            def main(BE):
                qv = BE.apply_rus(operand_prep)()
                return qv

            main(BE_cheb)

        """
        
        if rescale:
            from qrisp.algorithms.gqsp.qet import QET

            p = np.zeros(k + 1)
            p[-1] = 1.0
            return QET(self, p, kind="Chebyshev")

        m = len(self._anc_templates)

        iterations = k // 2

        # Following https://math.berkeley.edu/~linlin/qasc/qasc_notes.pdf (page 104):
        # T_{2k}(A) = (U_dg R U R)^k
        # T_{2k+1}(A) = (U R) (U_dg R U R)^k
        if k % 2 == 0:

            def new_unitary(*args):
                for _ in jrange(0, iterations):
                    reflection(args[:m])
                    with conjugate(self.unitary)(*args):
                        reflection(args[:m])

        else:

            def new_unitary(*args):
                for _ in jrange(0, iterations):
                    reflection(args[:m])
                    with conjugate(self.unitary)(*args):
                        reflection(args[:m])
                reflection(args[:m])
                self.unitary(*args)

        new_alpha = self.alpha if k == 1 else 1
        return BlockEncoding(
            new_alpha, self._anc_templates, new_unitary, num_ops=self.num_ops
        )

    #
    # Arithmetic
    #

    def __add__(self, other: BlockEncoding) -> BlockEncoding:
        r"""
        Returns a BlockEncoding of the sum of two operators.

        This method implements the linear combination $A + B$ via the LCU
        (Linear Combination of Unitaries) framework, where $A$ and $B$ are
        the operators encoded by the respective instances.

        Parameters
        ----------
        other : BlockEncoding
            The BlockEncoding instance to be added.

        Returns
        -------
        BlockEncoding
            A new BlockEncoding instance representing the operator sum.

        Notes
        -----
        - Can only be used when both BlockEncodings have the same operand structure.
        - The ``+`` operator should be used sparingly, primarily to combine a few block encodings. 
          For larger-scale polynomial transformations, 
          Quantum Signal Processing (QSP) is the superior method.

        Examples
        --------

        Define two block-encodings and add them.

        ::

            from qrisp import *
            from qrisp.block_encodings import BlockEncoding
            from qrisp.operators import X, Y, Z

            H1 = X(0)*X(1) + 0.2*Y(0)*Y(1)
            H2 = Z(0)*Z(1) + X(2)
            H3 = H1 + H2

            BE1 = BlockEncoding.from_operator(H1)
            BE2 = BlockEncoding.from_operator(H2)
            BE3 = BlockEncoding.from_operator(H3)

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
        m = len(self._anc_templates)
        n = len(other._anc_templates)

        def prep(qb, arr):
            theta = 2 * jnp.arctan(arr[1] / arr[0])
            ry(theta, qb)

        def new_unitary(*args):
            self_ancs = args[1 : 1 + m]
            other_ancs = args[1 + m : 1 + m + n]
            operands = args[1 + m + n :]

            with conjugate(prep)(
                args[0],
                jnp.sqrt(jnp.array([alpha, beta]) / (alpha + beta)),
            ):
                with control(args[0], ctrl_state=0):
                    self.unitary(*self_ancs, *operands)

                with control(args[0], ctrl_state=1):
                    other.unitary(*other_ancs, *operands)

        new_anc_templates = (
            [QuantumBool().template()] + self._anc_templates + other._anc_templates
        )
        new_alpha = alpha + beta
        return BlockEncoding(
            new_alpha,
            new_anc_templates,
            new_unitary,
            num_ops=self.num_ops,
            is_hermitian=self.is_hermitian and other.is_hermitian,
        )

    def __sub__(self, other: BlockEncoding) -> BlockEncoding:
        r"""
        Returns a BlockEncoding of the difference between two operators.

        This method implements the subtraction $A - B$ using a linear combination
        of unitaries (LCU), where $A$ is the operator encoded by this instance
        and $B$ is the operator encoded by 'other'.

        Parameters
        ----------
        other : BlockEncoding
            The BlockEncoding instance to be subtracted.

        Returns
        -------
        BlockEncoding
            A new BlockEncoding representing the operator difference.

        Notes
        -----
        - Can only be used when both BlockEncodings have the same operand structure.
        - The ``-`` operator should be used sparingly, primarily to combine a few block encodings.
          For larger-scale polynomial transformations,
          Quantum Signal Processing (QSP) is the superior method.

        Examples
        --------

        Define two block-encodings and subtract them.

        ::

            from qrisp import *
            from qrisp.block_encodings import BlockEncoding
            from qrisp.operators import X, Y, Z

            H1 = X(0)*X(1) + 0.2*Y(0)*Y(1)
            H2 = Z(0)*Z(1) + X(2)
            H3 = H1 - H2

            BE1 = BlockEncoding.from_operator(H1)
            BE2 = BlockEncoding.from_operator(H2)
            BE3 = BlockEncoding.from_operator(H3)

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
        m = len(self._anc_templates)
        n = len(other._anc_templates)

        def prep(qb, arr):
            theta = 2 * jnp.arctan(arr[1] / arr[0])
            ry(theta, qb)

        def new_unitary(*args):
            self_ancs = args[1 : 1 + m]
            other_ancs = args[1 + m : 1 + m + n]
            operands = args[1 + m + n :]

            with conjugate(prep)(
                args[0],
                jnp.sqrt(jnp.array([alpha, beta]) / (alpha + beta)),
            ):
                z(args[0])  # Apply Z gate to flip the sign for subtraction

                with control(args[0], ctrl_state=0):
                    self.unitary(*self_ancs, *operands)

                with control(args[0], ctrl_state=1):
                    other.unitary(*other_ancs, *operands)

        new_anc_templates = (
            [QuantumBool().template()] + self._anc_templates + other._anc_templates
        )
        new_alpha = alpha + beta
        return BlockEncoding(
            new_alpha,
            new_anc_templates,
            new_unitary,
            num_ops=self.num_ops,
            is_hermitian=self.is_hermitian and other.is_hermitian,
        )

    def __mul__(self, other: "ArrayLike") -> BlockEncoding:
        r"""
        Returns a BlockEncoding of the scaled operator.

        This method implements the scalar multiplication $c \cdot A$, where $A$
        is the operator encoded by this instance and $c$ is the
        provided scalar.

        Parameters
        ----------
        other : ArrayLike
            The scalar scaling factor (coefficient) to apply. Can be a Python float,
            a JAX/NumPy scalar, or a 0-dimensional array.

        Returns
        -------
        BlockEncoding
            A new BlockEncoding instance representing the scaled operator.

        Notes
        -----
        - Multiplying by a scalar $c$ results in a new BlockEncoding of $cA$ by updating $\alpha \rightarrow c\alpha$.

        Examples
        --------

        Define two block-encodings and implement their scaled sum as a new block encoding.

        ::

            from qrisp import *
            from qrisp.block_encodings import BlockEncoding
            from qrisp.operators import X, Y, Z

            # Commuting operators H1 and H2
            H1 = X(0)*X(1) + 0.2*Y(0)*Y(1)
            H2 = Z(0)*Z(1) + X(2)
            H3 = 2*H1 + H2

            BE1 = BlockEncoding.from_operator(H1)
            BE2 = BlockEncoding.from_operator(H2)
            BE3 = BlockEncoding.from_operator(H3)

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

            return BlockEncoding(
                self.alpha * jnp.abs(other),
                self._anc_templates,
                new_unitary,
                num_ops=self.num_ops,
                is_hermitian=self.is_hermitian,
            )

        return NotImplemented

    def __matmul__(self, other: "ArrayLike" | BlockEncoding) -> BlockEncoding:
        r"""
        Returns a BlockEncoding of the product of two operators.

        This method implements the operator product $A \cdot B$ by composing
        two BlockEncodings, where $A$ and $B$ are the operators encoded by the respective instances.

        Parameters
        ----------
        other : BlockEncoding
            The BlockEncoding instance to be multiplied.

        Returns
        -------
        BlockEncoding
            A new BlockEncoding representing the operator product.

        Notes
        -----
        - Can only be used when both BlockEncodings have the same operand structure.
        - The ``@`` operator should be used sparingly, primarily to combine a few block encodings. For larger-scale polynomial transformations, Quantum Signal Processing (QSP) is the superior method.
        - The product of two Hermitian operators A and B is Hermitian if and only if they commute, i.e., AB = BA.

        Examples
        --------

        Define two block-encodings and multiply them.

        ::

            from qrisp import *
            from qrisp.block_encodings import BlockEncoding
            from qrisp.operators import X, Y, Z

            # Commuting operators H1 and H2
            H1 = X(0)*X(1) + 0.2*Y(0)*Y(1)
            H2 = Z(0)*Z(1) + X(2)
            H3 = H1 * H2

            BE1 = BlockEncoding.from_operator(H1)
            BE2 = BlockEncoding.from_operator(H2)
            BE3 = BlockEncoding.from_operator(H3)

            BE_mul = BE1 @ BE2

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
            print("Result from BE1 @ BE2: ", res_be_mul)
            # Result from BE of H1 * H2:  {3.0: 0.5, 7.0: 0.5}
            # Result from BE1 @ BE2:  {3.0: 0.5, 7.0: 0.5}

        """
        if not isinstance(other, BlockEncoding):
            return NotImplemented

        m = len(self._anc_templates)
        n = len(other._anc_templates)

        def new_unitary(*args):
            other_args = args[m : m + n] + args[m + n :]
            other.unitary(*other_args)
            self_args = args[:m] + args[m + n :]
            self.unitary(*self_args)

        new_anc_templates = self._anc_templates + other._anc_templates
        new_alpha = self.alpha * other.alpha
        return BlockEncoding(
            new_alpha, new_anc_templates, new_unitary, num_ops=self.num_ops
        )

    __radd__ = __add__
    __rmul__ = __mul__

    def kron(self, other: BlockEncoding) -> BlockEncoding:
        r"""
        Returns a BlockEncoding of the Kronecker product (tensor product) of two operators.

        This method implements the operator $A \otimes B$, where $A$ and $B$ are
        the operators encoded by the respective instances. Following the
        construction in Chapter 10.2 in `Dalzell et al. <https://arxiv.org/abs/2310.03011>`_,
        the resulting BlockEncoding is formed by the tensor product of the underlying unitaries, $U_A \otimes U_B$.

        Parameters
        ----------
        other : BlockEncoding
            The BlockEncoding instance to be tensored.

        Returns
        -------
        BlockEncoding
            A new BlockEncoding representing the tensor product $A \otimes B$.

        Notes
        -----
        - **Normalization**: The normalization factors ($\alpha$) are combined multiplicatively.
        - The ``kron`` operator maps the operands of self to the first set of operands and the operands of other to the remaining operands in a single unified unitary.
        - The ``kron`` operator should be used sparingly, primarily to combine a few block encodings.
        - A more qubit-efficient implementation of the Kronecker product can be found in `this paper <https://arxiv.org/pdf/2509.15779>`_ and will be implemented in future updates.

        Examples
        --------

        **Example 1:**

        Define two block-encodings and perform their Kronecker product.

        ::

            from qrisp import *
            from qrisp.block_encodings import BlockEncoding
            from qrisp.operators import X, Y, Z

            H1 = X(0)*X(1) + 0.2*Y(0)*Y(1)
            H2 = Z(0)*Z(1) + X(2)

            BE1 = BlockEncoding.from_operator(H1)
            BE2 = BlockEncoding.from_operator(H2)

            BE_composed = BE1.kron(BE2)

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
            print("Result from BE1.kron(BE2): ", result)

        **Example 2:**

        Perform multiple Kronecker products of block-encodings in sequence.

        ::

            from qrisp import *
            from qrisp.operators import X, Y, Z

            H1 = X(0)*X(1)
            H2 = Z(0)*Z(1)
            H3 = Y(0)*Y(1)

            BE1 = BlockEncoding.from_operator(H1)
            BE2 = BlockEncoding.from_operator(H2)
            BE3 = BlockEncoding.from_operator(H3)

            # Compose BE1 with the composition of BE2 and BE3
            BE_composed = BE1.kron(BE2.kron(BE3))

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
            print("Result from BE1.kron(BE2.kron(BE3)): ", result)

        """
        m = len(self._anc_templates)
        n = len(other._anc_templates)

        def new_unitary(*args):
            self_ancs = args[:m]
            other_ancs = args[m : m + n]
            operands = args[m + n :]

            self.unitary(*self_ancs, *operands[: self.num_ops])
            other.unitary(*other_ancs, *operands[self.num_ops :])

        new_anc_templates = self._anc_templates + other._anc_templates
        new_alpha = self.alpha * other.alpha
        return BlockEncoding(
            new_alpha,
            new_anc_templates,
            new_unitary,
            num_ops=self.num_ops + other.num_ops,
            is_hermitian=self.is_hermitian and other.is_hermitian,
        )

    def __neg__(self) -> BlockEncoding:
        r"""
        Returns a BlockEncoding of the negated operator.

        This method implements the transformation $A \to -A$ by scaling the
        encoded operator by $-1$.

        Returns
        -------
        BlockEncoding
            A new BlockEncoding instance representing the operator $-A$.

        Examples
        --------

        Define a block-encoding and negate it.

        ::

            from qrisp import *
            from qrisp.block_encodings import BlockEncoding
            from qrisp.operators import X, Y, Z

            H1 = X(0)*X(1) - 0.2*Y(0)*Y(1)
            H2 = 0.2*Y(0)*Y(1) - X(0)*X(1)

            BE1 = BlockEncoding.from_operator(H1)
            BE2 = BlockEncoding.from_operator(H2)
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

        return BlockEncoding(
            self.alpha,
            self._anc_templates,
            new_unitary,
            num_ops=self.num_ops,
            is_hermitian=self.is_hermitian,
        )

    #
    # Transformations
    #

    def inv(self, eps: float, kappa: float) -> BlockEncoding:
        r"""
        Returns a BlockEncoding approximating the matrix inversion of the operator.

        For a block-encoded matrix $A$ with normalization factor $\alpha$, this function returns a BlockEncoding of an
        operator $\tilde{A}^{-1}$ such that $\|\tilde{A}^{-1} - A^{-1}\| \leq \epsilon$.
        The inversion is implemented via Quantum Eigenvalue Transformation (QET)
        using a polynomial approximation of $1/x$ over the domain $D_{\kappa} = [-1, -1/\kappa] \cup [1/\kappa, 1]$.

        Parameters
        ----------
        eps : float
            The target precision $\epsilon$.
        kappa : float
            An upper bound for the condition number $\kappa$ of $A$.
            This value defines the "gap" around zero where the function $1/x$ is not approximated.

        Returns
        -------
        BlockEncoding
            A new BlockEncoding instance representing an approximation of the inverse $A^{-1}$.

        Notes
        -----
        - **Complexity**: The polynomial degree scales as $\mathcal{O}(\kappa \log(\kappa/\epsilon))$.
        - It is assumed that the eigenvalues of $A/\alpha$ lie within $D_{\kappa}$.

        References
        ----------
        - Childs et. al (2017) `Quantum algorithm for systems of linear equations with exponentially improved dependence on precision <https://arxiv.org/pdf/1511.02306>`_.

        Examples
        --------

        Define a QSLP and solve it using :meth:`inv`.

        First, define a Hermitian matrix $A$ and a right-hand side vector $\vec{b}$.

        ::

            import numpy as np

            A = np.array([[0.73255474, 0.14516978, -0.14510851, -0.0391581],
                        [0.14516978, 0.68701415, -0.04929867, -0.00999921],
                        [-0.14510851, -0.04929867, 0.76587818, -0.03420339],
                        [-0.0391581, -0.00999921, -0.03420339, 0.58862043]])

            b = np.array([0, 1, 1, 1])

            kappa = np.linalg.cond(A)
            print("Condition number of A: ", kappa)
            # Condition number of A:  1.8448536035491883

        Generate a block-encoding of $A$ and use :meth:`inv` to find a block-encoding approximating $A^{-1}$.

        ::

            from qrisp import *
            from qrisp.block_encodings import BlockEncoding

            BA = BlockEncoding.from_array(A)

            BA_inv = BA.inv(0.01, 2)

            # Prepares operand variable in state |b>
            def prep_b():
                operand = QuantumVariable(2)
                prepare(operand, b)
                return operand

            @terminal_sampling
            def main():
                operand = BA_inv.apply_rus(prep_b)()
                return operand

            res_dict = main()
            amps = np.sqrt([res_dict.get(i, 0) for i in range(len(b))])

        Finally, compare the quantum simulation result with the classical solution:

        ::

            c = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)

            print("QUANTUM SIMULATION\n", amps, "\nCLASSICAL SOLUTION\n", c)
            # QUANTUM SIMULATION
            # [0.02844496 0.55538449 0.53010186 0.64010231]
            # CLASSICAL SOLUTION
            # [0.02944539 0.55423278 0.53013239 0.64102936]

        """
        from qrisp.algorithms.gqsp import inversion

        return inversion(self, eps, kappa)

    def sim(self, t: "ArrayLike" = 1, N: int = 1) -> BlockEncoding:
        r"""
        Returns a BlockEncoding approximating Hamiltonian simulation of the operator.

        For a block-encoded Hamiltonian $H$, this method returns a BlockEncoding of an approximation of
        the unitary evolution operator $e^{-itH}$ for a given time $t$.

        The approximation is based on the Jacobi-Anger expansion into Bessel functions 
        of the first kind ($J_n$):

        .. math ::

            e^{-it\cos(\theta)} \approx \sum_{n=-N}^{N}(-i)^nJ_n(t)e^{in\theta}

        Parameters
        ----------
        t : ArrayLike
            The scalar evolution time $t$. The default is 1.0.
        N : int
            The truncation order $N$ of the expansion. A higher order provides 
            better approximation for larger $t$ or higher precision requirements. 
            Default is 1.

        Returns
        -------
        BlockEncoding
            A new BlockEncoding instance representing an approximation of the unitary $e^{-itH}$.

        Notes
        -----
        - **Precision**: The truncation error scales (decreases) super-exponentially with $N$. 
          For a fixed $t$, choosing $N > |t|$ ensures rapid convergence.
        - **Normalization**: The resulting operator is nearly unitary, meaning its 
          block-encoding normalization factor $\alpha$ will be close to 1.

        References
        ----------
        - Motlagh & Wiebe (2025) `Generalized Quantum Signal Processing <https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.020368>`_.

        Examples
        --------

        Generate an Ising Hamiltonian $H$ and apply Hamiltonian simulation $e^{-itH}$ to the inital system state $\ket{0}$.

        ::

            # For larger systems, restart the kernel and adjust simulator precision
            # import os
            # os.environ["QRISP_SIMULATOR_FLOAT_THRESH"] = "1e-10"

            from qrisp import *
            from qrisp.block_encodings import BlockEncoding
            from qrisp.operators import X, Y, Z

            def create_ising_hamiltonian(L, J, B):
                H = sum(-J * Z(i) * Z(i + 1) for i in range(L-1))  \
                    + sum(B * X(i) for i in range(L))
                return H

            L = 4
            H = create_ising_hamiltonian(L, 0.25, 0.5)
            BE = BlockEncoding.from_operator(H)

            # Prepare inital system state |psi> = |0>
            def operand_prep():
                return QuantumFloat(L)

            # Prepare state|psi(t)> = e^{itH} |psi>
            def psi(t):
                BE_sim = BE.sim(t=t, N=8)
                operand = BE_sim.apply_rus(operand_prep)()
                return operand

            @terminal_sampling
            def main(t):
                return psi(t)

            res_dict = main(0.5)
            amps = np.sqrt([res_dict.get(i, 0) for i in range(2 ** L)])
            print(amps)
            #[0.88288218 0.224682   0.22269639 0.05723058 0.22269632 0.05669449                   
            # 0.0570588  0.01457775 0.22468192 0.05717859 0.05669445 0.0145699
            # 0.05723059 0.01456992 0.01457775 0.00372438]

        Finally, compare the quantum simulation result with the classical solution:

        ::

            import scipy as sp

            H_mat = H.to_array()

            # Prepare state|psi(t)> = e^{itH} |psi>
            def psi_(t):
                # Prepare inital system state |psi> = |0>
                psi0 = np.zeros(2**H.find_minimal_qubit_amount())
                psi0[0] = 1
                
                psi = sp.linalg.expm(-1.0j * t * H_mat) @ psi0
                psi = psi / np.linalg.norm(psi)
                return psi

            c = np.abs(psi_(0.5))
            print(c)
            #[0.88288217 0.22468197 0.22269638 0.05723056 0.22269638 0.05669446
            # 0.05705877 0.01457772 0.22468197 0.0571786  0.05669446 0.01456988
            # 0.05723056 0.01456988 0.01457772 0.00372439]

        """
        from qrisp.algorithms.gqsp import hamiltonian_simulation

        return hamiltonian_simulation(self, t, N)

    def poly(
        self, p: "ArrayLike", kind: Literal["Polynomial", "Chebyshev"] = "Polynomial"
    ) -> BlockEncoding:
        r"""
        Returns a BlockEncoding representing a polynomial transformation of the operator.

        For a block-encoded matrix $A$ and a (complex) polynomial $p(z)$, this method returns
        a BlockEncoding of the operator $p(A)$. This is achieved using
        Generalized Quantum Eigenvalue Transformation (GQET).

        Parameters
        ----------
        p : ArrayLike
            1-D array containing the polynomial coefficients, ordered from lowest order term to highest.
        kind : {"Polynomial", "Chebyshev"}
            The basis in which the coefficients are defined.

            - ``"Polynomial"``: $p(x) = \sum c_i x^i$

            - ``"Chebyshev"``: $p(x) = \sum c_i T_i(x)$, where $T_i$ are Chebyshev polynomials of the first kind.

            Default is ``"Polynomial"``.

        Returns
        -------
        BlockEncoding
            A new Block-Encoding instance representing the transformed operator $p(A)$.

        Examples
        --------

        Define a Hermitian matrix $A$ and a vector $\vec{b}$.

        ::

            import numpy as np

            A = np.array([[0.73255474, 0.14516978, -0.14510851, -0.0391581],
                        [0.14516978, 0.68701415, -0.04929867, -0.00999921],
                        [-0.14510851, -0.04929867, 0.76587818, -0.03420339],
                        [-0.0391581, -0.00999921, -0.03420339, 0.58862043]])

            b = np.array([0, 1, 1, 1])

        Generate a block-encoding $A$ of and use :meth:`poly` to find a block-encoding of $p(A)$.

        ::

            from qrisp import *
            from qrisp.block_encodings import BlockEncoding

            BA = BlockEncoding.from_array(A)

            BA_poly = BA.poly(np.array([1.,2.,1.]))

            # Prepares operand variable in state |b>
            def prep_b():
                operand = QuantumVariable(2)
                prepare(operand, b)
                return operand

            @terminal_sampling
            def main():
                operand = BA_poly.apply_rus(prep_b)()
                return operand

            res_dict = main()
            amps = np.sqrt([res_dict.get(i, 0) for i in range(len(b))])

        Finally, compare the quantum simulation result with the classical solution:

        ::

            c = (np.eye(4) + 2 * A + A @ A) @ b
            c = c / np.linalg.norm(c)

            print("QUANTUM SIMULATION\n", amps, "\nCLASSICAL SOLUTION\n", c)
            # QUANTUM SIMULATION
            # [0.02986315 0.57992481 0.62416743 0.52269535]
            # CLASSICAL SOLUTION
            # [-0.02986321  0.57992485  0.6241675   0.52269522]

        """
        from qrisp.algorithms.gqsp import GQET

        if isinstance(p, list):
            p = np.array(p)
        return GQET(self, p, kind=kind)
