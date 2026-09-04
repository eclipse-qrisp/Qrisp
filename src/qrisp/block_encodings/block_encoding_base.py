# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Defines the BlockEncoding dataclass for representing, applying, and combining block-encoded operators."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import NotImplementedType
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike

from qrisp.alg_primitives.reflection import reflection
from qrisp.alg_primitives.state_preparation import prepare
from qrisp.block_encodings.ancilla_layout import _AncillaLayout, _maximum_layout_size
from qrisp.core import QuantumVariable, mcx
from qrisp.core.gate_application_functions import gphase, h, measure, reset, ry, x, z
from qrisp.environments import conjugate, control, invert
from qrisp.jasp import (
    RUS,
    count_ops,
    depth,
    expectation_value,
    jrange,
    num_qubits,
    q_switch,
)
from qrisp.jasp import (
    check_for_tracing_mode as is_tracing,
)
from qrisp.jasp.tracing_logic import QuantumVariableTemplate
from qrisp.qtypes import QuantumBool, QuantumFloat

if TYPE_CHECKING:
    from qrisp.interface.backend import BackendLike

    # The pytree (children, aux_data) shapes produced/consumed by
    # tree_flatten/tree_unflatten below.
    PyTreeChildren = tuple[ArrayLike, list[QuantumVariableTemplate]]
    PyTreeAuxData = tuple[Callable[..., None], int, bool]


def _is_non_negative_real(value: Any) -> bool:
    try:
        value = np.asarray(value)
    except Exception:
        return False
    return bool(np.all(np.isreal(value)) and np.all(np.real(value) >= 0))


@register_pytree_node_class
@dataclass(frozen=False)
class BlockEncoding:
    r"""Central structure for representing block-encodings.

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
    ancillas : Sequence[QuantumVariable | QuantumVariableTemplate]
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
    - The ``is_hermitian`` attribute indicates whether the block-encoding unitary $U_A$ is Hermitian.
      This is distinct from the operator $A$ being being Hermitian. A Hermitian operator $A$ 
      can be block-encoded using a non-Hermitian unitary $U_A$. Conversely, if the unitary $U_A$
      is Hermitian, then the encoded operator must also be Hermitian.

    Examples
    --------
    **Example 1: Pauli Block Encoding**

    Define a :ref:`QubitOperator` repesenting a Heisenberg Hamiltonian,
    and construct a block-encoding based on LCU for its Pauli strings.

    ::

        from qrisp import *
        from qrisp.block_encodings import BlockEncoding
        from qrisp.operators import X, Y, Z

        H = sum(X(i)*X(i+1) + Y(i)*Y(i+1) + Z(i)*Z(i+1) for i in range(3))
        BE = BlockEncoding.from_operator(H)

        # Apply the Hermitian operator to an initial system state

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

    **Example 2: Custom LCU Block Encoding**

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

    Apply the operator to the initial system state $\ket{0}$.

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

    To perform quantum resource estimation for the quantum program (not counting repetitions), 
    replace the ``@terminal_sampling`` decorator with ``@count_ops(meas_behavior="0")``:

    ::

        @count_ops(meas_behavior="0")
        def main():
            operand = BE.apply_rus(operand_prep)()
            return operand

        main()
        # {'s': 4, 'gphase': 2, 'u3': 6, 't': 14, 't_dg': 16, 'x': 5, 'cx': 54, 
        # 'p': 2, 'h': 16, 'measure': 10}

    To perform resource estimations for the block-encoding use :meth:`resources`:

    ::

        BE.resources(QuantumFloat(3))
        # {'gate counts': {'s': 4, 't_dg': 16, 'h': 16, 't': 14, 'gphase': 2, 
        # 'p': 2, 'x': 5, 'cx': 54, 'u3': 6, 'measure': 4}, 'depth': 48, 'qubits': 9}

    """

    def __init__(
        self,
        alpha: "ArrayLike",
        ancillas: Sequence[QuantumVariable | QuantumVariableTemplate],
        unitary: Callable[..., None],
        num_ops: int = 1,
        is_hermitian: bool = False,
    ) -> None:
        self.alpha = alpha
        # Templates for the ancilla variables.
        self._anc_templates: list[QuantumVariableTemplate] = [
            anc.template() if isinstance(anc, QuantumVariable) else anc for anc in ancillas
        ]
        self.unitary = unitary
        self.is_hermitian = is_hermitian
        self.num_ancs = len(ancillas)
        # More robust than inferring the number of operands for the unitary via inspect.
        self.num_ops = num_ops

    def tree_flatten(self) -> tuple[PyTreeChildren, PyTreeAuxData]:
        """PyTree flatten for JAX.

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
    def tree_unflatten(cls: type[BlockEncoding], aux_data: PyTreeAuxData, children: PyTreeChildren) -> BlockEncoding:
        """PyTree unflatten for JAX.

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

    #
    # Utilities
    #

    def create_ancillas(self) -> list[QuantumVariable]:
        r"""Returns a list of ancilla QuantumVariables for the BlockEncoding.

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
        r"""Applies the BlockEncoding unitary to the given operands.

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
            raise ValueError(f"Operation expected {self.num_ops} operands, but got {len(operands)}.")

        ancillas = self.create_ancillas()
        self.unitary(*ancillas, *operands)
        return ancillas

    def apply_rus(self, operand_prep: Callable[..., Any]) -> Callable[..., Any]:
        r"""Applies the BlockEncoding using :ref:`RUS`.

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
        if not callable(operand_prep):
            raise TypeError(f"Expected 'operand_prep' to be a callable, but got {type(operand_prep).__name__}.")

        @RUS
        def rus_function(*args):
            operands = operand_prep(*args)
            if not isinstance(operands, tuple):
                operands = (operands,)

            if len(operands) != self.num_ops:
                raise ValueError(f"Operation expected {self.num_ops} operands, but got {len(operands)}.")

            ancillas = self.create_ancillas()
            self.unitary(*ancillas, *operands)

            bools = jnp.array([(measure(anc) == 0) for anc in ancillas])
            success_bool = jnp.all(bools)

            # garbage collection
            for anc in ancillas:
                reset(anc)
                anc.delete()

            return success_bool, *operands

        return rus_function

    def expectation_value(
        self,
        operand_prep: Callable[..., Any],
        shots: int = 100,
        backend: "BackendLike | None" = None,
    ) -> Callable[..., Any]:
        r"""Measures the expectation value of the operator using the Hadamard test protocol.

        For a block-encoded **Hermitian** operator $A$ and a state $\ket{\psi}$,
        this method measures the expectation value $\langle\psi|A|\psi\rangle$.

        Parameters
        ----------
        operand_prep : Callable
            A function ``operand_prep(*args)`` that prepares and returns the operand QuantumVariables.
        shots : int
            The amount of samples to take to compute the expectation value. The default is 100.
        backend : BackendLike, optional
            The backend on which to evaluate the quantum circuit. By default the Qrisp simulator is used.
            Ignored in Jasp mode.

        Returns
        -------
        Callable
            A function ``ev_function(*args)`` with the same signature
            as ``operand_prep`` returning

            - Jasp mode:
                a Jax array containing the expactation value,
            - Standard mode:
                a NumPy float representing the expectation value.

        Notes
        -----
        - **Precision:** The number of shots $N$ required for target precision $\epsilon$ scales quadratically with the inverse precision $(N\propto 1/\epsilon^2)$.

        Raises
        ------
        TypeError
            If ``operand_prep`` is not a callable object.
        ValueError
            If the number of provided operands does not match
            the required number of operands (self.num_ops).

        Examples
        --------
        **Example 1: Jasp Mode (Dynamic Execution)**

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

            @jaspify(terminal_sampling=True)
            def main(BE):
                ev = BE.expectation_value(operand_prep, shots=10000)(np.pi / 4)
                return ev

            ev = main(BE)
            print(ev)
            # 0.3429

        **Example 2: Standard Mode (Static Execution)**

        ::

            # Using the same Hamiltonian and prep function from the previous example
            ev = BE.expectation_value(operand_prep, shots=10000)(np.pi / 4)
            print(ev)
            # 0.3426

        """
        if not callable(operand_prep):
            raise TypeError(f"Expected 'operand_prep' to be a callable, but got {type(operand_prep).__name__}.")

        def state_prep(*args):
            operands = operand_prep(*args)
            if not isinstance(operands, tuple):
                operands = (operands,)

            if len(operands) != self.num_ops:
                raise ValueError(f"Operation expected {self.num_ops} operands, but got {len(operands)}.")

            # Hadamard test
            qbl = QuantumBool()
            h(qbl)
            with control(qbl):
                self.apply(*operands)
            h(qbl)
            return qbl

        # Dynamic (Jasp) mode
        if is_tracing():

            @jax.jit
            def post_processor(val):
                return jnp.where(val == 0, 1, -1)

            def ev_function_dynamic(*args):
                ev = expectation_value(state_prep, shots=shots, post_processor=post_processor)(*args)
                return ev * self.alpha

            return ev_function_dynamic

        # Static mode
        def ev_function_static(*args):
            qbl = state_prep(*args)
            res_dict = qbl.get_measurement(shots=shots, backend=backend)
            ev = res_dict.get(0, 0) - res_dict.get(1, 0)
            return np.float64(ev * jnp.float64(self.alpha))

        return ev_function_static

    def resources(
        self,
        *operands: QuantumVariable,
        meas_behavior: str | Callable = "0",
        max_qubits: int = 1024,
        max_allocations: int = 1000,
    ) -> dict[str, Any]:
        r"""Estimate the quantum resources required for the BlockEncoding.

        This method uses the ``count_ops``, ``depth`` and ``num_qubits`` decorators to obtain gate counts, circuit depth,
        and qubit usage for a single execution of block-encoding ``.unitary``.
        Unlike :meth:`apply_rus`, it does not run the simulator
        and does not include repetitions from the :ref:`RUS` procedure.

        .. warning::

            The :ref:`depth <depth>` metric is an experimental feature and may not behave as expected in certain edge cases.

        Parameters
        ----------
        *operands : QuantumVariable
            QuantumVariables serving as operands for the block-encoding.
        meas_behavior : str or callable, optional
            Specifies the measurement outcome to assume during the tracing process (e.g., "0", or "1"). The default is "0".
        max_qubits : int, optional
            The maximum number of qubits supported for depth computation. Default is 1024.
        max_allocations : int, optional
            The maximum number of allocation/deallocation events supported for tracking. Default is 1000.

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
            # 'depth': 12, 'qubits': 4}

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
            # 'gphase': 6, 'p': 2}, 'depth': 42, 'qubits': 5}

        """
        if len(operands) != self.num_ops:
            raise ValueError(f"Operation expected {self.num_ops} operands, but got {len(operands)}.")

        ops_templates = [op.template() for op in operands]

        def operand_prep():
            operands = [temp.construct() for temp in ops_templates]
            return operands

        def main():
            operands = operand_prep()
            ancillas = self.create_ancillas()
            self.unitary(*ancillas, *operands)
            return operands

        circuit_depth = depth(meas_behavior=meas_behavior, max_qubits=max_qubits)(main)()
        gate_counts = count_ops(meas_behavior=meas_behavior)(main)()
        qubit_counts = num_qubits(meas_behavior=meas_behavior, max_allocations=max_allocations)(main)()
        return {
            "gate counts": gate_counts,
            "depth": circuit_depth,
            "qubits": qubit_counts["peak_allocations"],
        }

    def dagger(self) -> BlockEncoding:
        r"""Returns a new BlockEncoding representing the Hermitian conjugate of the operator.

        For a block-encoded operator $A$ with block-encoding unitary $U_A$, this method returns a new BlockEncoding with unitary $U_A^{\dagger}$.
        The resulting block-encoding represents the operator $A^{\dagger}$ with the same scaling factor $\alpha$.

        Returns
        -------
        BlockEncoding
            A new BlockEncoding instance representing the Hermitian conjugate of the operator.

        Examples
        --------
        Define a block-encoding and obtain its Hermitian conjugate.

        ::

            from qrisp.block_encodings import BlockEncoding
            from qrisp.operators import X, Y, Z

            H = X(0)*Y(1) + 0.5*Z(0)*X(1)
            BE = BlockEncoding.from_operator(H)
            BE_dg = BE.dagger()

        """

        def new_unitary(*args):
            with invert():
                self.unitary(*args)

        return BlockEncoding(
            self.alpha,
            self._anc_templates,
            new_unitary,
            num_ops=self.num_ops,
            is_hermitian=self.is_hermitian,
        )

    def qubitization(self) -> BlockEncoding:
        r"""Returns a BlockEncoding representing the `qubitization walk operator <https://quantum-journal.org/papers/q-2019-07-12-163/>`_.

        For a block-encoded **Hermitian** operator $A$ with normalization factor $\alpha$,
        this method returns a BlockEncoding of the qubitization walk operator $W$
        satisfying $W^k=T_k(A/\alpha)$ where $T_k$ is the $k$-Chebyshev polynomial of the first kind.

        The action of $W$ partitions the Hilbert space into a direct sum of two-dimensional invariant subspaces giving it the name "qubitization".
        For an eigenstate $\ket{\lambda}$ of $A$ with eigenvalue $\lambda$, the two-dimensional space is spanned by

        - $\ket{\phi_1} = \ket{0}_a\ket{\lambda}$
        - $\ket{\phi_2} = \frac{(W-\lambda/\alpha\mathbb I)\ket{\phi_1}}{\sqrt{1-(\lambda/\alpha)^2}}$

        In this subspace, $W$ implements a Pauli-Y rotaion by angle $\theta=-2\arccos(\lambda/\alpha)$, i.e., $W=e^{i\arccos(\lambda/\alpha)Y}$.

        If the block-encoding unitary $U$ is Hermitian (i.e., $U^2=\mathbb I$), then $W=R U$ where $R = (2\ket{0}_a\bra{0}_a - \mathbb I)$
        is the reflection around the state $\ket{0}_a$ of the ancilla variables.
        Otherwise, $W = R \tilde{U}$ where $\tilde{U} = (H \otimes \mathbb I)(\ket{0}\bra{1} \otimes U) + (\ket{1}\bra{0} \otimes U^{\dagger})(H \otimes \mathbb I)$
        is a Hermitian block-encoding of $A=(A+A^{\dagger})/2$ requiring one additional ancilla qubit.
        Conjugation by $(H \otimes \mathbb I)$ rotates the basis so that the new ancilla is initialized and projected in $\ket{0}$, ensuring that $A$ sits in the upper-left block.

        Returns
        -------
        BlockEncoding
            A new BlockEncoding instance representing the qubitization walk operator.

        Notes
        -----
        - **Normalization**: The resulting block-encoding maintains the same scaling factor $\alpha$ as the original.

        References
        ----------
        - Low & Chuang (2019) `Hamiltonian Simulation by Qubitization <https://quantum-journal.org/papers/q-2019-07-12-163/>`_.

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
            def new_unitary_hermitian(*args):
                self.unitary(*args)
                reflection(args[:m])

            return BlockEncoding(
                self.alpha,
                self._anc_templates,
                new_unitary_hermitian,
                num_ops=self.num_ops,
                is_hermitian=True,
            )

        # We construct a strictly Hermitian block-encoding of A = (A + A†)/2.
        # To guarantee Hermiticity, we use an off-diagonal control structure:
        # S = (|0><1| ⊗ U) + (|1><0| ⊗ U†)
        # S is strictly Hermitian (S = S†). We implement S by applying an X gate
        # to the control ancilla before the controlled-U and controlled-U† operations.
        #
        # We then conjugate S by (H ⊗ I) to form U_tilde = (H ⊗ I) S (H ⊗ I).
        # This unitary is still strictly Hermitian, and the Hadamard sandwich
        # rotates the basis so that the new ancilla is initialized and projected in |0>.
        # Therefore, A cleanly sits in the upper-left block because:
        # <0| U_tilde |0> = <+| S |+> = (U + U†)/2.
        #
        # The qubitization walk operator is W = (2*|0><0| - I) U_tilde.
        def new_unitary(*args):
            with conjugate(h)(args[0]):
                # (|0><1| + |1><0|) = X
                x(args[0])

                # (|0><0| ⊗ U)
                with control(args[0], ctrl_state=0):
                    self.unitary(*args[1:])
                # (|1><1| ⊗ U†)
                with control(args[0], ctrl_state=1):
                    with invert():
                        self.unitary(*args[1:])

            reflection(args[0 : 1 + m])

        new_anc_templates = [QuantumBool().template()] + self._anc_templates
        return BlockEncoding(
            self.alpha,
            new_anc_templates,
            new_unitary,
            num_ops=self.num_ops,
            is_hermitian=True,
        )

    def _hermitianization(self) -> BlockEncoding:
        r"""Returns a BlockEncoding representing the `qubitization walk operator via Hermitianization <https://arxiv.org/pdf/2312.00723>`_.

        For a block-encoded (**not** necessarily Hermitian) operator $A$,
        this method returns a BlockEncoding of the qubitization walk operator via Hermitianization.
        The operator $A$ is encoded in the upper right block (measure new ancilla QuantumBool in $\ket{1}$):

        .. math::

            \begin{pmatrix} \mathbb{0} & A \\ A^{\dagger} & \mathbb{0} \end{pmatrix}

        """
        n = self.num_ancs

        def new_unitary(*args):

            anc = args[0]
            self_ancs = args[1 : n + 1]
            operands = args[n + 1 :]

            x(anc)

            with control(anc, ctrl_state=0):
                self.unitary(*self_ancs, *operands)

            with control(anc, ctrl_state=1):
                with invert():
                    self.unitary(*self_ancs, *operands)

            reflection(self_ancs)

        new_anc_templates = [QuantumBool().template()] + self._anc_templates
        return BlockEncoding(
            self.alpha,
            new_anc_templates,
            new_unitary,
            num_ops=self.num_ops,
            is_hermitian=True,
        )

    def chebyshev(self, k: int, rescale: bool = True) -> BlockEncoding:
        r"""Returns a BlockEncoding representing $k$-th Chebyshev polynomial of the first kind applied to the operator.

        For a block-encoded **Hermitian** operator $A$ with normalization factor $\alpha$,
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
        return BlockEncoding(new_alpha, self._anc_templates, new_unitary, num_ops=self.num_ops)

    #
    # Arithmetic
    #

    def _get_lcu_terms(self) -> _LCUTerms:
        return ((1, self),)

    def _get_product_factors(self) -> _ProductFactors:
        return (self,)

    @classmethod
    def linear_combination(
        cls,
        block_encodings: list[BlockEncoding],
        coefficients: "ArrayLike | None" = None,
    ) -> BlockEncoding:
        r"""Returns a BlockEncoding of a linear combination of operators.

        This method implements the linear combination $\sum_i\alpha_iA_i$ via the LCU
        (Linear Combination of Unitaries) framework, where $A_i$ are
        the operators encoded by the respective instances, and $\alpha_i$ are real coefficients.

        Equivalently, block-encoded operators can also be combined via ``+``, ``-`` and scalar multiplication.

        Parameters
        ----------
        block_encodings : list[BlockEncoding]
            Block-encodings of operators acting on the same operands.
        coefficients : ArrayLike, optional
            Coefficients multiplying the encoded operators. Defaults to one
            for every block-encoding.

        Returns
        -------
        BlockEncoding
            A block-encoding of the linear combination.

        Raises
        ------
        ValueError
            If no block-encodings are supplied, the coefficient count does
            not match, or operand counts differ.
        TypeError
            If an item is not a BlockEncoding.

        """
        if len(block_encodings) == 0:
            raise ValueError("At least one block-encoding is required.")

        if coefficients is None:
            coefficients = [1] * len(block_encodings)
        elif len(coefficients) != len(block_encodings):
            raise ValueError("The number of coefficients must match the number of block-encodings.")

        terms = []
        for block_encoding, coefficient in zip(block_encodings, coefficients):
            if not isinstance(block_encoding, BlockEncoding):
                raise TypeError(f"Expected every item to be a BlockEncoding, but got {type(block_encoding).__name__}.")
            terms.append((coefficient, block_encoding))

        return cls._from_lcu_terms(terms)

    @classmethod
    def _from_lcu_terms(cls, terms: Sequence[_LCUTerm]) -> BlockEncoding:
        """Build a linear-combination block encoding from weighted terms."""
        terms = _canonicalize_lcu_terms(terms)
        if len(terms) == 0:
            raise ValueError("Cannot construct a block encoding from an all-zero linear combination.")
        if len(terms) == 1:
            coefficient, block_encoding = terms[0]
            if isinstance(coefficient, (int, float, complex, np.number)) and coefficient == 1:
                return block_encoding
        return LinearCombinationBlockEncoding(terms)

    def __add__(self, other: BlockEncoding) -> BlockEncoding:
        r"""Returns a BlockEncoding of the sum of two operators.

        This method implements the addition $A + B$ via the LCU
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

        return type(self)._from_lcu_terms(self._get_lcu_terms() + other._get_lcu_terms())

    def __sub__(self, other: BlockEncoding) -> BlockEncoding:
        r"""Returns a BlockEncoding of the difference between two operators.

        This method implements the subtraction $A - B$ via the LCU
        (Linear Combination of Unitaries) framework, where $A$ and $B$ are
        the operators encoded by the respective instances.

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

        other_terms = tuple((-coefficient, block_encoding) for coefficient, block_encoding in other._get_lcu_terms())
        return type(self)._from_lcu_terms(self._get_lcu_terms() + other_terms)

    def __mul__(self, other: "ArrayLike") -> BlockEncoding:
        r"""Returns a BlockEncoding of the scaled operator.

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

        if isinstance(other, ArrayLike):
            terms = [(other * coefficient, block_encoding) for coefficient, block_encoding in self._get_lcu_terms()]
            return type(self)._from_lcu_terms(terms)

        return NotImplemented

    def __matmul__(self, other: BlockEncoding) -> BlockEncoding:
        r"""Returns a BlockEncoding of the product of two operators.

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

        return ProductBlockEncoding(self._get_product_factors() + other._get_product_factors())

    def __radd__(self, other: Any) -> BlockEncoding | NotImplementedType:
        """Support adding a BlockEncoding to the start value used by sum()."""
        if other == 0:
            return self
        return NotImplemented

    __rmul__ = __mul__

    def kron(self, other: BlockEncoding) -> BlockEncoding:
        r"""Returns a BlockEncoding of the Kronecker product (tensor product) of two operators.

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
        r"""Returns a BlockEncoding of the negated operator.

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

    # ------------------------------------------------------------------
    # The methods below are attached to this class after its definition, in
    # block_encoding.py: each one is implemented in its own module under
    # constructors/ or transformations/, and each of those modules needs to
    # import BlockEncoding itself, so importing them here at runtime would
    # be circular. Re-declaring them under TYPE_CHECKING (never executed at
    # runtime) makes them visible to type checkers as ordinary members of
    # this class, without changing any runtime behaviour or reintroducing
    # that circular import.
    # ------------------------------------------------------------------
    if TYPE_CHECKING:
        from .constructors import (
            build_from_array,
            build_from_eye,
            build_from_foqcs_lcu_operator,
            build_from_foqcs_lcu_prep,
            build_from_lcu,
            build_from_operator,
            build_from_projector,
        )
        from .transformations import (
            apply_inv,
            apply_poly,
            apply_pseudo_inv,
            apply_sim,
            apply_svt,
        )

        from_array = classmethod(build_from_array)
        from_eye = classmethod(build_from_eye)
        from_lcu = classmethod(build_from_lcu)
        from_foqcs_lcu_prep = classmethod(build_from_foqcs_lcu_prep)
        from_foqcs_lcu_operator = classmethod(build_from_foqcs_lcu_operator)
        from_operator = classmethod(build_from_operator)
        from_projector = classmethod(build_from_projector)

        inv = apply_inv
        poly = apply_poly
        pseudo_inv = apply_pseudo_inv
        sim = apply_sim
        svt = apply_svt


_LCUTerm = tuple[ArrayLike, BlockEncoding]
_LCUTerms = tuple[_LCUTerm, ...]
_ProductFactors = tuple[BlockEncoding, ...]
_ProductStrategy = Literal["separate", "qubit_efficient"]


def _is_statically_zero(value: Any) -> bool:
    """Return whether ``value`` is a concrete scalar zero."""
    if isinstance(value, jax.core.Tracer):
        return False
    try:
        value = np.asarray(value)
    except Exception:
        return False
    return value.ndim == 0 and bool(value == 0)


def _canonicalize_lcu_terms(terms: Sequence[_LCUTerm]) -> _LCUTerms:
    """Merge identity-equal child encodings and remove concrete zero terms."""
    merged_terms: list[_LCUTerm] = []
    for coefficient, block_encoding in terms:
        for index, (_, existing_block_encoding) in enumerate(merged_terms):
            if existing_block_encoding is block_encoding:
                merged_terms[index] = (merged_terms[index][0] + coefficient, block_encoding)
                break
        else:
            merged_terms.append((coefficient, block_encoding))

    return tuple(
        (coefficient, block_encoding)
        for coefficient, block_encoding in merged_terms
        if not _is_statically_zero(coefficient)
    )


@register_pytree_node_class
class LinearCombinationBlockEncoding(BlockEncoding):
    """A block encoding represented by an immutable tuple of LCU terms."""

    def __init__(self, terms: Sequence[_LCUTerm]) -> None:
        r"""Initialize a linear-combination block encoding from weighted terms.

        Each term is a ``(coefficient, block_encoding)`` pair. If the child
        block-encoding represents $A_i / \alpha_i$, the combined LCU uses
        ``coefficient * child.alpha`` as the coefficient of its child unitary.
        The resulting normalization is therefore
        ``sum(abs(coefficient * child.alpha))``.

        The child ancillas are packed into one shared workspace after one
        selector register. The workspace is sized to the largest child layout,
        and each selected branch reconstructs its typed ancillas as views into
        that workspace. Since the selector chooses exactly one child unitary,
        nested sums are lowered to one preparation/select/preparation sequence.
        The terms are retained as the immutable authoritative representation;
        the compiled unitary and metadata are derived from them.

        Parameters
        ----------
        terms : tuple[tuple[ArrayLike, BlockEncoding], ...]
            Weighted block-encoding terms. The terms must be non-empty, and all
            child block-encodings must have the same number of operands.

        Raises
        ------
        ValueError
            If no terms are supplied or the child operand counts differ.
        TypeError
            If a term does not contain a BlockEncoding.

        Notes
        -----
        A single term with coefficient ``1`` is returned as the original child
        by the factory that constructs this class. A single term with another
        coefficient uses the child's ancillas directly and applies the required
        phase for a negative coefficient.

        """
        terms = _canonicalize_lcu_terms(terms)
        if len(terms) == 0:
            raise ValueError("Cannot construct a block encoding from an all-zero linear combination.")
        if any(not isinstance(block_encoding, BlockEncoding) for _, block_encoding in terms):
            raise TypeError("Expected every item to be a BlockEncoding.")

        num_ops = terms[0][1].num_ops
        if any(block_encoding.num_ops != num_ops for _, block_encoding in terms):
            raise ValueError("All block-encodings must have the same number of operands.")

        self._terms = terms

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent reassignment of the authoritative term representation."""
        if name == "_terms" and hasattr(self, "_terms"):
            raise AttributeError("Linear-combination terms are immutable.")
        object.__setattr__(self, name, value)

    @property
    def terms(self) -> _LCUTerms:
        """The immutable weighted child block encodings."""
        return self._terms

    @property
    def alpha(self) -> ArrayLike:
        """Return the normalization derived from the weighted child encodings."""
        return sum(
            (jnp.abs(coefficient * block_encoding.alpha) for coefficient, block_encoding in self.terms),
            jnp.array(0),
        )

    @property
    def _anc_templates(self) -> list[QuantumVariableTemplate]:
        if len(self.terms) == 1:
            return list(self.terms[0][1]._anc_templates)

        selector_size = (len(self.terms) - 1).bit_length()
        term_layouts = [
            _AncillaLayout.from_templates(block_encoding._anc_templates) for _, block_encoding in self.terms
        ]
        shared_anc_size = _maximum_layout_size(term_layouts)
        return [
            QuantumFloat(selector_size).template(),
            QuantumVariable(shared_anc_size).template(),
        ]

    @property
    def unitary(self) -> Callable[..., None]:
        """Return the PREP-SELECT-PREP unitary derived from the terms."""
        if len(self.terms) == 1:
            coefficient, block_encoding = self.terms[0]

            def unitary(*args):
                block_encoding.unitary(*args)
                with control(coefficient < 0):
                    gphase(np.pi, args[0][0])

            return unitary

        term_layouts = [
            _AncillaLayout.from_templates(block_encoding._anc_templates) for _, block_encoding in self.terms
        ]
        selector_size = (len(self.terms) - 1).bit_length()

        def unitary(*args):
            selector = args[0]
            shared_ancilla = args[1]
            branches = []

            for term_index, (_, block_encoding) in enumerate(self.terms):
                child_ancillas = term_layouts[term_index].construct_views(shared_ancilla)

                def branch(
                    *branch_args,
                    block_encoding=block_encoding,
                    child_ancillas=child_ancillas,
                ):
                    block_encoding.unitary(*child_ancillas, *branch_args)

                branches.append(branch)

            def identity_branch(*unused_args):
                pass

            branches.extend([identity_branch] * ((1 << selector_size) - len(branches)))
            operands = args[2:]
            coefficients = jnp.array(
                [coefficient * block_encoding.alpha for coefficient, block_encoding in self.terms],
                dtype=complex,
            )
            coefficients = jnp.pad(coefficients, (0, (1 << selector_size) - len(self.terms)))
            alpha = jnp.sum(jnp.abs(coefficients))
            amplitudes = jnp.sqrt(coefficients / alpha)

            if all(_is_non_negative_real(coefficient) for coefficient, _ in self.terms):
                with conjugate(prepare)(selector, jnp.abs(amplitudes)):
                    q_switch(selector, branches, *operands)
            else:
                prepare(selector, amplitudes)
                q_switch(selector, branches, *operands)
                with invert():
                    prepare(selector, jnp.conjugate(amplitudes))

        return unitary

    @property
    def num_ops(self) -> int:
        """Return the common number of operands in the child encodings."""
        return self.terms[0][1].num_ops

    @property
    def num_ancs(self) -> int:
        """Return the number of selector and shared-workspace ancillas."""
        return len(self._anc_templates)

    @property
    def is_hermitian(self) -> bool:
        """Return whether every child encoding has a Hermitian unitary."""
        return all(block_encoding.is_hermitian for _, block_encoding in self.terms)

    def _get_lcu_terms(self) -> _LCUTerms:
        return self.terms

    def tree_flatten(self) -> tuple[tuple[ArrayLike | BlockEncoding, ...], int]:
        """Flatten only the authoritative terms for JAX pytree handling."""
        children = tuple(value for term in self.terms for value in term)
        return children, len(self.terms)

    @classmethod
    def tree_unflatten(
        cls: type[LinearCombinationBlockEncoding],
        aux_data: int,
        children: tuple[ArrayLike | BlockEncoding, ...],
    ) -> LinearCombinationBlockEncoding:
        """Reconstruct a linear combination from flattened terms."""
        terms = tuple((children[index], children[index + 1]) for index in range(0, 2 * aux_data, 2))
        return cls(terms)


@register_pytree_node_class
class ProductBlockEncoding(BlockEncoding):
    """A block encoding represented by an immutable product of factors.

    Factors are stored in mathematical order. The unitary applies them in
    reverse order, so ``A @ B`` applies ``B`` before ``A``. By default, the
    qubit-efficient strategy packs the factor ancillas into one shared
    workspace and uses a shift register to separate the garbage generated at
    successive steps. The separate strategy retains distinct ancillas for
    every factor.

    Parameters
    ----------
    factors : Sequence[BlockEncoding]
        Block-encoding factors in mathematical order.
    strategy : {"separate", "qubit_efficient"}
        Unitary implementation strategy. ``"qubit_efficient"`` is the default;
        it reuses one workspace and records failed factor applications in a
        shift register. ``"separate"`` uses distinct ancillas for every factor.

    """

    def __init__(
        self,
        factors: Sequence[BlockEncoding],
        strategy: _ProductStrategy = "qubit_efficient",
    ) -> None:
        if strategy not in ("separate", "qubit_efficient"):
            raise ValueError(f"Unknown product strategy: {strategy}")

        flattened_factors: list[BlockEncoding] = []
        for factor in factors:
            if not isinstance(factor, BlockEncoding):
                raise TypeError(f"Expected every factor to be a BlockEncoding, but got {type(factor).__name__}.")
            flattened_factors.extend(factor._get_product_factors())

        if len(flattened_factors) == 0:
            raise ValueError("At least one product factor is required.")

        num_ops = flattened_factors[0].num_ops
        if any(factor.num_ops != num_ops for factor in flattened_factors):
            raise ValueError("All product factors must have the same number of operands.")

        self._factors = tuple(flattened_factors)
        self._strategy = strategy

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent reassignment of the authoritative factor representation."""
        if name in {"_factors", "_strategy"} and hasattr(self, name):
            raise AttributeError("Product representation is immutable.")
        object.__setattr__(self, name, value)

    @property
    def factors(self) -> _ProductFactors:
        """The immutable product factors in mathematical order."""
        return self._factors

    @property
    def strategy(self) -> _ProductStrategy:
        """The unitary implementation strategy used by the product."""
        return self._strategy

    @property
    def alpha(self) -> ArrayLike:
        """Return the normalization derived from the product factors."""
        alpha = 1
        for factor in self.factors:
            alpha = alpha * factor.alpha
        return alpha

    @property
    def _anc_templates(self) -> list[QuantumVariableTemplate]:
        if self.strategy == "qubit_efficient":
            if len(self.factors) == 1:
                return list(self.factors[0]._anc_templates)
            if all(factor.num_ancs == 0 for factor in self.factors):
                return []

            factor_layouts = tuple(_AncillaLayout.from_templates(factor._anc_templates) for factor in self.factors)
            shift_size = (len(self.factors) - 1).bit_length()
            return [
                QuantumFloat(shift_size).template(),
                QuantumVariable(_maximum_layout_size(factor_layouts)).template(),
            ]
        return [template for factor in self.factors for template in factor._anc_templates]

    @property
    def unitary(self) -> Callable[..., None]:
        """Return the unitary selected by the product implementation strategy."""
        if self.strategy == "qubit_efficient":
            return self._unitary_qubit_efficient
        return self._unitary_separate

    @property
    def _unitary_separate(self) -> Callable[..., None]:
        """Return the unitary that composes factors with separate ancillas."""
        factor_ancilla_counts = tuple(factor.num_ancs for factor in self.factors)

        def unitary(*args):
            total_ancillas = sum(factor_ancilla_counts)
            operands = args[total_ancillas:]
            offset = 0
            factor_args = []
            for factor, num_ancillas in zip(self.factors, factor_ancilla_counts):
                factor_ancillas = args[offset : offset + num_ancillas]
                factor_args.append((factor, factor_ancillas))
                offset += num_ancillas

            for factor, factor_ancillas in reversed(factor_args):
                factor.unitary(*factor_ancillas, *operands)

        return unitary

    @property
    def _unitary_qubit_efficient(self) -> Callable[..., None]:
        """Return the unitary using one shared workspace and a shift register."""
        if len(self.factors) == 1 or all(factor.num_ancs == 0 for factor in self.factors):

            def unitary(*args):
                operands = args[self.num_ancs :]
                factor_ancillas = args[: self.num_ancs]
                for factor in reversed(self.factors):
                    factor.unitary(*factor_ancillas, *operands)

            return unitary

        factor_layouts = tuple(_AncillaLayout.from_templates(factor._anc_templates) for factor in self.factors)

        def unitary(*args):
            shift_register = args[0]
            shared_workspace = args[1]
            operands = args[2:]
            zero_flag = QuantumBool()

            for factor_index, factor in enumerate(reversed(self.factors)):
                layout_index = len(self.factors) - 1 - factor_index
                factor_layout = factor_layouts[layout_index]
                factor_ancillas = factor_layout.construct_views(shared_workspace)

                # Only the shift-zero sector carries the still-valid product
                # branch. Other sectors contain garbage from earlier factors.
                with conjugate(mcx)(shift_register, zero_flag, ctrl_state=0):
                    with control(zero_flag):
                        factor.unitary(*factor_ancillas, *operands)

                if factor_index == len(self.factors) - 1 or len(factor_layout.sizes) == 0:
                    continue

                # Implement |s, w> -> |s + 1, w> for w != 0 and leave w = 0
                # fixed on the workspace prefix used by this factor. This
                # reversible permutation moves newly generated garbage out of
                # shift zero without modifying its workspace value. The
                # compute-control-uncompute pattern restores the temporary
                # predicate qubit after every permutation.
                active_workspace = shared_workspace.reg[: factor_layout.total_size]
                with conjugate(mcx)(active_workspace, zero_flag, ctrl_state=0):
                    with control(zero_flag, ctrl_state=0):
                        shift_register += 1

            zero_flag.delete()

        return unitary

    @property
    def num_ops(self) -> int:
        """Return the common number of operands in the product factors."""
        return self.factors[0].num_ops

    @property
    def num_ancs(self) -> int:
        """Return the number of ancilla variables used by the strategy."""
        if self.strategy == "qubit_efficient":
            return len(self._anc_templates)
        return sum(factor.num_ancs for factor in self.factors)

    @property
    def is_hermitian(self) -> bool:
        """Return whether the product is known to have a Hermitian unitary."""
        return len(self.factors) == 1 and self.factors[0].is_hermitian

    def _get_product_factors(self) -> _ProductFactors:
        return self.factors

    def dagger(self) -> ProductBlockEncoding:
        """Return the product dagger with reversed, individually inverted factors."""
        return ProductBlockEncoding(
            tuple(factor.dagger() for factor in reversed(self.factors)),
            strategy=self.strategy,
        )

    def tree_flatten(self) -> tuple[_ProductFactors, _ProductStrategy]:
        """Flatten the authoritative product factors for JAX pytree handling."""
        return self.factors, self.strategy

    @classmethod
    def tree_unflatten(
        cls: type[ProductBlockEncoding],
        aux_data: _ProductStrategy,
        children: tuple[BlockEncoding, ...],
    ) -> ProductBlockEncoding:
        """Reconstruct a product from flattened factors."""
        return cls(children, strategy=aux_data)
