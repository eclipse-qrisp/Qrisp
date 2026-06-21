"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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

from typing import Callable, Optional, Tuple, Union

from qrisp.block_encodings.block_encoding_base import BlockEncoding
from qrisp.core.gate_application_functions import mcx, x
from qrisp.environments import invert
from qrisp.qtypes import QuantumBool


def build_from_projector(
    cls: BlockEncoding,
    left: Union[int, Tuple[int, ...], Callable],
    right: Optional[Union[int, Tuple[int, ...], Callable]] = None,
    kernel: bool = False,
    num_ops: int = 1,
) -> BlockEncoding:
    r"""
    Constructs a BlockEncoding of a projector.

    Parameters
    ----------
    left : int | tuple of int | Callable
        An integer or a tuple of integers representing a computational basis state $\ket{\phi}$,
        or a function ``left(*operands)`` preparing a state $\ket{\phi}$ from $\ket{0}$.
    right : int | tuple of int | Callable
        An integer or a tuple of integers representing a computational basis state $\ket{\psi}$,
        or a function ``right(*operands)`` preparing a state $\ket{\psi}$ from $\ket{0}$.
        Defaults to ``left``.
    kernel : bool
        If `True`, the kernel projector $\mathbb I - \ket{\phi}\bra{\phi}$ is block-encoded.
        If `False`, the projector $\ket{\phi}\bra{\psi}$ is block-encoded. Defaults to `False`.
    num_ops : int
        The number of operand quantum variables.
        Automatically inferred when ``left`` or ``right`` is an integer or tuple of integers.
        Defaults to 1.

    Returns
    -------
    BlockEncoding
        A BlockEncoding representing either the projector $\ket{\phi}\bra{\psi}$
        or the kernel projector $\mathbb I - \ket{\phi}\bra{\phi}$, depending on the value of ``kernel``.

    Examples
    --------

    **Example 1: Computational basis states**

    Define a block-encoding for the projector $P=\ket{1}\bra{3}$.

    ::

        from qrisp import *
        from qrisp.block_encodings import BlockEncoding

        P = BlockEncoding.from_projector(1, 3)

        # Prepare operand in superposition state
        def operand_prep():
            operand = QuantumFloat(2)
            h(operand)
            return operand

        @terminal_sampling
        def main():
            operand = P.apply_rus(operand_prep)()
            return operand

        res_dict = main()
        print(res_dict)
        # {1.0: 1.0}

    **Example 2: Custom states**

    Define a block-encoding for the projector $P=\ket{\psi}\bra{\psi}$ where $\ket{\psi}\propto\ket{0}+\ket{1}+\ket{2}+\ket{3}$.

    ::

        from qrisp import *
        from qrisp.block_encodings import BlockEncoding

        def prep_psi(qv):
            h(qv)

        P = BlockEncoding.from_projector(prep_psi)

        # Prepare operand in |0> state
        def operand_prep():
            operand = QuantumFloat(2)
            return operand

        @terminal_sampling
        def main():
            operand = P.apply_rus(operand_prep)()
            return operand

        res_dict = main()
        print(res_dict)
        # {0.0: 0.25, 1.0: 0.25, 2.0: 0.25, 3.0: 0.25}

    """

    if kernel or (right == None):
        right = left

    # left
    num_left = 1
    if isinstance(left, int):

        def prep_left(arg):
            arg.encode(left, permit_dirtyness=True)

    elif isinstance(left, tuple):
        num_left = len(left)

        def prep_left(*args):
            for i, arg in enumerate(args):
                arg.encode(left[i], permit_dirtyness=True)

    elif callable(left):
        prep_left = left
    else:
        return NotImplemented

    # right
    num_right = 1
    if isinstance(right, int):

        def prep_right(arg):
            arg.encode(right, permit_dirtyness=True)

    elif isinstance(right, tuple):
        num_right = len(right)

        def prep_right(*args):
            for i, arg in enumerate(args):
                arg.encode(right[i], permit_dirtyness=True)

    elif callable(right):
        prep_right = right
    else:
        return NotImplemented

    if not (isinstance(left, Callable) or isinstance(right, Callable)):
        if num_left != num_right:
            raise ValueError(f"Size mismatch: left has {num_left} elements, but right has {num_right}.")
    num_ops = max(num_left, num_right, num_ops)

    def unitary(*args):
        anc = args[0]
        operands = args[1:]

        if not kernel:
            x(anc)

        with invert():
            prep_right(*operands)

        qubits = sum([operand.reg for operand in operands], [])
        mcx(qubits, anc[0], ctrl_state=0)

        prep_left(*operands)

    return cls(1, [QuantumBool().template()], unitary, num_ops=num_ops)
