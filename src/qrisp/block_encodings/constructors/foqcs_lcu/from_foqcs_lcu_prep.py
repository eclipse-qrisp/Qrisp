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

from typing import Callable

from qrisp.block_encodings.block_encoding_base import BlockEncoding
from qrisp.environments import conjugate, invert
from qrisp.jasp import qache
from qrisp.qtypes import QuantumVariable
from qrisp.core import cx, cz
from .foqcs_preps import get_foqcs_lcu_prep_num_of_ancillae


def build_from_foqcs_lcu_prep(
    cls: BlockEncoding,
    prep_r: Callable[[QuantumVariable], None],
    prep_l: Callable[[QuantumVariable], None],
    num_q_ops: int = 1,
    is_hermitian: bool = False,
    norm: "ArrayLike" = 1,
    num_q_anc: int = -1
) -> BlockEncoding:
    r"""

    .. note::

        This implementation is designed for building custom FOQCS-LCU
        block encodings. For automatic construction from a given operator, use
        :meth:`from_foqcs_lcu_operator`.

    This method implements the Fast One-Qubit-Controlled Select Linear Combination of Unitaries (FOQCS-LCU) structure.
    The provided ``prep_r`` routine prepares the right PREP (:math:`PREP_{R}`) state on the FOQCS-LCU ancilla register.
    The provided ``prep_l`` routine is the corresponding left PREP (:math:`PREP_{L}`) routine and is applied inversely
    after ``SELECT``. Based on the methodology established in https://arxiv.org/abs/2507.20887.

    In order not to restrict this implementation to just the Heisenberg and spin-glass models,
    :meth:`from_foqcs_lcu_prep` is designed using partial functions for the ``prep_r`` and ``prep_l`` parameters, letting
    users pass more or less anything for the :math:`P_{R}` and :math:`P_{L}` subroutines. This means you can experiment
    with different :math:`P_{R}` and :math:`P_{L}` pairs to your hearts desire! 🦊

    Note, custom :math:`P_{R}` and :math:`P_{L}` routines must prepare an ancilla register
    containing at least :math:`2L` qubits, where :math:`L` is the number of operand
    qubits. The final :math:`2L` qubits of this ancilla register are interpreted by
    the FOQCS-LCU ``SELECT`` block as two activation registers of length
    :math:`L`:

    .. math::

        [x_0, \dots, x_{L-1}, z_0, \dots, z_{L-1}].

    The :math:`x_i` qubits control the application of :math:`X_i` to the operand
    register, while the :math:`z_i` qubits control the application of :math:`Z_i`.

    Any additional ancilla qubits required by the custom PREP routine must precede
    these :math:`2L` activation qubits. Thus, if the PREP routine uses
    :math:`m` extra ancillas, the ancilla register layout is

    .. math::

        [\text{extra}_0, \dots, \text{extra}_{m-1},
        x_0, \dots, x_{L-1},
        z_0, \dots, z_{L-1}].

    For example, with :math:`L = 2`, the FOQCS-LCU activation part consists of
    four ancilla qubits,

    .. math::

        [x_0, x_1, z_0, z_1],

    possibly preceded by any extra PREP ancillas.

    Parameters
    ----------
    prep_r : Callable[[QuantumVariable], None]
        Right FOQCS-LCU PREP routine, corresponding to :math:`P_{R} = \mathrm{PREP}(\alpha)`
        The callable should prepare the right coefficient state on the FOQCS-LCU ancilla register.

        The callable is expected to take only the ancilla :class:`QuantumVariable` as its remaining argument.
        All classical parameters of the PREP routine, such as the system size or coefficient dictionaries,
        should already be fixed, for example by using :class:`functools.partial`.

        For example, :func:`foqcs_prep_heisenberg` has parameters of the form

        .. code-block:: python

            def foqcs_prep_heisenberg(
                prep_qv: QuantumVariable | Sequence[Qubit],
                L: int,
                g: dict,
                J: dict,
                conjugate: bool = False
            ) -> None:

        In this case, ``L``, ``g``, and ``J`` should be fixed before passing the
        routine to :func:`from_foqcs_lcu_prep`, leaving only ``prep_qv`` to be
        supplied internally by the :class:`BlockEncoding` during :meth:`apply` or
        :meth:`apply_rus`.

        In this example case the partial construction would look as follows:

        .. code-block:: python

            prep_r = partial(
                foqcs_prep_heisenberg, # PREP function
                L=heis_L,              # Number of qubits
                g=_g,                  # Heisenberg model g coefficients
                J=_J,                  # heisenberg model J coefficients
            )


    prep_l : Callable[[QuantumVariable], None]
        Left FOQCS-LCU PREP routine, corresponding to
        :math:`P_{L} = \mathrm{PREP}(a^*)`. The block-encoding circuit applies this
        routine under inversion, realizing :math:`P_{L}^\dagger` after ``SELECT``

        In the common case, ``prep_l`` is constructed from the same PREP routine as
        ``prep_r``, but with conjugated coefficients. It should follow the same
        calling convention as ``prep_r``: all classical parameters should
        already be fixed, leaving only the ancilla :class:`QuantumVariable` to be
        supplied internally by the :class:`BlockEncoding`.

        Following the example set in ``prep_r``, the partial construction will
        be of form:

        .. code-block:: python

            prep_l = partial(
                foqcs_prep_heisenberg, # PREP function
                L=heis_L,              # Number of qubits
                g=_g,                  # Heisenberg model g coefficients
                J=_J,                  # Heisenberg model J coefficients
                conjugate=True         # Conjugate g and J coefficients
            )

        However, ``prep_l`` is not required to be built from the same Python function
        as ``prep_r``. Any callable is valid as long as it prepares the correct left
        PREP state :math:`P_{L}` for the chosen FOQCS-LCU representation and accepts
        the ancilla quantum variable as its only remaining argument.

    num_q_ops : int
        Number of operand qubits, i.e. ``L`` argument for FOQCS-LCU PREP routines.
        The default is 1.

    is_hermitian : bool
        Indicates whether the block-encoding unitary is Hermitian. For more information
        see ``is_hermitian`` parameter from :class:`BlockEncoding` class.
        The default is ``False``.

    norm : "ArrayLike"
        Normalization factor.
        The default is `1` in case no normalization factor is passed.

    num_q_anc : int
        Number of ancillary qubits required for the passed PREP method. (Minimum 2 * ``num_q_ops``)
        For example, :func:`foqcs_prep_heisenberg` requires 2 * ``num_q_ops`` + 6 qubits.
        This parameter is necessary for custom PREP routines whose ancilla count
        cannot be inferred automatically. For built-in FOQCS-LCU PREP routines,
        the ancilla count is predefined and this argument can be omitted.
        The default is -1.

    Returns
    -------
    BlockEncoding
        A BlockEncoding using FOQCS LCU.

    Raises
    ------
    ValueError
        When the operator is not representing spin-glass model.
    KeyError
        If method received an unsupported FOQCS-LCU PREP method

    Notes
    -----

    **State Preparation in FOQCS-LCU**

    The FOQCS-LCU block encoding unitary $U$ relies on distinct right and left state preparation subroutines, denoted as $P_R$ and $P_L$, alongside a $\text{SELECT}$ operation:

    * $\mathbf{P_R}$ **(Right State Preparation):** This subroutine prepares the state based on the target system's original coefficients, $\alpha$.
    * $\mathbf{P_L}$ **(Left State Preparation):** This subroutine acts identically to $P_R$, but operates on the complex conjugated coefficients, $\alpha^*$.

    **Note on the Adjoint:** Because $P_L$ utilizes conjugated coefficients, its adjoint $P_L^\dagger$ is in general *not* the mathematical inverse of $P_R$.

    **Summary:**

    $P_R = \text{PREP}(\alpha),\; P_L = \text{PREP}(\alpha^*)$

    $U = P_L^\dagger \cdot \text{SELECT} \cdot P_R$

    Examples
    --------

    This example constructs a FOQCS-LCU block encoding for the one-dimensional
    nearest-neighbour Heisenberg Hamiltonian

    .. math::

        H =
        \sum_i (g_X X_i + g_Y Y_i + g_Z Z_i)
        +
        \sum_i (J_X X_i X_{i+1} + J_Y Y_i Y_{i+1} + J_Z Z_i Z_{i+1}).

    The arrays ``g`` and ``J`` store the three field and coupling coefficients.
    The helper :func:`foqcs_prep_heisenberg` prepares the FOQCS-LCU state
    on the ancillary qubits that correspond to these terms.

    The coefficients ``_g`` and ``_J`` are the amplitudes used internally by the
    PREP routine. Their squared norm gives the LCU normalization factor
    :math:`\alpha`, so the resulting block encoding represents approximately
    :math:`{H} / {\alpha}`. Finally, a random input state ``psi`` is prepared on the
    operand register and :meth:`apply_rus` applies the block encoding using
    repeat-until-success sampling

    The constructor

    ::

        BlockEncoding.from_foqcs_lcu_prep(...)

    wraps these PREP routines into a :class:`BlockEncoding`. Essentially,
    the user supplies the state preparation routines, the number of
    operand qubits, and the normalization factor, while the constructor builds
    the corresponding FOQCS-LCU block-encoding structure.

    It can be applied in two common ways. The method :meth:`apply` applies the
    block encoding circuit directly, leaving the success/failure information
    in the block-encoding ancilla qubits, requiring us to explicitly filter out
    the non-zero ancillary qubits. The method :meth:`apply_rus` instead uses
    repeat-until-success sampling: it repeatedly applies the block encoding
    until the success condition is observed, and then returns the transformed
    operand register. This example uses :meth:`apply_rus` so that the sampled result
    corresponds to a successful application of the encoded operator to the random
    input state ``psi``.

    This example uses :meth:`apply_rus`:

    ::

        import numpy as np
        from qrisp import QuantumVariable, terminal_sampling
        from qrisp.block_encodings import BlockEncoding
        from qrisp.block_encodings.constructors.foqcs_lcu import foqcs_prep_heisenberg
        from functools import partial

        def _prep_psi(q_num):
            # Generate state amplitudes.
            psi = np.random.uniform(-1, 1, 2 ** (q_num)) + 1j * np.random.uniform(
                -1, 1, 2 ** (q_num)
            )
            psi /= np.linalg.norm(psi)
            return psi

        # Initialize variables + their values
        L = 4
        g = np.array(np.random.uniform(-1, 1, 3), dtype="complex")
        J = np.array(np.random.uniform(-1, 1, 3), dtype="complex")

        # Normalize
        norm = np.linalg.norm(np.block([g, J]))
        g /= norm
        J /= norm

        # Calculating the normalization factor
        _g = np.zeros((3,), dtype="complex")
        _J = np.zeros((3,), dtype="complex")

        for i in range(3):
            _g[i] = np.sqrt(g[i] * L)
            _J[i] = np.sqrt(J[i] * (L - 1))

        # Correction for XZ = -iY
        _J[1] = 1j * _J[1]
        _g[1] = (1 - 1j) * _g[1] / np.sqrt(2)

        # Normalization for block encoding
        norm = np.linalg.norm(np.block([_g, _J]))

        # Construct dictionary input expected by foqcs_prep_heisenberg()
        heis_g = {"X": g[0], "Y": g[1], "Z": g[2]}
        heis_J = {"X": J[0], "Y": J[1], "Z": J[2]}

        # Create partial PREP_R and PREP_L^dagger functions to be used by FOQCS-LCU
        prep_r = partial(
            foqcs_prep_heisenberg,
            L=L,
            g=heis_g,
            J=heis_J,
        )
        prep_l = partial(
            foqcs_prep_heisenberg,
            L=L,
            g=heis_g,
            J=heis_J,
            conjugate=True
        )

        be = BlockEncoding.from_foqcs_lcu_prep(prep_r=prep_r, prep_l=prep_l, num_q_ops=L, norm=norm ** 2)

        psi = _prep_psi(L)

        def operand_prep(psi):
            qv = QuantumVariable(4)
            qv.init_state(psi, method="qswitch")
            return qv

        @terminal_sampling
        def main_apply_rus(BE):
            return BE.apply_rus(operand_prep)(psi)

        # Do the measurement using RUS
        result_rus = main_apply_rus(be)
        print(result_rus)

    This example uses :meth:`apply`:

    ::

        import numpy as np
        from functools import partial

        from qrisp import QuantumVariable, terminal_sampling
        from qrisp.block_encodings import BlockEncoding
        from qrisp.block_encodings.constructors.foqcs_lcu import foqcs_prep_heisenberg


        def _prep_psi(q_num):
            # Generate random normalized input state amplitudes.
            psi = np.random.uniform(-1, 1, 2**q_num) + 1j * np.random.uniform(
                -1, 1, 2**q_num
            )
            psi /= np.linalg.norm(psi)
            return psi


        # Initialize variables + their values.
        L = 4
        g = np.array(np.random.uniform(-1, 1, 3), dtype="complex")
        J = np.array(np.random.uniform(-1, 1, 3), dtype="complex")

        # Normalize coefficients.
        norm = np.linalg.norm(np.block([g, J]))
        g /= norm
        J /= norm

        # Calculate the normalization factor used by the block encoding.
        _g = np.zeros((3,), dtype="complex")
        _J = np.zeros((3,), dtype="complex")

        _g[0] = np.sqrt(g[0] * L)
        _g[1] = np.sqrt(g[1] * L * -1j)
        _g[2] = np.sqrt(g[2] * L)
        _J[0] = np.sqrt(J[0] * (L - 1))
        _J[1] = np.sqrt(J[1] * -(L - 1))
        _J[2] = np.sqrt(J[2] * (L - 1))

        norm = np.linalg.norm(np.block([_g, _J]))
        _g /= norm
        _J /= norm

        # Construct dictionary input expected by foqcs_prep_heisenberg().
        heis_g = {"X": _g[0], "Y": _g[1], "Z": _g[2]}
        heis_J = {"X": _J[0], "Y": _J[1], "Z": _J[2]}

        # Create PREP_R and PREP_L^dagger functions.
        prep_r = partial(
            foqcs_prep_heisenberg,
            L=L,
            g=heis_g,
            J=heis_J,
        )

        prep_l = partial(
            foqcs_prep_heisenberg,
            L=L,
            g=heis_g,
            J=heis_J,
            conjugate=True,
        )

        # Constructing the block encoding using the :func:`from_foqcs_lcu_prep` function.
        be = BlockEncoding.from_foqcs_lcu_prep(
            prep_r=prep_r,
            prep_l=prep_l,
            num_q_ops=L,
            norm=norm**2,
        )

        # Generate the operands state
        psi = _prep_psi(L)

        # Create a quantum variable with the psi state
        def operand_prep(psi):
            qv = QuantumVariable(L)
            qv.init_state(psi, method="qswitch")
            return qv

        qv = operand_prep(psi)

        # apply() applies the block-encoding circuit directly.
        # Unlike apply_rus(), it does not repeat until the success branch occurs.

        # Since apply() only applies the block-encoding circuit once, this branch is not
        # automatically selected or renormalized as in apply_rus(). Therefore, these
        # amplitudes represent the unnormalized block-encoded action on the input state:
        ancillas = be.apply(qv)

        qc = qv.qs.compile()
        sv = qc.statevector_array()

        # The remaining amplitude of the full statevector lives in the failure branches,
        # where the ancilla register is nonzero.
        res_ops = []

        for i in range(0, 2 ** L):
            qi = int(f"{i:0{L}b}"[::-1], 2) # Reverses bits
            ind = qi << (len(ancillas[0]))
            res_ops.append(sv[ind])

        # res_ops contains the amplitudes of the operand register in the postselected
        # success branch of the block encoding, i.e. the branch where all ancillas are 0.

        #     res_ops ≈ H|psi> / alpha

        # where alpha is the LCU/block-encoding normalization factor passed as `norm`.
        print(res_ops)

    The following example shows how to define and use a custom PREP function.

    ::

        from functools import partial
        from qrisp import *
        from qrisp.block_encodings import BlockEncoding

        L = 2 # 2 operand qubits
        n_anc_custom_prep = 5 # 4 base ancillary qubits + one extra

        # Custom PREP_R subroutine.
        # Defines how many ancillary qubits the circuit requires.
        # This example does not result in a viable block encoding,
        # it only shows the process of defining a custom subroutine.
        def custom_prep(qv: QuantumVariable | Sequence[Qubit], L: int):
            # Ancilla layout for L = 2 and n_anc = 5:
            #
            #   [extra, x0, x1, z0, z1]
            #
            # SELECT will use only the final 2L qubits:
            #
            #   [x0, x1, z0, z1]
            #
            # This PREP sets extra = 1 and copies it into x0.
            # Also it sets z1 = 1.
            # Thus the selected operation should be X(0)Z(1)
            x(qv[0]) # Extra ancillary
            cx(qv[0], qv[1]) # x[0] taken from extra ancillary.
            x(qv[4]) # z[1]

        # Then, custom_prep is used for both prep_r and prep_l, as there is no
        # specific handling required. (For example, parametrised subcircuit
        # would have required conjugated parameters. See the foqcs_prep_heisenberg()
        # usage from the previous example)
        prep_r = partial(
            custom_prep,
            L=L
        )
        prep_l = partial(
            custom_prep,
            L=L
        )

        be = BlockEncoding.from_foqcs_lcu_prep(
            prep_r = prep_r,
            prep_l = prep_l,
            num_q_ops = L,
            num_q_anc = n_anc_custom_prep
        )

        qv = QuantumVariable(L)
        ancillas = be.apply(qv)

        res = multi_measurement([qv] + ancillas)

        # res contains the measurement probabilities for the operand register
        # together with the FOQCS-LCU ancilla register.
        #
        # In this example, the custom PREP activates x[0] and z[1], so SELECT applies
        # X(0)Z(1). Since the operand starts in |00>, the Z(1) part has no visible
        # phase effect and X(0) flips the first operand qubit. The inverse of prep_l 
        # then uncomputes the PREP register, so all ancillas return to |00000>.
        # Therefore, the expected measurement result is the operand state |10> and
        # the ancilla state |00000>, with probability 1: {('10', '00000'): 1.0}
        print(res)


    """
    if num_q_anc == -1:
        n_anc = get_foqcs_lcu_prep_num_of_ancillae(prep_r, num_q_ops)
    elif num_q_anc >= num_q_ops * 2:
        n_anc = num_q_anc
    else:
        raise ValueError(f"FOQCS-LCU requires at least 2L ancillary qubits."
                         f" Expected at least {num_q_ops * 2}, but received {num_q_anc}.")

    # FOQCS-LCU SELECT
    def _select(num_q_ops: int, n_anc: int, ancillae, *operands):
        extra_anc = n_anc - num_q_ops * 2
        cx(ancillae[extra_anc:extra_anc + num_q_ops], operands[0])
        cz(ancillae[extra_anc + num_q_ops:], operands[0])

    @qache
    def unitary(*args):
        # LCU = PREP_R SELECT PREP_L^dg (note: PREP(a)^dg != PREP(a*)^dg, where PREP(a) = PREP_R, and PREP(a*) = PREP_L)
        prep_r(args[0])
        _select(num_q_ops, n_anc, args[0], *args[1:])
        with invert():
            prep_l(args[0])

    return cls(
        norm,
        [QuantumVariable(n_anc).template()],
        unitary,
        num_ops=1,
        is_hermitian=is_hermitian,
    )
