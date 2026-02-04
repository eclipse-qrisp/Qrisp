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

import numpy as np
from jax.errors import TracerArrayConversionError

from qrisp.alg_primitives.state_preparation.qiskit_state_preparation import (
    prepare_qiskit,
)
from qrisp.alg_primitives.state_preparation.qswitch_state_preparation import (
    prepare_qswitch,
)
from qrisp.jasp.tracing_logic import check_for_tracing_mode


def prepare(qv, target_array, reversed: bool = False, method: str = "auto"):
    r"""
    Prepare a quantum state on ``qv`` from a target amplitude vector.

    Given a vector :math:`b=(b_0,\dotsc,b_{N-1})` (corresponding to ``target_array``),
    this routine prepares a state proportional to

    .. math::

        \sum_{i=0}^{N-1} b_i \ket{i}.

    The ``target_array`` is normalized internally, i.e.

    .. math::

        \tilde b_i = \frac{b_i}{\|b\|},
        \qquad
        \ket{0} \;\mapsto\; \sum_{i=0}^{N-1} \tilde b_i \ket{i}.

    By default, the little-endian convention is used for the computational basis ordering.
    For example, for a 2-qubit system, the basis states are ordered as

    .. math::

        \ket{0} \equiv \ket{q_0 = 0, q_1 = 0}, \quad \ket{1} \equiv \ket{q_0 = 1, q_1 = 0}, \dotsc

    Parameters
    ----------
    qv : QuantumVariable
        Quantum variable to prepare.

    target_array : numpy.ndarray or jax.numpy.ndarray
        Target amplitude vector :math:`b`. Must have length :math:`2^n` where
        :math:`n` is the size of ``qv`` (validated for concrete arrays).

    reversed : bool, optional
        If ``True``, applies a bit-reversal permutation (big-endian ordering).
        Default is ``False``.

    method : {'qiskit', 'qswitch', 'auto'}, optional
        Compilation method for state preparation. Possible values are:

        - ``'qiskit'``: requires concrete arrays (e.g. NumPy).
        - ``'qswitch'``: supports traced arrays (e.g. JAX tracers in Jasp mode).
          Note that shape validation is not performed in Jasp mode.
        - ``'auto'``: automatically selects between the above.

        Default is ``'auto'``.

    Examples
    --------

    In this example, we create a :ref:`QuantumFloat` and prepare the normalized state
    $\sum_{i=0}^3 \tilde b_i\ket{i}$ for $\tilde b=(0,1,2,3)/\sqrt{14}$.

    ::

        import numpy as np
        from qrisp import QuantumFloat, prepare

        b = np.array([0, 1, 2, 3], dtype=float)
        b /= np.linalg.norm(b)

        qf = QuantumFloat(2)
        prepare(qf, b)

    We can verify that the state has been correctly prepared.

    For example, we can use the ``statevector`` method to get a function that maps basis states to amplitudes:

    ::

        sv_function = qf.qs.statevector("function")

        print(f"b[1]: {b[1]:.6f} -> {sv_function({qf: 1}):.6f}")
        # b[1]: 0.267261 -> 0.267261-0.000000j
        print(f"b[2]: {b[2]:.6f} -> {sv_function({qf: 2}):.6f}")
        # b[2]: 0.534522 -> 0.534522-0.000000j

    where index 1 in little-endian corresponds to the basis state :math:`\ket{q_0=1, q_1=0}`
    and index 2 to :math:`\ket{q_0=0, q_1=1}`.  With ``reversed=True``, we can
    switch to big-endian ordering. That is, we can map ``b[1]`` to ``sv_function({qf: 2})``
    instead, and so on.

    We can perform a similar verification even if the statevector is not directly
    accessible (for example when running on hardware), by using measurement results:

    ::

        qf = QuantumFloat(2)
        prepare(qf, b)

        res_dict = qf.get_measurement()

        ref = np.sqrt(res_dict[1])
        amps = {k: round(np.sqrt(v) / ref) for k, v in res_dict.items()}

        print(amps)
        # Yields: {3: 3, 2: 2, 1: 1}

    The output indicates that the magnitudes of the amplitudes for the basis states
    :math:`\ket{1}`, :math:`\ket{2}`, and :math:`\ket{3}` are in the ratio :math:`1 : 2 : 3`,
    exactly matching the input vector :math:`b = (0,1,2,3)` up to normalization.

    """

    if method not in {"auto", "qiskit", "qswitch"}:
        raise ValueError("method must be 'auto', 'qiskit', or 'qswitch'")

    is_tracing = check_for_tracing_mode()

    if not is_tracing:
        expected = 1 << qv.size
        if target_array.size != expected:
            raise ValueError(
                f"Statevector length must be {expected} for {qv.size} qubits, "
                f"got {target_array.size}."
            )
        target_array = np.asarray(target_array)
        norm = np.linalg.norm(target_array)
        if np.isclose(norm, 0.0):
            raise ValueError("The provided statevector has zero norm.")
        target_array = target_array / norm

    if method == "auto":
        try:
            target_array = np.asarray(target_array)
            method = "qiskit"
        except TracerArrayConversionError:
            method = "qswitch"

    if method == "qiskit":
        prepare_qiskit(qv, target_array, reversed)
    else:
        prepare_qswitch(qv, target_array, reversed)
