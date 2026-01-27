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
    Performs quantum state preparation on a quantum variable.

    Given a vector :math:`b=(b_0,\dotsc,b_{N-1})` with :math:`b \neq 0`, this
    function prepares a state proportional to

    .. math::

        \sum_{i=0}^{N-1} b_i \ket{i}.

    If ``target_array`` is a concrete array (non-Jasp mode), it is normalized internally, i.e.

    .. math::

        \tilde b_i = \frac{b_i}{\|b\|},
        \qquad
        \ket{0} \;\mapsto\; \sum_{i=0}^{N-1} \tilde b_i \ket{i}.

    Parameters
    ----------

    qv : QuantumVariable
        The quantum variable on which to apply state preparation.
    target_array : numpy.ndarray or jax.numpy.ndarray
        Target amplitude vector :math:`b`. Must have length :math:`2^n` where
        :math:`n` is the size of ``qv`` (validated for concrete arrays).
    reversed : bool, optional
        If ``True``, applies a bit-reversal permutation to the computational
        basis ordering (endianness). Equivalently, amplitudes are remapped so
        that index ``i`` is interpreted with reversed bit order. Default is
        ``False``.
    method : {'qiskit', 'qswitch', 'auto'}, optional
        Compilation method for state preparation.

        - ``'qiskit'``: requires concrete arrays (e.g. NumPy).
        - ``'qswitch'``: supports traced arrays (e.g. JAX tracers). Note that
          input validation/normalization may be deferred.
        - ``'auto'``: automatically selects between the above.

        Default is ``'auto'``.

    Examples
    --------

    In the following example, we create a :ref:`QuantumFloat` and prepare the state $\sum_{i=0}^3b_i\ket{i}$ for $b=(0,1,2,3)$.

    ::

        b = np.array([0,1,2,3])

        qf = QuantumFloat(2)
        prepare(qf, b)

        res_dict = qf.get_measurement()

        for k, v in res_dict.items():
            res_dict[k] = v**0.5

        for k, v in res_dict.items():
            res_dict[k] = v/res_dict[1.0]

        print(res_dict)
        # Yields: {3: 2.9999766670425863, 2: 1.999965000393743, 1: 1.0}

    """

    is_tracing = check_for_tracing_mode()

    if not is_tracing:
        expected = 1 << qv.size
        if target_array.size != expected:
            raise ValueError(
                f"Statevector length must be {expected} for {qv.size} qubits, "
                f"got {target_array.size}."
            )
        norm = np.linalg.norm(np.asarray(target_array))
        if np.isclose(norm, 0.0):
            raise ValueError("The provided statevector has zero norm.")

        target_array = np.asarray(target_array) / norm

    if method == "auto":
        try:
            target_array = np.array(target_array)
            method = "qiskit"
        except TracerArrayConversionError:
            method = "qswitch"

    if method == "qiskit":
        prepare_qiskit(qv, target_array, reversed)

    elif method == "qswitch":
        if reversed:
            raise NotImplementedError(
                "Reversed state preparation is currently not available for method qswitch"
            )
        prepare_qswitch(qv, target_array)
    else:
        raise ValueError("method must be 'auto', 'qiskit', or 'qswitch'")
