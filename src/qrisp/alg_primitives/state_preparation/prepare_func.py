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

from qrisp.alg_primitives.state_preparation.qswitch_state_preparation import prepare_qswitch
from qrisp.alg_primitives.state_preparation.qiskit_state_preparation import prepare_qiskit

def prepare(qv, target_array, reversed=False, method = "auto"):
    r"""
    This method performs quantum state preparation. Given a vector $b=(b_0,\dotsc,b_{N-1})$, the function acts as

    .. math::

        \ket{0} \rightarrow \sum_{i=0}^{N-1}b_i\ket{i}

    Parameters
    ----------
    qv : QuantumVariable
        The quantum variable on which to apply state preparation.
    target_array : numpy.ndarray
        The vector $b$.
    reversed : boolean
        If set to ``True``, the endianness is reversed. The default is ``False``.
    method : str, optional
        String to specify the compilation method. Available are ``qiskit``, ``qswitch`` and ``auto``.
        ``qiskit`` is more gate-efficient but ``qswitch`` can also process dynamic arrays.
        The default is ``auto``.

    Examples
    --------

    We create a :ref:`QuantumFloat` and prepare the state $\sum_{i=0}^3b_i\ket{i}$ for $b=(0,1,2,3)$.

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
    
    from qrisp.jasp import check_for_tracing_mode
    from qrisp.misc import check_if_fresh
    
    tracing = int(check_for_tracing_mode())
    
    if not tracing:
        expected = 1 << qv.size
        if target_array.size != expected:
            raise ValueError(
                f"Statevector length must be {expected} for {qv.size} qubits, "
                f"got {target_array.size}."
            )
        norm = np.linalg.norm(np.asarray(target_array))
        if np.isclose(norm, 0.0):
            raise ValueError("The provided statevector has zero norm.")
        #if not check_if_fresh(qv.reg, qv.qs):
        #    raise ValueError(
        #        "Tried to initialize qubits which are not fresh anymore."
        #    )
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
        if not reversed:
            raise Exception("Reversed state preparation is currently not available for method qswitch")
        prepare_qswitch(qv, target_array)
    else:
        raise ValueError("method must be 'auto', 'qiskit', or 'qswitch'")
