"""
\********************************************************************************
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
********************************************************************************/
"""

from qrisp.core import h
from qrisp.alg_primitives import QPE


def quantum_counting(qv, oracle, precision):
    """
    This algorithm estimates the amount of solutions for a given Grover oracle.

    Parameters
    ----------
    qv : QuantumVariable
        The QuantumVariable on which to evaluate.
    oracle : function
        The oracle function.
    precision : int
        The precision to perform the quantum phase estimation with.

    Returns
    -------
    M : float
        An estimate of the amount of solutions.

    Examples
    --------

    We create an oracle, which performs a simple phase flip on the last qubit. ::

        from qrisp import quantum_counting, z, QuantumVariable

        def oracle(qv):
            z(qv[-1])


    We expect half of the state-space of the input to be a solution.

    For 3 qubits, the state space is $2^3 = 8$ dimensional.

    >>> quantum_counting(QuantumVariable(3), oracle, 3)
    3.999999999999999

    For 4 qubits, the state space is $2^4 = 16$ dimensional.

    >>> quantum_counting(QuantumVariable(4), oracle, 3)
    7.999999999999998


    """

    from qrisp import gate_wrap, measure
    from qrisp.grover import diffuser
    from qrisp.jasp import check_for_tracing_mode

    @gate_wrap
    def grover_operator(qv):
        oracle(qv)
        diffuser(qv)

    h(qv)
    res = QPE(qv, grover_operator, precision=precision)

    if check_for_tracing_mode():
        mes_res = measure(res)
        import jax.numpy as jnp
    else:
        mes_res = list(res.get_measurement().keys())[0]
        import numpy as jnp

    theta = mes_res * jnp.pi

    N = 2**qv.size
    M = N * jnp.sin(theta) ** 2

    return M
