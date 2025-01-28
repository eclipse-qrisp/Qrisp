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

from qrisp import QuantumFloat
import numpy as np

def prepare(array, reversed=False):
    r"""
    This method returns a function that performs quantum state preparation. Given a vector $b=(b_0,\dotsc,b_{N-1})$, the returned function acts as 

    .. math:: 

        \ket{0} \rightarrow \sum_{i=0}^{N-1}b_i\ket{i}

    Parameters
    ----------
    array : numpy.ndarray
        The vector $b$.
    reversed : boolean
        If set to ``True``, the endianness is reversed. The default is ``False``.

    Returns
    -------
    function
        A Python function that takes a :ref:`QuantumVariable` as argument and performs quantum state preparation.

    Examples
    --------

    We create a :ref:`QuantumFloat` and prepare the state $\sum_{i=0}^3b_i\ket{i}$ for $b=(0,1,2,3)$.

    ::

        from qrisp import *
        import numpy as np

        b = np.array([0,1,2,3])
        prep_b = prepare(b)

        qf = QuantumFloat(2)
        prep_b(qf)

        res_dict = qf.get_measurement()

        for k, v in res_dict.items():
            res_dict[k] = v**0.5

        for k, v in res_dict.items():
            res_dict[k] = v/res_dict[1.0]

        print(res_dict)
        # Yields: {3: 2.9999766670425863, 2: 1.999965000393743, 1: 1.0}

    """

    n = len(array)
    m = int(np.ceil(np.log2(n)))

    qf = QuantumFloat(m)
    qf.init_state({i : array[i] for i in range(n)})

    if reversed:
        qf.reg = qf.reg[::-1]

    qc = qf.qs.compile()
    op = qc.to_gate()

    def prepare_fun(qf):
        qf.qs.append(op,[qf[i] for i in range(m)])

    return prepare_fun