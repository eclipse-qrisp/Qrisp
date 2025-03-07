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

from qrisp import h, control, rz, measure, reset, QuantumBool, Clbit
from qrisp.jasp import jrange
import numpy as np


def IQPE(args, U, precision, iter_spec=False, ctrl_method=None, kwargs={}):
    r"""
    Evaluates the `iterative quantum phase estimation algorithm
    <https://arxiv.org/pdf/quant-ph/0610214>`_.

    The unitary to estimate is expected to be given as Python function, which is called
    on ``args``.

    Parameters
    ----------
    args : list
        A list of arguments (could be QuantumVariables) which represent the state,
        the quantum phase estimation is performed on.
    U : function
        A Python function, which will receive the list ``args`` as arguments in the
        course of this algorithm.
    precision : int
        The precision of the estimation.
    iter_spec : bool, optional
        If set to ``True``, ``U`` will be called with the additional keyword
        ``iter = i`` where ``i`` is the amount of iterations to perform (instead of
        simply calling ``U`` for ``i`` times). The default is False.
    ctrl_method : string, optional
        Allows to specify which method should be used to generate the
        controlled U circuit. For more information check
        :meth:`.control <qrisp.Operation.control>`. The default is None.
    kwargs : dict, optional
        A dictionary of keyword arguments to pass to ``U``. The default is {}.

    Returns
    -------
    res : list[list[ClBits]]
        A list of classical bits containing the result.

    Examples
    --------

    We define a function that applies two rotations onto its input and estimate the
    applied phase. ::

        from qrisp import p, QuantumVariable, IQPE, multi_measurement, h, run, x, rx, QuantumFloat
        import numpy as np

        def f():
            def U(qv):
                x = 1/2**3
                y = 1/2**2

                rx(x*2*np.pi, qv[0])
                rx(y*2*np.pi, qv[1])

            qv = QuantumFloat(2)

            x(qv)
            h(qv)

            return IQPE(qv, U, precision = 4)
        jaspr = make_jaspr(f)()

    >>> jaspr()
    (Array(False, dtype=bool),
    Array(True, dtype=bool),
    Array(True, dtype=bool),
    Array(False, dtype=bool))
    >>> IQPE_binary_decimals_to_float(jaspr())
    0.375

    """
    iqpe_aux = QuantumBool()
    clbits: Clbit = []
    for k in range(precision):
        reset(iqpe_aux)
        h(iqpe_aux)
        if iter_spec:
            with control(iqpe_aux, ctrl_method=ctrl_method):
                U(args, iter=2**(precision-k), **kwargs)
        else:
            with control(iqpe_aux[0]):
                for j in jrange(2**(precision-k)):
                    U(args, **kwargs)
        for i in range(k):
            with control(clbits[i] == 1):
                rz(-np.pi/2**(i+1), iqpe_aux)
        h(iqpe_aux)
        clbits.insert(0, measure(iqpe_aux))
    return clbits

def IQPE_binary_decimals_to_float(bits):
    theta = 0
    for i in range(len(bits)):
        theta += int(bits[i].tolist())/2**(i+1)
    return theta