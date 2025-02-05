"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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

from qrisp.core.gate_application_functions import h
from qrisp.alg_primitives.qft import QFT
from qrisp.jasp import jrange


def QPE(
    args, U, precision=None, target=None, iter_spec=False, ctrl_method=None, kwargs={}
):
    r"""
    Evaluates the `quantum phase estimation algorithm
    <https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm>`_.

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
    precision : int, optional
        The precision of the estimation. The default is None.
    target : QuantumFloat, optional
        A target QuantumFloat to perform the estimation into. The default is None.
        If given neither a precision nor a target, an Exception will be raised.
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

    Raises
    ------
    Exception
        Tried to perform quantum phase estimation without precision specification.

    Returns
    -------
    res : QuantumFloat
        The QuantumFloat containing the estimated phase as a fraction of $2 \pi$.

    Examples
    --------

    We define a function that applies two phase gates onto its input and estimate the
    applied phase. ::

        from qrisp import p, QuantumVariable, QPE, multi_measurement

        def U(qv):
            x = 0.5
            y = 0.125

            p(x*2*np.pi, qv[0])
            p(y*2*np.pi, qv[1])

        qv = QuantumVariable(2)

        h(qv)

        res = QPE(qv, U, precision = 3)

    >>> multi_measurement([qv, res])
    {('00', 0.0): 0.25,
     ('10', 0.5): 0.25,
     ('01', 0.125): 0.25,
     ('11', 0.625): 0.25}
    >>> res.qs.depth()
    66

    During the phase estimation, ``U`` is called $2^{\text{precision}}$ times. We can
    reduce that number by abusing that we can bundle repeated calls into a single call
    with a modified phase. ::

        def U(qv, iter = None):
            x = 0.5
            y = 0.125

            p(x*2*np.pi*iter, qv[0])
            p(y*2*np.pi*iter, qv[1])

        qv = QuantumVariable(2)

        h(qv)

        res = QPE(qv, U, precision = 3, iter_spec = True)

    >>> multi_measurement([qv, res])
    {('00', 0.0): 0.25,
     ('10', 0.5): 0.25,
     ('01', 0.125): 0.25,
     ('11', 0.625): 0.25}
    >>> res.qs.depth()
    34

    """

    from qrisp import QuantumFloat, control

    if target is None:
        if precision is None:
            raise Exception(
                "Tried to perform quantum phase estimation without"
                "precision specification"
            )
        qpe_res = QuantumFloat(precision, -precision, signed=False)
    else:
        qpe_res = target

    h(qpe_res)

    for i in jrange(qpe_res.size):
        if iter_spec:
            with control(qpe_res[i], ctrl_method=ctrl_method):
                U(args, iter=2**i, **kwargs)
        else:
            with control(qpe_res[i], ctrl_method=ctrl_method):
                for j in jrange(2**i):
                    U(args, **kwargs)

    QFT(qpe_res, inv=True)

    return qpe_res

