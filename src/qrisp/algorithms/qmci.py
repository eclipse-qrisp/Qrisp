"""
\********************************************************************************
* Copyright (c) 2024 the Qrisp authors
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

from qrisp import h, IQAE, cx, z, auto_uncompute, QuantumBool

def uniform(*args):
    for arg in args:
        h(arg)


def QMCI(qargs, function, distribution=None):
    """
    Implements a general algorithms for `Quantum Monte Carlo Integration <https://www.nature.com/articles/s41598-024-61010-9>`.

    Parameters
    ----------
    qargs : list[ref:`QuantumFloat`]
        The quantum variables the given ``function`` acts on.
    function : function
        A Python function which takes :ref:`QuantumFloats <QuantumFloat>` as inputs and applies the ``function`` which is to be integrated.
    distribution : function
        A Python function which takes :ref:`QuantumFloats <QuantumFloat>` as inputs and applies the distribution over which to integrate.

    Returns
    -------
    float
        The result of the integration.

    Examples
    --------

    We integrate the function $f(x)=x^2$ over the integral $[0,1]$.
    Therefore the function is evaluated at $8=2^3$ points as specified by ``QuantumFloat(3,-3)``.

    ::

        from qrisp import QuantumFloat
        from qrisp.algorithms.qmci import *

        def f(qf):
            return qf*qf

        qf = QuantumFloat(3,-3)
        QMCI([qf], f)

    """
    if distribution==None:
        distribution = uniform

    dupl_args = [arg.duplicate() for arg in qargs]
    dupl_res_qf = function(*dupl_args)
    qargs.append(dupl_res_qf.duplicate())

    for arg in dupl_args:
        arg.delete()
    dupl_res_qf.delete()

    V0=1
    for arg in qargs:
        V0 *= 2**(arg.size+arg.exponent)
    qargs.append(QuantumBool())

    @auto_uncompute
    def state_function(*args):
        qf_x = args[0]
        qf_y = args[1]
        tar = args[2]

        distribution(qf_x)
        h(qf_y)
        qbl = (qf_y < function(qf_x))
        cx(qbl,tar)

    def oracle_function(*args):  
        tar = args[2]
        z(tar)

    a = IQAE(qargs, state_function, oracle_function, eps=0.01, alpha=0.01)   

    V = V0*a
    return V