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

from qrisp import h, IQAE, cx, x, z, auto_uncompute, QuantumBool, control

def uniform(*args):
    for arg in args:
        h(arg)


def QMCI(qargs, function, distribution=None):
    r"""
    Implements a general algorithm for `Quantum Monte Carlo Integration <https://www.nature.com/articles/s41598-024-61010-9>`_.
    This implementation utilizes :ref:`IQAE`. A detailed explanation can be found in the :ref:`tutorial <QMCItutorial>`.

    QMCI performs numerical integration of (high-dimensional) functions w.r.t. probability distributions:

    .. math::

        \int_{[0,1]^n} f(x_1 , ... , x_n) \mathrm{d}\mu (x_1 , ... , x_n)

    Parameters
    ----------
    qargs : list[:ref:`QuantumFloat`]
        The quantum variables representing the $x$-axes (the variables the given ``function`` acts on), and a quantum variable representing the $y$-axis.
    function : function
        A Python function which takes :ref:`QuantumFloats <QuantumFloat>` as inputs, 
        and returns a :ref:`QuantumFloat` containing the values of the integrand.
    distribution : function
        A Python function which takes :ref:`QuantumFloats <QuantumFloat>` as inputs and applies the distribution over which to integrate.
        By default, the uniform distribution is applied.

    Returns
    -------
    float
        The result of the numerical integration.

    Examples
    --------

    We integrate the function $f(x)=x^2$ over the integral $[0,1]$.
    Therefore, the function is evaluated at $8=2^3$ sampling points as specified by ``QuantumFloat(3,-3)``.
    The $y$-axis is representend by ``QuantumFloat(6,-6)``.

    ::

        from qrisp import QuantumFloat
        from qrisp.algorithms.qmci import QMCI

        def f(qf):
            return qf*qf

        qf_x = QuantumFloat(3,-3)
        qf_y = QuantumFloat(6,-6)
        QMCI([qf_x,qf_y], f)
        # Yields: 0.27373180511103606

    This result is consistent with numerically calculating the integral by evaluating the function $f$ at 8 sampling points:

    ::

        N = 8
        sum((i/N)**2 for i in range(N))/N
        # Yields: 0.2734375

    A detailed explanation of QMCI and its implementation in Qrisp can be found in the :ref:`QMCI tutorial <QMCItutorial>`.

    """
    if distribution==None:
        distribution = uniform

    #dupl_args = [arg.duplicate() for arg in qargs]
    #dupl_res_qf = function(*dupl_args)
    #qargs.append(dupl_res_qf.duplicate())

    #for arg in dupl_args:
    #    arg.delete()
    #dupl_res_qf.delete()

    V0=1
    for arg in qargs:
        V0 *= 2**(arg.size+arg.exponent)

    qargs.append(QuantumBool())

    @auto_uncompute
    def state_function(*args):
        qf_x = args[:-2]
        qf_y = args[-2]
        tar = args[-1]

        distribution(*qf_x)
        h(qf_y)

        with(qf_y < function(*qf_x)):
            x(tar)

    a = IQAE(qargs, state_function, eps=0.01, alpha=0.01)   

    V = V0*a
    return V

