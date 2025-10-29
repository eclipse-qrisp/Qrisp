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

from qrisp.alg_primitives.arithmetic.adders.qcla.classical_quantum.cq_carry_path import *
from qrisp.alg_primitives.arithmetic.adders.qcla.classical_quantum.cq_qcla_adder import *
from qrisp.alg_primitives.arithmetic.adders.qcla.classical_quantum.cq_sum_path import *
from qrisp.alg_primitives.arithmetic.adders.qcla.quantum_quantum.qq_carry_path import *
from qrisp.alg_primitives.arithmetic.adders.qcla.quantum_quantum.qq_qcla_adder import *
from qrisp.alg_primitives.arithmetic.adders.qcla.quantum_quantum.qq_sum_path import *


def qcla(a, b, radix_base=2, radix_exponent=1, t_depth_reduction=True, ctrl=None):
    r"""
    Implementation of the higher radix quantum carry lookahead adder (QCLA) as described
    `here <https://arxiv.org/abs/2304.02921>`__. This adder stands out for having logarithmic
    T-depth like `Drapers QCLA <https://arxiv.org/abs/quant-ph/0406142>`_. Compared to Drapers
    QCLA, the higher radix QCLA allows a more dynamic structure and the use of customizable
    "sub-adders" which enables this adder to beat Drapers QCLA in terms of speed (ie. T-depth).

    In Python syntax, this function performs the inplace addition:

    ::

        b += a


    Apart from the two quantum arguments, this function supports the specification of the
    adder-radix R. The adder-radix can be specified in the form of an exponential of integers:

    .. math::

        R = r^k

    Where $r$ is the radix base and $k$ is the radix exponent.

    Calling ``qcla`` with radix base $r$ and radix exponent $k$, will precalculate the
    carry values using the `Brent-Kung tree <https://en.wikipedia.org/wiki/Brent%E2%80%93Kung_adder>`_
    with carry-radix $r$ and cancel the recursion $k$ layers before conclusion.

    An additional compilation option is given with the ``t_depth_reduction`` keyword.
    This compilation option modifies the way the carry values are uncomputed.
    If ``t_depth_reduction`` is set to ``True`` the carry values will be uncomputed using
    the intermediate result of the sub-adder - if set to ``False`` they will be uncomputed
    using the automatic uncomputation algorithm.

    The advantage of the automated version is that, both T-depth and CNOT-depth are scaling
    with the the logarithm of the input size. For ``t_depth_reduction = True`` the T-depth
    is significantly reduced (and still logarithmic) however the CNOT depth becomes linear.


    Parameters
    ----------
    a : QuantumFloat or List[Qubit] or int
        The value that is added.
    b : QuantumFloat or List[Qubit]
        The value that is operated on.
    radix_base : integer, optional
        The radix of the Brent-Kung tree. The default is 2.
    radix_exponent : integer, optional
        The cancellation threshold for the Brent-Kung recursion. The adder-Radix is then
        $R = r^k$. The default is 1.
    t_depth_reduction : bool, optional
        A compilation option that reduces T-depth but in turn weakens CNOT depth
        to linear scaling. The default is True.

    Raises
    ------
    Exception
        Tried to add QuantumFloat of higher precision onto QuantumFloat of lower precision.

    Examples
    --------

    We try out several constellations of parameters:

    >>> from qrisp import QuantumFloat, qcla
    >>> a = QuantumFloat(8)
    >>> b = QuantumFloat(8)
    >>> a[:] = 4
    >>> b[:] = 15
    >>> qcla(a, b)
    >>> print(b)
    {19: 1.0}

    We now measure the T-depth. To get the optimal result, we need to tell the compiler
    that we only care about T-gates. This can be achieved with the ``gate_speed`` keyword
    of the :meth:`compile <qrisp.QuantumSession.compile>` method. This keyword allows
    you to specify a function of :ref:`Operation` objects, which returns the speed
    of that Operation. For more information check out the
    :meth:`compile <qrisp.QuantumSession.compile>` documentation page.

    For T-depth, there is already a pre-coded function: :meth:`T-depth <qrisp.t_depth_indicator>`.

    >>> from qrisp import t_depth_indicator
    >>> gate_speed = lambda x : t_depth_indicator(x, epsilon = 2**-10)
    >>> qc = b.qs.compile(gate_speed = gate_speed, compile_mcm = True)
    >>> qc.t_depth()
    17

    This function contains many allocations/deallocations that can be leveraged into
    parallelism, implying it can profit a lot from additional workspace:

    >>> qc = b.qs.compile(workspace = 10, gate_speed = gate_speed, compile_mcm = True)
    >>> qc.t_depth()
    7

    We can verify the logarithmic behavior by comparing to the
    `Gidney-adder <https://arxiv.org/abs/1709.06648>`_:

    >>> from qrisp import gidney_adder
    >>> a = QuantumFloat(40)
    >>> b = QuantumFloat(40)
    >>> gidney_adder(a, b)
    >>> qc = b.qs.compile(gate_speed = gate_speed, compile_mcm = True)
    >>> qc.t_depth()
    40

    >>> a = QuantumFloat(40)
    >>> b = QuantumFloat(40)
    >>> qcla(a, b)
    >>> qc = b.qs.compile(workspace = 50, gate_speed = gate_speed, compile_mcm = True)
    >>> qc.t_depth()
    19

    The function can also be used to perform semi-classical in-place addition

    >>> b = QuantumFloat(10)
    >>> b[:] = 20
    >>> qcla(22, b)
    >>> print(b)
    {42: 1.0}
    """

    if isinstance(a, (int, str)):
        return cq_qcla(
            a,
            b,
            radix_base=radix_base,
            radix_exponent=radix_exponent,
            t_depth_reduction=t_depth_reduction,
            ctrl=ctrl,
        )
    elif isinstance(a, (list, QuantumVariable)):
        return qq_qcla(
            a,
            b,
            radix_base=radix_base,
            radix_exponent=radix_exponent,
            t_depth_reduction=t_depth_reduction,
        )
    else:
        raise Exception(f"Don't know how to handle type {type(a)} for QCLA addition")
