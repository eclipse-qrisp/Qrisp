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

import jax.numpy as jnp

from qrisp.alg_primitives.arithmetic.adders.gidney.cq_gidney_adder import *
from qrisp.alg_primitives.arithmetic.adders.gidney.qq_gidney_adder import *
from qrisp.alg_primitives.arithmetic.adders.adder_tools import ammend_inpl_adder
from qrisp.environments import custom_control
from qrisp.core import QuantumVariable

from qrisp.jasp import check_for_tracing_mode, DynamicQubitArray


def gidney_adder(a, b, c_in=None, c_out=None):
    """
    In-place adder function based on `this paper <https://arxiv.org/abs/1709.06648>`__.
    Performs the addition:

    ::

        b += a


    Parameters
    ----------
    a : int or QuantumVariable or list[Qubit]
        The value that should be added.
    b : QuantumVariable or list[Qubit]
        The value that should be modified in the in-place addition.
    c_in : Qubit, optional
        An optional carry in value. The default is None.
    c_out : Qubit, optional
        An optional carry out value. The default is None.

    Examples
    --------

    We add two integers:

    >>> from qrisp import QuantumFloat, gidney_adder
    >>> a = QuantumFloat(4)
    >>> b = QuantumFloat(4)
    >>> a[:] = 4
    >>> b[:] = 5
    >>> gidney_adder(a,b)
    >>> print(b)
    {9: 1.0}

    """
    if check_for_tracing_mode():
        from qrisp.alg_primitives.arithmetic.jasp_arithmetic import (
            jasp_qq_gidney_adder,
            jasp_cq_gidney_adder,
        )

        if isinstance(a, (QuantumVariable, DynamicQubitArray)):
            return jasp_qq_gidney_adder(a, b)
        else:
            return jasp_cq_gidney_adder(jnp.array(a, dtype="int32"), b)

    if isinstance(a, (int, str)):
        return custom_control(cq_gidney_adder)(a, b, c_in=c_in, c_out=c_out)
    else:
        return qq_gidney_adder(a, b, c_in=c_in, c_out=c_out)


# temp = gidney_adder.__doc__
# gidney_adder = ammend_inpl_adder(gidney_adder, ammend_cl_int=False)
# gidney_adder.__doc__ = temp
