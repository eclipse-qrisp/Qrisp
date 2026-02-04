"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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

from qrisp import *
from qrisp.alg_primitives.arithmetic.adders.cdkpm_adder import cdkpm_adder

def inpl_add(
    qf1,
    qf2,
    adder=None,
    ctrl=None,
    ignore_rounding_error=False,
    ignore_overflow_error=False,
):
    """
    Performs in-place addition of the second argument onto the first.
    In Python syntax: ::

        qf1 += qf2
    
    Currently, the Cuccaro adder as introduced in `this paper <https://arxiv.org/abs/quant-ph/0410184>`_
    is supported in both static and dynamic modes. 

    
    Parameters
    ----------
    qf1 : QuantumFloat
        The QuantumFloat that will be in-place modified.
    qf2 : QuantumFloat
        The QuantumFloat that is being added.

    adder : str, optional
        Specifies the adder. Available option is "cuccaro" (also the default).

    Raises
    ------
    Exception
        Tried to add signed QuantumFloat onto non signed QuantumFloat.

    Examples
    --------

    We create two QuantumFloats and apply the inplace adder

    >>> from qrisp import QuantumFloat, inpl_add
    >>> qf_0 = QuantumFloat(5)
    >>> qf_1 = QuantumFloat(5)
    >>> qf_0[:] = 4
    >>> qf_1[:] = 3
    >>> inpl_add(qf_0, qf_1)
    >>> print(qf_0)
    {7.0: 1.0}
    """
    if adder is not None and adder != "cuccaro":
        raise NotImplementedError(f"Adder {adder} not implemented for the inpl_add function.")
    
    if adder == "cuccaro" or adder is None:
        cdkpm_adder(qf1, qf2, ctrl=ctrl) 