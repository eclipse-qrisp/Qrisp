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
from qrisp import (
    QuantumVariable,
    QuantumBool,
    conjugate,
    p,
)
from qrisp.alg_primitives import QFT
from qrisp.algorithms.gqsp.gqsp import GQSP
from qrisp.jasp import qache, jrange
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax.typing import ArrayLike


# https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.020368
def convolve(qarg: QuantumVariable, weights: "ArrayLike") -> QuantumBool:
    r"""
    Performs cyclic convolution of a quantum state with a filter.

    Given $\ket{\psi}=\sum_{j=0}^{N-1}x_j\ket{j}$ and a filter $f=\{a_k\}_{k=-d}^d$
    the cyclic convolution of $\ket{\psi}$ with $f$ is

    .. math::

        \ket{\psi \star f} = \sum_{m=0}^{N-1}(\psi \star f)_m\ket{m}

    where

    .. math::

        (\psi \star f)_m = \sum_{k=-d}^d a_k x_{[m-k \mod N]}


    Parameters
    ----------
    qarg : QuantumVariable
        The input state.
    weights : ArrayLike
        1-D array of weights with shape ``(2d+1,)``.

    Returns
    -------
    QuantumBool
        Auxiliary variable after applying the GQSP protocol. 
        Must be measuered in state $\ket{0}$ for the GQSP protocol to be successful.


    Examples
    --------

    ::

        import numpy as np
        from qrisp import *
        from qrisp.gqsp import convolve
        from scipy.ndimage import convolve as sp_convolve

        # A simple square wave signal
        psi = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        # A simple 3-point smoothing filter {a_{-1}, a_0, a_1} = {0.25, 0.5, 0.25}
        f = np.array([0.25, 0.5, 0.25])

        # Mode 'wrap' performs cyclic convolution
        convolved_signal_target = sp_convolve(psi, f, mode='wrap')
        print("target:", convolved_signal_target)
        # target: [0.75 1.   1.   0.75 0.25 0.   0.   0.25]

        def psi_prep():
            qv = QuantumFloat(3)
            prepare(qv, psi)
            return qv

        # Converts the function to be executed within a 
        # repeat-until-success (RUS) procedure.
        @RUS
        def conv_psi_prep():
            qarg = psi_prep()
            qbl = convolve(qarg, f)
            success_bool = measure(qbl) == 0
            reset(qbl)
            qbl.delete()
            return success_bool, qarg

        # The terminal_sampling decorator performs a hybrid simulation,
        # and afterwards samples from the resulting quantum state.
        @terminal_sampling
        def main():
            psi_conv = conv_psi_prep()
            return psi_conv

        res_dict = main()
        max_ = max(res_dict.values())
        convolved_signal_qsp = np.sqrt([res_dict.get(key,0) / max_ for key in range(8)])
        print("qsp:", convolved_signal_qsp)
        # qsp: [0.7499999 , 1.        , 1.        , 0.74999996, 0.24999994, 
        # 0.        , 0.        , 0.25000007])

    """

    @qache
    def U(qv):
        for i in jrange(qv.size):
            p(np.pi * 2.0 ** (i - qv.size + 1), qv[i])

    d = len(weights) // 2

    qbl = QuantumBool()

    with conjugate(QFT)(qarg):
        GQSP(qbl, qarg, unitary=U, p=weights, k=d)

    return qbl