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
import numpy as np
from qrisp import *
from qrisp.gqsp import convolve
from scipy.ndimage import convolve as scipy_convolve

def test_qsp_convolution():

    # A simple square wave signal
    psi = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    # A simple 3-point smoothing filter {a_{-1}, a_0, a_1} = {0.25, 0.5, 0.25}
    f = np.array([0.25, 0.5, 0.25])

    # Mode 'wrap' performs cyclic convolution
    convolved_signal_scipy = scipy_convolve(psi, f, mode='wrap')
    # [0.75 1.   1.   0.75 0.25 0.   0.   0.25]

    def psi_prep():
        qv = QuantumFloat(3)
        prepare(qv, psi)
        return qv

    # Converts the function to be executed within a repeat-until-success (RUS) procedure.
    @RUS
    def conv_psi_prep():
        qarg = psi_prep()
        qbl = convolve(qarg, f)
        success_bool = measure(qbl) == 0
        return success_bool, qarg

    # The terminal_sampling decorator performs a hybrid simulation,
    # and afterwards samples from the resulting quantum state.
    @terminal_sampling
    def main():
        psi_conv = conv_psi_prep()
        return psi_conv

    # Convert the resulting measurement probabilities to amplitudes by appling the square root.
    res_dict = main()
    max_ = max(res_dict.values())
    for k,v in res_dict.items():
        res_dict[k] = (v / max_) ** 0.5 
    convolved_signal = np.array([res_dict.get(key, 0) for key in range(8)])
    #array([0.7499999 , 1.        , 1.        , 0.74999996, 0.24999994, 0.        , 0.        , 0.25000007])
    assert np.linalg.norm(convolved_signal_scipy - convolved_signal) < 1e-4