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
import jax.numpy as jnp
from qrisp import (
    QuantumVariable,
    QuantumBool,
    h,
    p,
)
from qrisp.algorithms.gqsp.gqsp import GQSP
from qrisp.jasp import qache, jrange
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from jax.typing import ArrayLike


# QSP version of https://iopscience.iop.org/article/10.1088/2058-9565/acfc62
def fourier_series_loader(
    qarg: QuantumVariable, 
    signal: Optional["ArrayLike"] = None, 
    frequencies: Optional["ArrayLike"] = None, 
    k: int = 1, 
    mirror: bool = False
) -> QuantumBool:
    r"""
    Performs band-limited quantum state preparation.

    Given an input array of $M$ values $\{a_{j}\}_{j=0}^{M-1}$ representing a signal sampled at equidistant points, 
    this method prepares an $n$-qubit quantum state $(N=2^{n})$ by reconstructing a smooth approximation of the signal using its lowest $2k+1$ frequency components.

    First, the method computes the frequency coefficients $c_{l}$:

    .. math::

        c_l = \frac{1}{M}\sum_{j=0}^{M-1} a_j e^{-i \frac{2\pi jl}{M}}

    The method prepares the $n$-qubit state:

    .. math::

        \ket{\psi} = \sum_{m=0}^{N-1} \psi_m \ket{m}

    where the amplitudes $\psi_{m}$ are computed using a band-limited inverse transform:

    .. math::

        \psi_m = \frac{1}{\mathcal{K}}\sum_{l=-k}^k c_l e^{i \frac{2\pi lm}{N}}

    In this expression, $\mathcal{K}$ is a normalization constant ensuring $\sum |\psi _{m}|^{2}=1$.

    Notes
    -----
    - This method is particularly useful for preparing smooth states or approximating continuous functions where high-frequency noise should be filtered out.
    - If \(2k+1=M=N\), this reduces to a standard state preparation from a full DFT.


    Parameters
    ----------
    qarg : QuantumVariable
        The input variable in state $\ket{0}$.
    signal : ArrayLike, optional
        1-D array of input signal values with shape ``(M,)``.
        Either ``signal`` or ``frequencies`` must be specified.
    frequencies : ArrayLike, optional
        1-D array of input frequency values in the range $[-K,K]$ with shape ``(2K+1,)``.
    k : int
        The frequency cutoff. Only frequencies in the range $[-k,k]$ are preserved. The default is 1.
    mirror : bool
        If True, frequencies are caluclated from mirror padded ``signal`` via FFT to mitigate artifacts at the boundaries.
        The default is False.

    Returns
    -------
    QuantumBool
        Auxiliary variable after applying the GQSP protocol. 
        Must be measured in state $\ket{0}$ for the GQSP protocol to be successful.

    Examples
    --------

    We prepare a quantum state with amplitudes following a Gaussian distribution.

    ::

        # Restart the kernel to enable high-precision simulation
        import os
        os.environ["QRISP_SIMULATOR_FLOAT_THRESH"] = "1e-10"

        import jax.numpy as jnp
        import matplotlib.pyplot as plt
        import numpy as np
        from qrisp import *
        from qrisp.gqsp import fourier_series_loader

        # Gaussian 
        def f(x, alpha):
            return jnp.exp(-alpha * x ** 2)

        # Converts the function to be executed within a 
        # repeat-until-success (RUS) procedure.
        @RUS(static_argnames=["k"])
        def prepare_gaussian(n, alpha, k):
            # Use 32 sampling points to evaluate f
            N_samples = 32
            x_val = jnp.arange(-1.0, 1.0, 2.0 / N_samples)
            y_val = f(x_val, alpha)
            y_val = y_val / jnp.linalg.norm(y_val)

            qv = QuantumFloat(n)
            qbl = fourier_series_loader(qv, y_val, k=k)
            success_bool = measure(qbl) == 0
            reset(qbl)
            qbl.delete()
            return success_bool, qv

        # The terminal_sampling decorator performs a hybrid simulation,
        # and afterwards samples from the resulting quantum state.
        @terminal_sampling
        def main(n, alpha):
            qv =  prepare_gaussian(n, alpha, 3)
            return qv   

        # Run the simulation for n-qubit state
        n = 6
        alpha = 4
        res_dict = main(n, alpha)
        y_val_sim = np.sqrt([res_dict.get(key, 0) for key in range(2 ** n)])

        # Compare to target amplitudes
        x_val = np.arange(-1, 1, 2 ** (-n + 1))
        y_val = f(x_val, alpha)
        y_val = y_val / np.linalg.norm(y_val)

        plt.scatter(x_val, y_val, color='#20306f', marker="d", linestyle="solid", s=20, label="target")
        plt.plot(x_val, y_val_sim, color='#6929C4', marker="o", linestyle="solid", alpha=0.5, label="qsp")
        plt.xlabel("x", fontsize=15, color="#444444")
        plt.ylabel("Amplitudes f(x)", fontsize=15, color="#444444")
        plt.legend(fontsize=15, labelcolor='linecolor')
        plt.tick_params(axis='both', labelsize=12)
        plt.grid()
        plt.show()

    .. image:: /_static/qsp_gaussian.png
        :alt: Gaussian state preparation
        :align: center
        :width: 600px
        
    To perform quantum resource estimation, replace the ``@terminal_sampling`` decorator with ``@count_ops(meas_behavior="0")``:

    ::

        @count_ops(meas_behavior="0")
        def main(n, alpha):
            qv =  prepare_gaussian(n, alpha, 3)
            return qv   

        main(6, 4)
        # {'h': 6, 'rz': 8, 'p': 108, 'x': 6, 'cx': 72, 'rx': 7, 'measure': 1}

    """
    
    ALLOWED_MODES = {"standard", "mirror"}

    if frequencies is not None:
        K = (len(frequencies) - 1) // 2 
        compressed_frequencies = frequencies[K-k : K+k+1]
        scaling_factor = 1.0
    elif signal is not None:
        if mirror:
            # Mirror padding to mitiagate artifacts at the boundaries
            target_array_mirr = jnp.concatenate([signal, signal[::-1]])
            # Discrete Fourier transform
            frequencies = jnp.fft.fft(target_array_mirr)
            scaling_factor = 0.5
        else:
            # Discrete Fourier transform
            frequencies = jnp.fft.fft(signal)
            scaling_factor = 1.0

        # Compression
        compressed_frequencies = jnp.concatenate([frequencies[-k:], frequencies[:k+1]])
    else:
        raise Exception("Either signal or frequencies must be specified")
    
    @qache
    def U(qv, scaling_factor=1.0):
        # Scaling factor: 1.0 standard FFT; 0.5 mirror padding + FFT
        for i in jrange(qv.size):
            p(scaling_factor * np.pi * 2.0 ** (i - qv.size + 1), qv[i])

    h(qarg)

    qbl = QuantumBool()

    GQSP(qbl, qarg, unitary=U, p=compressed_frequencies, k=k, kwargs={"scaling_factor" : scaling_factor})

    return qbl