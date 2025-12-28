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
    h,
    p,
)
from qrisp.algorithms.gqsp.gqsp import GQSP
from qrisp.jasp import qache, jrange


@qache
def U(qv, scaling_factor=1.0):
    # Scaling factor: 1.0 standard FFT; 0.5 mirror padding + FFT
    for i in jrange(qv.size):
        p(scaling_factor * np.pi * 2.0 ** (i - qv.size + 1), qv[i])


def state_preparation(qarg, target_array, k=1, mode="standard"):
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
    target_array : ndarray
        1-D array of input signal values with shape ``(M,)``.
    k : int, optional
        The frequency cutoff. Only frequencies in the range $[-k,k]$ are preserved.
    mode: str, optional
        Available are 
        
        - ``"standard"``: frequencies are caluclated from ``target_array`` via FFT,
        - ``"mirror"``: frequencies are caluclated from mirror padded ``target_array`` via FFT to mitigate artifacts at the boundaries.

    Returns
    -------
    QuantumBool
        Auxiliary variable after applying the GQSP protocol. 
        Must be measuered in state $\ket{0}$ for the GQSP protocol to be successful.

    Examples
    --------

    We prepare a quantum state with amplitudes following a Gaussian distribution.

    ::

        import jax.numpy as jnp
        import matplotlib.pyplot as plt
        import numpy as np
        from qrisp import *
        from qrisp.gqsp import state_preparation


        # Gaussian 
        def f(x):
            return jnp.exp(-2*x**2)


        # Converts the function to be executed within a repeat-until-success (RUS) procedure.
        @RUS(static_argnums=1)
        def preprare_gaussian(n, k):

            # Evaluate f at equidistant sample points
            delta = 2.0 ** (-2*k)
            x_val = jnp.arange(-1, 1 ,delta)
            y_val = f(x_val)
            y_val = y_val / jnp.linalg.norm(y_val)

            qv = QuantumFloat(n)
            qbl = state_preparation(qv, y_val, k=k)
            success_bool = measure(qbl) == 0
            return success_bool, qv


        # The terminal_sampling decorator performs a hybrid simulation,
        # and afterwards samples from the resulting quantum state.
        @terminal_sampling
        def main():
            qv =  preprare_gaussian(10, 2)
            return qv   


        # Convert the resulting measurement probabilities to amplitudes by appling the square root.
        res_dict = main()
        for k,v in res_dict.items():
            res_dict[k] = v**0.5 
        y_val_sim = np.array([res_dict.get(key,0) for key in sorted(res_dict.keys())])
        y_val_sim = y_val_sim/np.linalg.norm(y_val_sim)

        # Compare to classical values
        x_val = np.linspace(-1, 1, len(y_val_sim))
        y_val = f(x_val)
        y_val = y_val / np.linalg.norm(y_val)

        plt.scatter(x_val, y_val, color='#20306f', marker="o", linestyle="solid", s=20, label="classical")
        plt.scatter(x_val, y_val_sim, color='#6929C4', marker="d", linestyle="solid", s=20, label="quantum")
        plt.xlabel("x", fontsize=16, color="#444444")
        plt.ylabel("Amplitudes f(x)", fontsize=16, color="#444444")
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=16, labelcolor='linecolor')
        plt.tight_layout()
        plt.grid()
        plt.show()

    .. image:: /_static/gaussian.png
        :alt: Gaussian state preparation
        :align: center
        :width: 600px
        
    To perform quantum resource estimation, replace the ``@terminal_sampling`` decorator with ``@count_ops(meas_behavior="0")``:

    ::

        @count_ops(meas_behavior="0")
        def main():
            qv =  preprare_gaussian(10, 2)
            return qv   

        main()
        # {'p': 120, 'z': 5, 'h': 10, 'cx': 80, 'u3': 5, 'gphase': 5, 'x': 4, 'measure': 1}

    """
    
    ALLOWED_MODES = {"standard", "mirror"}
    
    if mode=="standard":
        # Discrete Fourier transform
        frequencies = jnp.fft.fft(target_array)
        scaling_factor = 1.0
    elif mode=="mirror":
        # Mirror padding to mitiagate artifacts at the boundaries
        target_array_mirr = jnp.concatenate([target_array, target_array[::-1]])
        # Discrete Fourier transform
        frequencies = jnp.fft.fft(target_array_mirr)
        scaling_factor = 0.5
    else:
        raise ValueError(
            f"Invalid mode specified: '{mode}'. "
            f"Allowed kinds are: {', '.join(ALLOWED_MODES)}"
        )

    # Compression
    compressed_frequencies = jnp.concatenate([frequencies[-k:], frequencies[:k+1]])

    h(qarg)

    qbl = GQSP(qarg, U, compressed_frequencies, k=k, kwargs={"scaling_factor" : scaling_factor})

    return qbl