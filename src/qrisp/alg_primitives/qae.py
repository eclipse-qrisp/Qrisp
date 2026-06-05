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

from collections.abc import Callable, Sequence
from typing import Any

from qrisp.core import QuantumVariable, QuantumArray
from qrisp.qtypes import QuantumFloat
from qrisp.alg_primitives.amplitude_amplification import amplitude_amplification
from qrisp.alg_primitives.qpe import QPE


def QAE(
    args: QuantumVariable | QuantumArray | Sequence[QuantumVariable | QuantumArray],
    state_function: Callable,
    oracle_function: Callable,
    kwargs_oracle: dict[str, Any] | None = None,
    precision: int | None = None,
    target: QuantumFloat | None = None,
) -> QuantumFloat:
    r"""
    This method implements the canonical quantum amplitude estimation (QAE) algorithm by `Brassard et al. <https://arxiv.org/abs/quant-ph/0005055>`_.

    The problem of quantum amplitude estimation is described as follows:

    * Given a unitary operator :math:`\mathcal{A}`, let :math:`\ket{\Psi}=\mathcal{A}\ket{0}`.
    * Write :math:`\ket{\Psi}=\ket{\Psi_1}+\ket{\Psi_0}` as a superposition of the orthogonal good and bad components of :math:`\ket{\Psi}`.
    * Find an estimate for :math:`a=\langle\Psi_1|\Psi_1\rangle`, the probability that a measurement of :math:`\ket{\Psi}` yields a good state.

    Parameters
    ----------
    args : QuantumVariable | QuantumArray | Sequence[QuantumVariable | QuantumArray]
        The quantum variable, array, or collection thereof on which quantum amplitude estimation
        is performed. These variables must initially be in the zero state (:math:`\ket{0}`).
        The QAE algorithm will internally apply the ``state_function`` to these variables
        to prepare the initial state :math:`\ket{\Psi}`.
    state_function : Callable
        A Python function preparing the state :math:`\ket{\Psi}` from the zero state.
        This function is applied internally by the algorithm.
        The required signature of this function depends on the input ``args``:

        - if ``args`` is a single variable or array, it receives that single object.
        - if ``args`` is a sequence, the elements are unpacked and passed as separate
          positional arguments (e.g., for ``args=[qv1, qv2]``, the signature
          must be ``state_function(qv1, qv2)``).

    oracle_function : Callable
        A Python function tagging the good state :math:`\ket{\Psi_1}`.
        Like ``state_function``, its required signature matches the structure of ``args``:
        it takes a single argument if ``args`` is a single object, or unpacked
        positional arguments if ``args`` is a sequence.
    kwargs_oracle : dict, optional
        A dictionary containing keyword arguments for the oracle. The default is None.
    precision : int, optional
        The precision of the estimation (the number of evaluation qubits). The default is None.
    target : QuantumFloat, optional
        A target QuantumFloat to perform the estimation into. The default is None.
        If neither ``precision`` nor ``target`` is provided, a ValueError will be raised.

    Returns
    -------
    QuantumFloat
        A QuantumFloat encoding the angle :math:`\theta` as a fraction of :math:`\pi`,
        such that :math:`\tilde{a}=\sin^2(\theta)` is an estimate for :math:`a`.

        More precisely, we have :math:`\theta=\pi\frac{y}{M}` for :math:`y\in\{0,\dotsc,M-1\}` and :math:`M=2^{\text{precision}}`.
        After measurement, the estimate :math:`\tilde{a}=\sin^2(\theta)` satisfies

        .. math::

            |a-\tilde{a}|\leq\frac{2\pi}{M}+\frac{\pi^2}{M^2}

        with probability of at least :math:`8/\pi^2`.

    Raises
    ------
    ValueError
        If neither ``precision`` nor ``target`` is provided.

    Examples
    --------

    We define a function that prepares the state :math:`\ket{\Psi}=\cos(\frac{\pi}{8})\ket{0}+\sin(\frac{\pi}{8})\ket{1}`
    and an oracle that tags the good state :math:`\ket{1}`. In this case, we have :math:`a=\sin^2(\frac{\pi}{8})`.

    ::

        from qrisp import z, ry, QuantumBool, QAE
        import numpy as np

        def state_function(qb):
            ry(np.pi/4,qb)

        def oracle_function(qb):
            z(qb)

        qb = QuantumBool()

        res = QAE([qb], state_function, oracle_function, precision=3)

    >>> res.get_measurement()
    {0.125: 0.5, 0.875: 0.5}

    That is, after measurement we find $\theta=\frac{\pi}{8}$ or $\theta=\frac{7\pi}{8}$ with probability $\frac12$, respectively.
    Therefore, we obtain the estimate $\tilde{a}=\sin^2(\frac{\pi}{8})$ or $\tilde{a}=\sin^2(\frac{7\pi}{8})$.
    In this case, both results coincide with the exact value $a$.


    **Numerical integration**


    Here, we demonstrate how to use QAE for numerical integration.

    Consider a continuous function $f\colon[0,1]\rightarrow[0,1]$. We wish to evaluate

    .. math::

        A=\int_0^1f(x)\mathrm dx.

    For this, we set up the corresponding ``state_function`` acting on the ``input_list``:

    ::

        from qrisp import QuantumFloat, QuantumBool, control, z, h, ry, QAE
        import numpy as np

        n = 6
        inp = QuantumFloat(n,-n)
        tar = QuantumBool()
        input_list = [inp, tar]

    Here, $N=2^n$ is the number of sampling points the function $f$ is evaluated on. The ``state_function`` acts as

    .. math::

        \ket{0}\ket{0}\rightarrow\frac{1}{\sqrt{N}}\sum\limits_{x=0}^{N-1}\ket{x}\left(\sqrt{1-f(x/N)}\ket{0}+\sqrt{f(x/N)}\ket{1}\right).

    Then the probability of measuring $1$ in the target state ``tar`` is

    .. math::

            p(1)=\frac{1}{N}\sum\limits_{x=0}^{N-1}f(x/N),

    which acts as an approximation for the value of the integral $A$.

    The ``oracle_function``, therefore, tags the $\ket{1}$ state of the target state:

    ::

        def oracle_function(inp, tar):
            z(tar)

    For example, if $f(x)=\sin^2(x)$ the ``state_function`` can be implemented as follows:

    ::

        def state_function(inp, tar):
            h(inp)

            N = 2**inp.size
            for k in range(inp.size):
                with control(inp[k]):
                    ry(2**(k+1)/N,tar)

    Finally, we apply QAE and obtain an estimate $a$ for the value of the integral $A=0.27268$.

    ::

        prec = 6
        res = QAE(input_list, state_function, oracle_function, precision=prec)
        meas_res = res.get_measurement()

        theta = np.pi*max(meas_res, key=meas_res.get)
        a = np.sin(theta)**2

    >>> a
    0.26430

    """

    if kwargs_oracle is None:
        kwargs_oracle = {}

    if precision is None and target is None:
        raise ValueError(
            "Tried to call Quantum Amplitude Estimation without specifying "
            "either 'precision' or 'target'."
        )

    if isinstance(args, (QuantumVariable, QuantumArray)):
        args = [args]

    state_function(*args)
    res = QPE(
        args,
        amplitude_amplification,
        precision,
        target,
        iter_spec=True,
        kwargs={
            "state_function": state_function,
            "oracle_function": oracle_function,
            "kwargs_oracle": kwargs_oracle,
        },
    )

    return res
