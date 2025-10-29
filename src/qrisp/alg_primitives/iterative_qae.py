"""
********************************************************************************
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
********************************************************************************
"""

from jax.lax import while_loop

from qrisp import control, z
from qrisp.alg_primitives.qae import amplitude_amplification
from qrisp.jasp import check_for_tracing_mode, expectation_value


def IQAE(qargs, state_function, eps, alpha, mes_kwargs={}):
    r"""
    Accelerated Quantum Amplitude Estimation (IQAE). This function performs :ref:`QAE <QAE>` with a fraction of the quantum resources of the well-known `QAE algorithm <https://arxiv.org/abs/quant-ph/0005055>`_.
    See `Accelerated Quantum Amplitude Estimation without QFT <https://arxiv.org/abs/2407.16795>`_.

    The problem of iterative quantum amplitude estimation is described as follows:

    * Given a unitary operator :math:`\mathcal{A}`, let :math:`\ket{\Psi}=\mathcal{A}\ket{0}\ket{\text{False}}`.
    * Write :math:`\ket{\Psi}=\sqrt{a}\ket{\Psi_1}\ket{\text{True}}+\sqrt{1-a}\ket{\Psi_0}\ket{\text{False}}` as a superposition of the orthogonal good and bad components of :math:`\ket{\Psi}`.
    * Find an estimate for $a$, the probability that a measurement of $\ket{\Psi}$ yields a good state.

    Parameters
    ----------
    qargs : list[:ref:`QuantumVariable`] or callable
        A list of QuantumVariables which represent the state on which the quantum amplitude estimation is performed,
        or a function preparing a list of QuantumVariables.
        The last variable in the list must be of type :ref:`QuantumBool`.
    state_function : callable
        A Python function preparing the state :math:`\ket{\Psi}`.
        This function will receive the variables returned by ``init_function`` as arguments.
    eps : float
        Accuracy $\epsilon>0$ of the algorithm.
    alpha : float
        Confidence level $\alpha\in (0,1)$ of the algorithm.
    mes_kwargs : dict, optional
        The keyword arguments for the measurement function. Default is an empty dictionary.

    Returns
    -------
    a : float
        An estimate $\hat{a}$ of $a$ such that

    .. math::

        \mathbb P\{|\hat{a}-a|<\epsilon\}\geq 1-\alpha

    Examples
    --------

    We show the same **Numerical integration** example which can also be found in the :ref:`QAE documentation <QAE>`.

    We wish to evaluate

    .. math::

        A=\int_0^1f(x)\mathrm dx.

    For this, we set up the corresponding ``state_function`` acting on the variables in ``input_list``:

    ::

        from qrisp import QuantumFloat, QuantumBool, control, z, h, ry, IQAE
        import numpy as np

        n = 6
        inp = QuantumFloat(n,-n)
        tar = QuantumBool()
        input_list = [inp, tar]

    For example, if $f(x)=\sin^2(x)$, the ``state_function`` can be implemented as follows:

    ::

        def state_function(inp, tar):
            h(inp)

            N = 2**inp.size
            for k in range(inp.size):
                with control(inp[k]):
                    ry(2**(k+1)/N,tar)

    Finally, we apply IQAE and obtain an estimate $a$ for the value of the integral $A=0.27268$.

    ::

        a = IQAE(input_list, state_function, eps=0.01, alpha=0.01)

    >>> a
    0.26782038552705856

    """

    if callable(qargs):

        init_function = qargs

    else:

        templates = [qv.template() for qv in qargs]

        def init_function():
            qargs_ = [temp.construct() for temp in templates]
            return qargs_

    # The oracle tagging the good states
    def oracle_function(*args):
        tar = args[-1]
        z(tar)

    if check_for_tracing_mode:
        import jax.numpy as jnp
    else:
        import numpy as jnp

    E = 1 / 2 * jnp.pow(jnp.sin(jnp.pi * 3 / 14), 2) - 1 / 2 * pow(
        jnp.sin(jnp.pi * 1 / 6), 2
    )
    F = 1 / 2 * jnp.arcsin(jnp.sqrt(2 * E))

    C = 4 / (6 * F + jnp.pi)
    break_cond = 2 * eps + 1

    K_i = 1
    m_i = 0

    theta_b = 0
    theta_sh = 0

    L_arr = jnp.array([3, 3, 3, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7])
    m_arr = jnp.array([0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6])

    def cond_fun(state):
        L_arr, m_arr, break_cond, alpha, eps, m_i, K_i, theta_b, theta_sh = state
        return break_cond > 2 * eps

    def body_fun(state):
        L_arr, m_arr, break_cond, alpha, eps, m_i, K_i, theta_b, theta_sh = state

        alp_i = C * alpha * eps * K_i
        N_i = jnp.int64(jnp.ceil(1 / (2 * jnp.pow(E, 2)) * jnp.log(2 / alp_i)))

        # Perform quantum step
        A_i = quantum_step(
            jnp.int64((K_i - 1) / 2),
            N_i,
            init_function,
            state_function,
            oracle_function,
            mes_kwargs,
        )

        # Compute new thetas
        theta_b, theta_sh = compute_thetas(m_i, K_i, A_i, E)

        # Compute new L_i
        L_new, m_new = compute_Li(L_arr, m_arr, m_i, K_i, theta_b, theta_sh)
        m_i = m_new
        K_i = L_new * K_i

        break_cond = jnp.float64(jnp.abs(theta_b - theta_sh))

        return L_arr, m_arr, break_cond, alpha, eps, m_i, K_i, theta_b, theta_sh

    state = (L_arr, m_arr, break_cond, alpha, eps, m_i, K_i, theta_b, theta_sh)

    if check_for_tracing_mode():
        L_arr, m_arr, break_cond, alpha, eps, m_i, K_i, theta_b, theta_sh = while_loop(
            cond_fun, body_fun, state
        )
    else:
        while cond_fun(state):
            state = body_fun(state)
        L_arr, m_arr, break_cond, alpha, eps, m_i, K_i, theta_b, theta_sh = state

    final_res = jnp.sin((theta_b + theta_sh) / 2) ** 2
    return final_res


def quantum_step(k, N, init_function, state_function, oracle_function, mes_kwargs):
    """
    Performs the quantum step, i.e., Quantum Amplitude Amplification,
    in accordance to `Accelerated Quantum Amplitude Estimation without QFT <https://arxiv.org/abs/2407.16795>`_

    Parameters
    ----------
    k : int
        The amount of amplification steps, i.e., the power of :math:`\mathcal{Q}` in amplitude amplification.
    N : int
        The amount of shots, i.e., the amount of times the last qubit is measured after the amplitude amplification steps.
    init_function : callable
        A Python function that returns a list of QuantumVariables representing the state on which the quantum amplitude estimation is performed.
        The last variable in the list must be of type :ref:`QuantumBool`.
    state_function : callable
        A Python function preparing the state :math:`\ket{\Psi}`.
        This function will receive the variables in the list returnded by ``init_function`` as arguments.
    oracle_function : callable
        A Python function tagging the good state :math:`\ket{\Psi_1}`.
        This function will receive the variables in the list ``args`` as arguments in the
        course of this algorithm.
    mes_kwargs : dict, optional
        The keyword arguments for the measurement function. Default is an empty dictionary.
    """

    def state_prep(k):
        qargs = init_function()
        state_function(*qargs)
        amplitude_amplification(qargs, state_function, oracle_function, iter=k)
        return qargs[-1]

    if check_for_tracing_mode():
        a_i = expectation_value(state_prep, shots=N)(k)
    else:
        mes_kwargs["shots"] = N
        res_dict = state_prep(k).get_measurement(**mes_kwargs)
        a_i = res_dict.get(True, 0)

    return a_i


def compute_thetas(m_i, K_i, A_i, E):
    """
    Helper function to compute the angles for the next iteration.
    See `the original paper <https://arxiv.org/abs/2407.16795>`_ , Algorithm 1.

    Parameters
    ----------
    m_i : int
        Used for the computation of the interval of allowed angles.
    K_i : int
        Maximal amount of amplitude amplification steps for the next iteration.
    A_i : float
        Share of ``1``-measurements in amplitude amplification steps.
    E : float
        :math:`\epsilon` limit.
    """

    if check_for_tracing_mode:
        import jax.numpy as jnp
    else:
        import numpy as jnp

    b_max = jnp.max(jnp.array([A_i - E, 0]))
    sh_min = jnp.min(jnp.array([A_i + E, 1]))

    theta_b = (
        (m_i + m_i % 2) * jnp.pi / 2
        + jnp.pow(-1, m_i % 2) * jnp.arcsin(jnp.sqrt(b_max))
    ) / K_i
    theta_sh = (
        (m_i + m_i % 2) * jnp.pi / 2
        + jnp.pow(-1, m_i % 2) * jnp.arcsin(jnp.sqrt(sh_min))
    ) / K_i

    # assert np.round( np.pow( np.sin(K_i * theta_b),2) , 8 )  == np.round(b_max, 8)
    # assert np.round( np.pow( np.sin(K_i * theta_sh),2), 8 )  == np.round(sh_min, 8)

    return theta_b, theta_sh


def compute_Li(L_arr, m_arr, m_i, K_i, theta_b, theta_sh):
    """
    Helper function to compute further values for the next iteration.
    See `the original paper <https://arxiv.org/abs/2407.16795>`_ , Algorithm 1.

    Parameters
    ----------
    m_i : int
        Used for the computation of the interval of allowed angles.
    K_i : int
        Maximal amount of amplitude amplification steps for the next iteration.
    theta_b : float
        Lower bound for angle from last iteration.
    theta_b : float
        Upper bound for angle from last iteration.
    """

    if check_for_tracing_mode:
        import jax.numpy as jnp
    else:
        import numpy as jnp

    first_arr = L_arr * K_i * theta_b
    second_arr = L_arr * K_i * theta_sh

    lower_arr = (L_arr * m_i + m_arr) * jnp.pi / 2
    upper_arr = lower_arr + jnp.pi / 2

    index = jnp.argmax(
        (first_arr >= lower_arr)
        & (first_arr <= upper_arr)
        & (second_arr >= lower_arr)
        & (second_arr <= upper_arr)
    )

    L_new = L_arr[index]
    m_new = L_new * m_i + m_arr[index]

    return L_new, m_new
