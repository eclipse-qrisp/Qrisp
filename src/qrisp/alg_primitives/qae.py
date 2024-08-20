"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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

from qrisp.alg_primitives.qpe import QPE

def amplitude_amplification(args, state_function, oracle_function, kwargs_oracle={}, iter=1):
    r"""
    This method performs `quantum amplitude amplification <https://arxiv.org/abs/quant-ph/0005055>`_.

    The problem of quantum amplitude amplification is described as follows:

    * Given a unitary operator :math:`\mathcal{A}`, let :math:`\ket{\Psi}=\mathcal{A}\ket{0}`.
    * Write :math:`\ket{\Psi}=\ket{\Psi_1}+\ket{\Psi_0}` as a superposition of the orthogonal good and bad components of :math:`\ket{\Psi}`.
    * Enhance the probability :math:`a=\langle\Psi_1|\Psi_1\rangle` that a measurement of $\ket{\Psi}$ yields a good state.

    Let $\theta_a\in [0,\pi/2]$ such that $\sin^2(\theta_a)=a$. Then the amplitude amplification operator $\mathcal Q$ acts as

    .. math::

        \mathcal Q^j\ket{\Psi}=\frac{1}{\sqrt{a}}\sin((2j+1)\theta_a)\ket{\Psi_1}+\frac{1}{\sqrt{1-a}}\cos((2j+1)\theta_a)\ket{\Psi_0}.

    Therefore, after $m$ iterations the probability of measuring a good state is $\sin^2((2m+1)\theta_a)$. 
    
    Parameters
    ----------

    args : QuantumVariable or list[QuantumVariable]
        The (list of) QuantumVariables which represent the state,
        the amplitude amplification is performed on.
    state_function : function
        A Python function preparing the state $\ket{\Psi}$.
        This function will receive the variables in the list ``args`` as arguments in the
        course of this algorithm.
    oracle_function : function
        A Python function tagging the good state $\ket{\Psi_1}$.
        This function will receive the variables in the list ``args`` as arguments in the
        course of this algorithm.
    kwargs_oracle : dict, optional
        A dictionary containing keyword arguments for the oracle. The default is {}.
    iter : int, optional
        The amount of amplitude amplification iterations to perform. The default is 1.

    Examples
    --------

    We define a function that prepares the state :math:`\ket{\Psi}=\cos(\frac{\pi}{16})\ket{0}+\sin(\frac{\pi}{16})\ket{1}`
    and an oracle that tags the good state :math:`\ket{1}`. In this case, we have :math:`a=\sin^2(\frac{\pi}{16})\approx 0.19509`.
     
    ::

        from qrisp import z, ry, QuantumBool, amplitude_amplification
        import numpy as np

        def state_function(qb):
            ry(np.pi/8,qb)

        def oracle_function(qb):   
            z(qb)
        
        qb = QuantumBool()

        state_function(qb)

    >>> qb.qs.statevector(decimals=5)
    0.98079∣False⟩+0.19509∣True⟩

    We can enhance the probability of measuring the good state with amplitude amplification:

    >>> amplitude_amplification([qb], state_function, oracle_function)
    >>> qb.qs.statevector(decimals=5)
    0.83147*|False> + 0.55557*|True> 

    >>> amplitude_amplification([qb], state_function, oracle_function)
    >>> qb.qs.statevector(decimals=5)
    0.55557*|False> + 0.83147*|True> 

    >>> amplitude_amplification([qb], state_function, oracle_function)
    >>> qb.qs.statevector(decimals=5)
    0.19509*|False> + 0.98079*|True>

    """

    from qrisp import merge, IterationEnvironment, recursive_qs_search
    from qrisp.grover import diffuser

    merge(args)
    qs = recursive_qs_search(args)[0]
    with IterationEnvironment(qs, iter):
        oracle_function(*args, **kwargs_oracle)
        diffuser(args, state_function=state_function)


def QAE(args, state_function, oracle_function, kwargs_oracle={}, precision=None, target=None):
    r"""
    This method implements the canonical quantum amplitude estimation (QAE) algorithm by `Brassard et al. <https://arxiv.org/abs/quant-ph/0005055>`_.

    The problem of quantum amplitude estimation is described as follows:

    * Given a unitary operator :math:`\mathcal{A}`, let :math:`\ket{\Psi}=\mathcal{A}\ket{0}`.
    * Write :math:`\ket{\Psi}=\ket{\Psi_1}+\ket{\Psi_0}` as a superposition of the orthogonal good and bad components of :math:`\ket{\Psi}`.
    * Find an estimate for :math:`a=\langle\Psi_1|\Psi_1\rangle`, the probability that a measurement of $\ket{\Psi}$ yields a good state.

    Parameters
    ----------
    args : QuantumVariable or list[QuantumVariable]
        The (list of) QuantumVariables which represent the state,
        the quantum amplitude estimation is performed on.
    state_function : function
        A Python function preparing the state :math:`\ket{\Psi}`.
        This function will receive the variables in the list ``args`` as arguments in the
        course of this algorithm.
    oracle_function : function
        A Python function tagging the good state :math:`\ket{\Psi_1}`.
        This function will receive the variables in the list ``args`` as arguments in the
        course of this algorithm.
    kwargs_oracle : dict, optional
        A dictionary containing keyword arguments for the oracle. The default is {}.
    precision : int, optional
        The precision of the estimation. The default is None.
    target : QuantumFloat, optional
        A target QuantumFloat to perform the estimation into. The default is None.
        If given neither a precision nor a target, an Exception will be raised.

    Returns
    -------
    
    res : QuantumFloat
        A QuantumFloat encoding the angle :math:`\theta` as a fraction of :math:`\pi`,
        such that :math:`\tilde{a}=\sin^2(\theta)` is an estimate for :math:`a`. 

        More precisely, we have :math:`\theta=\pi\frac{y}{M}` for :math:`y\in\{0,\dotsc,M-1\}` and :math:`M=2^{\text{precision}}`.
        After measurement, the estimate :math:`\tilde{a}=\sin^2(\theta)` satisfies

        .. math::

            |a-\tilde{a}|\leq\frac{2\pi}{M}+\frac{\pi^2}{M^2}

        with probability of at least :math:`8/\pi^2`.

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

    
    Here, we demonstarate how to use QAE for numerical integration. 

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

    state_function(*args)
    res = QPE(args, amplitude_amplification, precision, target, iter_spec=True,
                kwargs={'state_function':state_function, 'oracle_function':oracle_function, 'kwargs_oracle':kwargs_oracle})
  
    return res