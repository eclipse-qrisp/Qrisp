"""
\********************************************************************************
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
********************************************************************************/
"""

def amplitude_amplification(args, state_function, oracle_function, kwargs_oracle={}, iter=1, reflection_indices=None):
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
    refection_indices : list[int], optional
        A list indicating with respect to which variables the reflection within the diffuser is performed, i.e. oblivious amplitude amplification is performed.
        By default, the reflection is performed with respect to all variables in ``args``, i.e. standard amplitude amplification is performed.

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

    from qrisp.grover import diffuser
    from qrisp import merge, recursive_qs_search, IterationEnvironment
    from qrisp.jasp import check_for_tracing_mode, jrange

    if check_for_tracing_mode():
        for i in jrange(iter):
            oracle_function(*args, **kwargs_oracle)
            diffuser(args, state_function=state_function, reflection_indices=reflection_indices)
    else:
        merge(args)
        qs = recursive_qs_search(args)[0]
        with IterationEnvironment(qs, iter):
            oracle_function(*args, **kwargs_oracle)
            diffuser(args, state_function=state_function, reflection_indices=reflection_indices)
