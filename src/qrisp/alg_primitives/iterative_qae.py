"""
\********************************************************************************
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
********************************************************************************/
"""

from qrisp import z
from qrisp.alg_primitives.qae import amplitude_amplification 
import numpy as np


def IQAE(qargs, state_function, eps, alpha, kwargs_oracle = {}):
    r"""
    Accelerated Quantum Amplitude Estimation (IQAE). This function performs :ref:`QAE <QAE>` with a fraction of the quantum resources of the well-known `QAE algorithm <https://arxiv.org/abs/quant-ph/0005055>`_.
    See `Accelerated Quantum Amplitude Estimation without QFT <https://arxiv.org/abs/2407.16795>`_.

    The problem of iterative quantum amplitude estimation is described as follows:

    * Given a unitary operator :math:`\mathcal{A}`, let :math:`\ket{\Psi}=\mathcal{A}\ket{0}\ket{\text{False}}`.
    * Write :math:`\ket{\Psi}=\sqrt{a}\ket{\Psi_1}\ket{\text{True}}+\sqrt{1-a}\ket{\Psi_0}\ket{\text{False}}` as a superposition of the orthogonal good and bad components of :math:`\ket{\Psi}`.
    * Find an estimate for $a$, the probability that a measurement of $\ket{\Psi}$ yields a good state.

    Parameters
    ----------
    qargs : list[QuantumVariable]
        The list of QuantumVariables which represent the state,
        the quantum amplitude estimation is performed on. The last variable in the list must be of type :ref:`QuantumBool`.
    state_function : function
        A Python function preparing the state :math:`\ket{\Psi}`.
        This function will receive the variables in the list ``qargs`` as arguments in the
        course of this algorithm.
    eps : float
        Accuracy $\epsilon>0$ of the algorithm.
    alpha : float
        Confidence level $\alpha\in (0,1)$ of the algorithm.

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

    # The oracle tagging the good states
    def oracle_function(*args):  
        tar = args[-1]
        z(tar)

    E = 1/2 * pow(np.sin(np.pi * 3/14), 2) -  1/2 * pow(np.sin(np.pi * 1/6), 2) 
    F = 1/2 * np.arcsin(np.sqrt(2 * E))
    
    C = 4/ (6*F + np.pi)
    break_cond =  2 * eps + 1
    #theta_b = 0
    #theta_sh = 1
    K_i = 1
    m_i = 0
    index_tot = 0
    while break_cond > 2 * eps: 
        index_tot +=1
        
        alp_i = C*alpha * eps * K_i 
        N_i = int(np.ceil(1/(2 * pow(E, 2) ) * np.log(2/alp_i) ) )

        # Perform quantum step
        qargs_dupl = [qarg.duplicate() for qarg in qargs]
        A_i  = quant_step( int((K_i -1 )/2) , N_i, qargs_dupl, state_function, 
                        oracle_function, kwargs_oracle ) 
        
        for qarg in qargs_dupl:
            qarg.delete()
        
        # Compute new thetas
        theta_b, theta_sh = compute_thetas(m_i,  K_i, A_i, E)
        # Compute new Li
        L_new, m_new = compute_Li(m_i , K_i, theta_b, theta_sh)
        
        m_i = m_new
        K_i = L_new * K_i

        break_cond = abs( theta_b - theta_sh )
    
    final_res = np.sin((theta_b+theta_sh)/2)**2
    return final_res


def quant_step(k, N, qargs, state_function, oracle_function, kwargs_oracle):
    """
    Performs the quantum step, i.e., Quantum Amplitude Amplification, 
    in accordance to `Accelerated Quantum Amplitude Estimation without QFT <https://arxiv.org/abs/2407.16795>`_

    Parameters
    ----------
    k : int
        The amount of amplification steps, i.e., the power of :math:`\mathcal{Q}` in amplitude amplification.
    N : int
        The amount of shots, i.e., the amount of times the last qubit is measured after the amplitude amplification steps.
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
    """

    if k==0:
        state_function(*qargs)
    else:
        state_function(*qargs)
        amplitude_amplification(qargs, state_function, oracle_function, kwargs_oracle, iter = k)

    # store result of last qubit 
    # shot-based measurement 
    res_dict = qargs[-1].get_measurement(shots = N)
    
    # case of single dict return, this should not occur but to be safe
    if True not in list(res_dict.keys()):
        return 0 
    
    a_i = res_dict[True]

    return a_i 


def compute_thetas(m_i, K_i, A_i, E): 
    """
    Helper function to perform statistical evaluation and compute the angles for the next iteration. 
    See `the original paper <https://arxiv.org/abs/2407.16795>`_ , Algorithm 1.
    
    Parameters
    ----------
    m_i : Int
        Used for the computation of the interval of allowed angles.
    K_i : Int
        Maximal amount of amplitude amplification steps for the next iteration.
    A_i : Float
        Share of ``1``-measurements in amplitude amplification steps
    E : 
        :math:`\epsilon` limit
    """
    
    b_max = max(A_i - E, 0)
    sh_min = min(A_i + E, 1)

    theta_b_intermed = update_angle(b_max, m_i)
    theta_b = theta_b_intermed/K_i

    sh_theta_intermed = update_angle(sh_min, m_i)
    sh_theta = sh_theta_intermed/K_i

    assert round( pow( np.sin(K_i * theta_b),2) , 8 )  == round(b_max, 8)
    assert round( pow( np.sin(K_i * sh_theta),2), 8 )  == round(sh_min, 8)

    return theta_b, sh_theta


def update_angle(old_angle, m_in):
    """
    Subroutine to compute new angles.

    Parameters
    ----------
    old_angle : float
        Old angle from last iteration.    
    m_in : int
        Used for the computation of the interval of allowed angles.
    """

    val_intermed1 = np.arcsin( np.sqrt(old_angle) ) - np.pi
    val_intermed2 = np.arcsin( - np.sqrt(old_angle) )
    cond_break = True
    while cond_break :
        if not (m_in*np.pi/2 <= val_intermed1 <= (m_in+1)*np.pi/2):
            val_intermed1 += np.pi
        else: 
            final_intermed = val_intermed1
            cond_break = False
            break

        if not (m_in*np.pi/2 <= val_intermed2 <= (m_in+1)*np.pi/2):
            val_intermed2 += np.pi
        else: 
            final_intermed = val_intermed2
            cond_break = False
            break

    return final_intermed
    

def compute_Li(m_i , K_i, theta_b, theta_sh):
    """
    Helper function to perform statistical evaluation and compute further values for the next iteration. 
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

    Li_list = (3,5,7)

    # create the Li-values
    for Li in Li_list:
        # create the possible M_new vals
        m_new_list = range(Li*m_i, Li*m_i +Li)
        first_val = Li *K_i * theta_b
        sec_val = Li *K_i * theta_sh

        for m_new in m_new_list:
            # check the conditions
            if m_new*np.pi/2 <= first_val <= (m_new+1)*np.pi/2:
                if m_new*np.pi/2 <= sec_val <= (m_new+1)*np.pi/2:
                    
                    #assert  m_new*np.pi/2 <= Li *K_i * theta_b <= (m_new+1)*np.pi/2
                    #assert  m_new*np.pi/2 <= Li *K_i * theta_sh <= (m_new+1)*np.pi/2
                    elem1 = Li
                    elem2 = m_new

                    return  elem1, elem2
