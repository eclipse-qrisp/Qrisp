from qrisp import  amplitude_amplification 
import numpy as np


def acc_IQAE(qargs,state_function, oracle_function, eps, alpha, kwargs_oracle = {}):
    """
    Accelerated Quantum Amplitude Estimation (IQAE). This function performs QAE with a fraction of the quantum resources of the well known QAE algorithm.
    See `Accelerated Quantum Amplitude Estimation without QFT <https://arxiv.org/abs/2407.16795>`_

    Parameters
    ----------
    qargs : QuantumVariable or list[QuantumVariable]
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
    eps: Float
        Accuracy of the algorithm. Choose to be :math:`> 0`. See paper for explanation.
    alpha: Float
        Confidence level of the algorithm. Choose to be :math:`\in (0,1)`. See paper for explanation.
    kwargs_oracle : dict, optional
        A dictionary containing keyword arguments for the oracle. The default is {}.

    Returns
    -------
    a_l, a_u : Float, Float
        Confidence bounds on the amplitude which is to be estimated.

    Examples
    --------

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

    We show the same QAE **Numerical integration** example which can also be found in the original QAE documentation.

    We wish to evaluate

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
    
    We define the oracle:

        def oracle_function(qb):   
            z(qb)


    For the ``state_function`` we want to evaluate $f(x)=\sin^2(x)$:  

    ::

        def state_function(inp, tar):
            h(inp)
    
            N = 2**inp.size
            for k in range(inp.size):
                with control(inp[k]):
                    ry(2**(k+1)/N,tar)

    Finally, we apply QAE and obtain an estimate $a$ for the value of the integral $A=0.27268$.

    ::

        eps = 0.01
        alpha = 0.01

        a_l, a_u = acc_QAE(input_list, state_function, oracle_function,eps=eps, alpha=alpha )

    >>> a_l 
    0.26782038552705856

    >>> a_u
    0.2741565776801965

    """


    E = 1/2 * pow(np.sin(np.pi * 3/14), 2) -  1/2 * pow(np.sin(np.pi * 1/6), 2) 
    F = 1/2 * np.arcsin(np.sqrt(2 * E))
    
    C = 4/ (6*F + np.pi)
    break_cond =  2 * eps + 1
    #theta_b = 0
    #theta_sh = 1
    K_i = 1
    m_i = 0
    index_tot = 0
    while break_cond > 2 * eps : 
        index_tot +=1
        
        alp_i = C*alpha * eps * K_i 
        N_i = int(np.ceil(1/(2 * pow(E, 2) ) * np.log(2/alp_i) ) )

        # perform quantum-stuff
        qargs_dupl = [qarg.duplicate() for qarg in qargs]
        A_i  = quantCirc( int((K_i -1 )/2) , N_i, qargs_dupl, state_function, 
                        oracle_function, kwargs_oracle ) 
        
        for qarg in qargs_dupl:
            qarg.delete()
        
        
        #for qarg in qargs_dupl:
            #qarg.delete()
        
        # compute new thetas
        theta_b, theta_sh = compute_thetas(m_i,  K_i, A_i, E)
        #compute new Li
        L_new, m_new = compute_Li(m_i , K_i, theta_b, theta_sh)
        
        m_i = m_new
        K_i = L_new * K_i
        

        break_cond = abs( theta_b - theta_sh )
    
    #full_res: 
    final_res = np.sin((theta_b+theta_sh)/2)**2
    return final_res



def quantCirc(k,N, qargs,state_function, 
        oracle_function, kwargs_oracle ):
    """
    Performs the quantum diffusion step, i.e. Quantum Amplitude Amplification, 
    in accordance to `Accelerated Quantum Amplitude Estimation without QFT <https://arxiv.org/abs/2407.16795>`_

    Parameters
    ----------
    k : int
        The amount of amplification steps, i.e. the power of :math:`\mathcal{Q}` in amplitude amplification.
    N : int
        The amount of shots, i.e. the amount of times the last qubit is measured after the amplitude amplification steps.
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

    #for arg in qargs:
        #print(arg)

    # store result of last qubit 
    # shot-based measurement auf den zustand, aber eigentlich nur auf das letzte QUbit, bzw. quantum_bool, welche getaggede zustände widerspiegelt
    res_dict = qargs[-1].get_measurement(shots = N)
    
    # case of single dict return, this should not occur but to be safe
    if True not in  list(res_dict.keys()):
        return 0 
    
    a_i = res_dict[True]

    return a_i 



# theta arithmetics
def compute_thetas(m_i, K_i, A_i, E): 
    """
    Helper function to perform statistical evaluation and compute the angles for the next iteration. 
    See `the original paper <https://arxiv.org/abs/2407.16795>`_ , Algorithm 1.
    
    Parameters
    ----------
    m_i : Int
        Used for the computation of the interval of allowed angles.
    K_i : Int
        Maximal power of ``oracle_function`` for next iteration.
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

    assert round( pow( np.sin(K_i * theta_b),2) , 8 )  == round( b_max,8)
    assert round( pow( np.sin(K_i * sh_theta),2), 8 )  == round( sh_min,8)

    return theta_b, sh_theta



def update_angle(old_angle, m_in):
    """
    Subroutine to compute new angles
    Parameters
    ----------

    old_angle : Float
        Ond angle from last iteration.    
    m_in : Int
        Used for the computation of the interval of allowed angles.

    """
    val_intermed1 = np.arcsin( np.sqrt(old_angle) ) - np.pi
    val_intermed2 = np.arcsin( - np.sqrt(old_angle) )
    cond_break = True
    while cond_break :
        if not (m_in*np.pi/2 <= val_intermed1 <= (m_in+1)*np.pi/2):
            val_intermed1 +=   np.pi
        else: 
            final_intermed = val_intermed1
            cond_break = False
            break

        if not (m_in*np.pi/2 <= val_intermed2 <= (m_in+1)*np.pi/2):
            val_intermed2 +=  np.pi
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
    m_i : Int
        Used for the computation of the interval of allowed angles.
    K_i : Int
        Maximal power of ``oracle_function`` for next iteration.
    theta_b : Float
        Lower bound for angle from last iteration.
    theta_b : Float
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
            #check the conditions
            if m_new*np.pi/2 <= first_val <= (m_new+1)*np.pi/2:
                if m_new*np.pi/2 <= sec_val <= (m_new+1)*np.pi/2:
                    
                    #assert  m_new*np.pi/2 <= Li *K_i * theta_b <= (m_new+1)*np.pi/2
                    #assert  m_new*np.pi/2 <= Li *K_i * theta_sh <= (m_new+1)*np.pi/2
                    elem1 = Li
                    elem2 = m_new

                    return  elem1, elem2
