.. _QMCItutorial:

Quantum Monte Carlo Integration with Iterative QAE
==================================================

This tutorial will provide you with an introduction of Quantum Monte Carlo Integration within Qrisp.

For this purpose we will first give you a theoretical overview of what this technique is about and where it is used. 
Then we will dive into the practical implemention within Qrisp. This also includes the usage of :ref:`Iterative Quantum Amplitude estimation <acciqae>` . 
To finish of this tutorial, we investigate the full implementation of simple example by integrating :math:`x^2` over a uniform distribution in the interval :math:`\lbrack 0,1 \rbrack` .

The relevant literature can be found in the following papers: `A general quantum algorithm for numerical integration <https://www.nature.com/articles/s41598-024-61010-9>`_ and `Option pricing using Quantum computers <https://arxiv.org/pdf/1905.02666>`_ for QMCI and `Accelerated Quantum Amplitude Estimation
without QFT <https://arxiv.org/pdf/2407.16795>`_ for IQAE.

Theoretical overview of QMCI
----------------------------

QMCI tackels the same problems as its classical counterpart: Numerical integration of high-dimensional functions over probility distributions.

Mathemically speaking, we want to find an approximation for the following (general) integral

.. math::

    \int_{ { \lbrack 0,1 \rbrack }^n } f(x_1 , ... , x_n) \text{d} \mu (x_1 , ... , x_n)

As one does, we approximate an integral which cannot be solved analytically as a bunch of sums.

Integrals like these may appear in many different places, from chemistry, over many-body physics to mathematical finance.

Implementation in Qrisp
-----------------------

The implementation in Qrisp requires the implementation of the function on a :ref:`QuantumFloat` . We will see how this can be done later in the example. 

There is multiple ways to implement Monte Carlo integration in a quantum fashion. Within Qrisp, we use the approach of :ref:`quantum counting <quantum_counting>` . The idea is to discretize not only the :math:`x`-axis but the :math:`y`-axis aswell. We use two ``QuantumFloats``  for this. 
One ``QuantumFloat`` will hold the discretized values of the distribution, i.e. the relevant support on the :math:`x`-axis, while the other one will hold the discretized values of the :math:`y`-axis into which we will encode the the function values.
We can then simply count, in quantum fashion, the number of points under the function curve, and divide it by the number of total points.

Don't give up just yet, the mathematical description will bring you more clarity!

We start with a ``state_function`` that encodes the ``QuantumFloats`` as follows

.. math::

    \ket{0} \ket{0} \rightarrow \frac{1}{\sqrt{M \cdot N}} \sum^{N-1}_{x=0} \sum^{M-1}_{1=0} \ket{x} \ket{y}

The ``oracle_function`` encodes the "number of points under the curve" condition as follows:

.. math::

    \ket{x} \ket{y} \rightarrow (-1)^{f(x) \leq y} \ket{x} \ket{y}

We now arrive at the central step of this algorithm, which is :ref:`Quantum Amplitude Estimation <QAE>`. We use it to find

.. math::

    p(\{ (x,y) \mid f(x) \leq y \}) = \frac{1}{N} \sum^{N-1}_{x=0} \frac{1}{M} \sum^{M-1}_{y=0}  \mathbb{1}_{f(x) \leq y} \approx \frac{1}{N} \sum^{N-1}_{x=0} \frac{f(x)}{M}

The last expression is then the approximation for the integral in question. 


Iterative Quantum Amplitude Estimation
--------------------------------------

In Qrisp we have the option of using an alternative amplitude estimation algorithm, namely `Accelerated Quantum Amplitude Estimation, see Algorithm 1 <https://arxiv.org/pdf/2407.16795>`_ , which iteratively applies :ref:`amplitude amplification <QAE>` to find an estimation for the amplitude of the good state.
The goal of the algorithm is as follows: 

We start with a unitary operator :math:`\textbf{A}`, which represents a quantum circuit on :math:`w+1` qubits s.t.

.. math::

    \textbf{A} \ket{0}_{w+1} = \sqrt{1-a} \ket{\psi_0}_{w} \ket{0} + \sqrt{a} \ket{\psi_1}_{w} \ket{1},

where :math:`a \in \lbrack 0 , 1 \rbrack` is unknown. :math:`\ket{\psi_{0,1}}_{w}` are :math:`w`-qubit states. 

The algorithm in question allows for us to establish an estimate :math:`\hat{a}` of the unknown :math:`a`. 

Mathematically speaking this means, given an error :math:`\epsilon` and a confidence level :math:`\alpha`, the accelerated Iterative Quantum Amplitude estimation finds an estimate :math:`\hat{a}`, s.t.

.. math::

    \text{P} \{ \mid \hat{a} - a \mid \leq \epsilon \} \geq 1 - \alpha 

Below you can see the base function with implements this algorithm, including explanatory comments. It is a straight-forward translation from the theoretical ideas presented in the paper. For further explanations, have a look at the paper itself!

The implementations of subroutines can found in the :ref:`accelerated IQAE <acciqae>` reference.

::

    def acc_IQAE(qargs,state_function, oracle_function, eps, alpha, kwargs_oracle = {}):
        
        # start by defining the relevant constants 
        E = 1/2 * pow(np.sin(np.pi * 3/14), 2) -  1/2 * pow(np.sin(np.pi * 1/6), 2) 
        F = 1/2 * np.arcsin(np.sqrt(2 * E))
        C = 4/ (6*F + np.pi)

        # the break condition defines when the algorithm converges with the desired accurarcy
        break_cond =  2 * eps + 1
        K_i = 1
        m_i = 0
        index_tot = 0
        
        # the main loop
        while break_cond > 2 * eps : 
            index_tot +=1
            
            # further constant defined
            alp_i = C*alpha * eps * K_i 
            N_i = int(np.ceil(1/(2 * pow(E, 2) ) * np.log(2/alp_i) ) )

            # perform Quantum Amplitude amplification, and measure the number of |1> for the last qubit
            qargs_dupl = [qarg.duplicate() for qarg in qargs]
            A_i  = quantCirc( int((K_i -1 )/2) , N_i, qargs_dupl, state_function, 
                            oracle_function, kwargs_oracle ) 
            
            for qarg in qargs_dupl:
                qarg.delete()

            
            # compute new thetas
            theta_b, theta_sh = compute_thetas(m_i,  K_i, A_i, E)
            # compute new Li
            L_new, m_new = compute_Li(m_i , K_i, theta_b, theta_sh)
            
            # assign new parameters
            m_i = m_new
            K_i = L_new * K_i
            
            # set new breaking condition
            break_cond = abs( theta_b - theta_sh )
        
        # return the final approximation 
        final_res = np.sin((theta_b+theta_sh)/2)**2
        return final_res




The QMCI class - full example
-----------------------------

Next up, we will go through a full example implementation to integrate :math:`x^2` over a uniform distribution in :math:`\lbrack 0,1 \rbrack`. This is the equivalent to the QMCI function. 

First, we define the uniform distribution on a ``QuantumFloat``, which is just a uniform superposition of all qubits.

::

    def uniform(*args):
        for arg in args:
            h(arg)

We also need a function that we want to integrate.

::

    def f(qf):
        return qf*qf
    

Next, we create the ``QuantumFloat``, on which we evaluate our function and a duplicate for the discretization of the :math:`y`-axis


::

    
    qf = QuantumFloat(2,-2)

    dupl_args = [arg.duplicate() for arg in qargs]
    dupl_res_qf = function(*dupl_args)
    qargs.append(dupl_res_qf.duplicate())

    for arg in dupl_args:
        arg.delete()
    dupl_res_qf.delete()


We also have consider whether the ``QuantumFloat`` is not definded within a interval that differs from :math:`\lbrack 0, 1 \rbrack` . 
In a way we calculate the volume of space over which the ``QuantumFloat`` is defined.

We also append a ``QuantumBool`` to our input ``qargs``, which will serve as the final qubit to be measured, i.e. the qubit in register :math:`w+1`.  

::

    V0=1
    for arg in qargs:
        V0 *= 2**(arg.size+arg.exponent)
    
    qargs.append(QuantumBool())

Now we arrive at the heart of the algorithm, the definition of the ``oracle_function`` and the ``state_function``.

Let us first look at the ``state_function``:

::

    @auto_uncompute
    def state_function(*args):
        qf_x = args[0]
        qf_y = args[1]
        tar = args[2]

        distribution(qf_x)
        h(qf_y)
        qbl = (qf_y < function(qf_x))
        cx(qbl,tar)

It receives the ``@auto_uncompute`` :ref:`decorator <uncomputation>`. We apply the chosen distribution to ``qf_x``, which represents the :math:`x`-axis support. As explained earlier, we also discretize the :math:`y`-axis by appling an ``h``-gate to ``qf_y``.
We then evaluate in superposition which states in ``qf_y`` are smaller than the chosen function acting on ``qf_x``, i.e. the function's support in the distribution.

We save the result of the comparison in a ``QuantumBool``, from which we can extract the measurement of the final qubit in register :math:`w+1` by applying a ``cx`` gate on the previously mentioned ``QuantumBool``

This leads us to the ``oracle_function``

::
    
    def oracle_function(*args):  
        tar = args[2]
        z(tar)

It simply serves the function of tagging the :math:`\ket{1}`-state of the final qubit.

With everything in place we can now execute the Iterative QAE algorithm, with a chosen error tolerance ``eps`` and a confidence level ``alpha``
We also have to rescale with the previously calculated volume ``V0`` .

::

    a = acc_QAE(qargs, state_function, oracle_function, eps= 0.01, alpha= 0.01) 
    V = V0*a

Aaaand that's it! The QMCI is complete! 

Let us now have a look at the result, and compare it to the expected result:

::

    >>> V
    0.21855991519015455

    >>> (0+0.25**2+0.5**2+0.75**2)/4
    0.21855991519015455

