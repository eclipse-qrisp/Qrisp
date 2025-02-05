.. _QMCItutorial:

Quantum Monte Carlo Integration with Iterative QAE
==================================================

This tutorial will provide you with an introduction to Quantum Monte Carlo (QMCI) Integration within Qrisp.

For this purpose, we will first give you a theoretical overview of what this technique is about and where it is used. 
Then we will dive into the practical implemention within Qrisp. This also includes the usage of :ref:`Iterative Quantum Amplitude Estimation <IQAE>` (IQAE). 
To finish of this tutorial, we investigate the full implementation of a simple example by integrating :math:`f(x)=x^2` w.r.t. the uniform distribution over the interval :math:`\lbrack 0,1 \rbrack`.

The relevant literature can be found in the following papers: `A general quantum algorithm for numerical integration <https://www.nature.com/articles/s41598-024-61010-9>`_ and `Option pricing using Quantum computers <https://arxiv.org/pdf/1905.02666>`_ for QMCI and `Accelerated Quantum Amplitude Estimation
without QFT <https://arxiv.org/pdf/2407.16795>`_ for IQAE.

QMCI tackles the same problems as its classical counterpart: Numerical integration of high-dimensional functions w.r.t. probility distributions.

Mathematically speaking, we want to find an approximation for the following (general) integral

.. math::

    \int_{ { \lbrack 0,1 \rbrack }^n } f(x_1 , \dotsc , x_n) \text{d} \mu (x_1 , \dotsc , x_n)

Numerical integration approximates an integral which cannot be solved analytically as a bunch of sums.

Integrals like these may appear in many different places, from chemistry, through many-body physics, to mathematical finance.

Theory & Implementation in Qrisp
--------------------------------

The implementation in Qrisp requires the implementation of the function acting on a :ref:`QuantumFloat`. We will see how this can be done later in the example.

There is multiple ways to implement Monte Carlo integration in a quantum fashion. In this tutorial, we use an approach based on quantum counting. The idea is to discretize not only the :math:`x`-axis but the :math:`y`-axis as well. We use two ``QuantumFloats`` for this. 
One ``QuantumFloat`` will hold the discretized values of the :math:`x`-axis, while the other one will hold the discretized values of the :math:`y`-axis. 
We can then simply count, in quantum fashion, the number of points under the function curve, and divide it by the number of total points. This is enabled by a ``QuantumBool`` which for each state $\ket{x}\ket{y}$ indicates whether the condition $y<f(x)$ is satisfied.

Don't give up just yet, the mathematical description will bring you more clarity!

For simplicity, we consider the situation where $f\colon [0,1]\rightarrow[0,C]$ is a bounded continuous function of one variable. The strategy presented below can straightforwardly be generalized to the higher-dimensional case.
We wish to evaluate:

.. math::

    \int_{0}^{1}f(x)\mathrm dx

This integral is approximated as `Riemann sum <https://en.wikipedia.org/wiki/Riemann_integral>`_:

.. math::

    \frac{1}{N}\sum\limits_{i=0}^{N-1}f(i/N)

As a first step, we prepare a uniform superposition state in the ``QuantumFloats`` (or any other distibution for the variable discretizing the $x$-axis):

.. math::

    \ket{0} \ket{0} \rightarrow \frac{1}{\sqrt{M \cdot N}} \sum^{N-1}_{i=0} \sum^{M-1}_{j=0} \ket{x_i} \ket{y_j}

where $x_i=i/N$ and $y=j\cdot C/M$.

As a second step, we apply the oracle that evaluates the "points under the curve" condition:

.. math::

    \ket{x_i} \ket{y_j} \ket{\text{False}} \rightarrow \mathbb{1}_{y_j \geq f(x_i)} \ket{x_i} \ket{y_j} \ket{\text{False}} + \mathbb{1}_{y_j < f(x_i)} \ket{x_i} \ket{y_j} \ket{\text{True}}

where $\mathbb{1}_{y_j < f(x_i)}$ is $1$ if $y_j < f(x_i)$ and $0$ otherwise, and similarly for $\mathbb{1}_{y_j\geq f(x_i)}$.

We now arrive at the central step of this algorithm, which is :ref:`Quantum Amplitude Estimation <QAE>`. We use it to find the probability of measuring a good state $\ket{x_i}\ket{y_j}\ket{\text{True}}$, i.e.

.. math::

    p(\{ (x_i,y_j) \mid y_j < f(x_i) \}) = \frac{1}{N} \sum^{N-1}_{i=0} \frac{1}{M} \sum^{M-1}_{j=0}  \mathbb{1}_{y_j < f(x_i)} \approx \frac{1}{N} \sum^{N-1}_{x=0} \frac{f(x_i)}{C}

In the last step, we calculate the ratio between the number of points under the curve and the total number of points $M$ in the interval $[0,C]$. This serves as an approximation for $f(x)/C$.
The last expression is then (up to the scaling factor $C$) an approximation for the integral in question. (For more information on why this is the case check out this `link <https://en.wikipedia.org/wiki/Riemann_integral>`_.)



Iterative Quantum Amplitude Estimation
--------------------------------------

In Qrisp we have the option of using a resource efficient amplitude estimation algorithm, namely `Accelerated Quantum Amplitude Estimation, see Algorithm 1 <https://arxiv.org/pdf/2407.16795>`_ , which iteratively applies :ref:`amplitude amplification <QAE>` to find an estimation for the probability of measureing a good state.
The goal of the algorithm is as follows: 

We start with a unitary operator :math:`\mathcal{A}`, which acts on the input quantum variables as

.. math::

    \textbf{A} \ket{0}\ket{\text{False}} = \sqrt{1-a} \ket{\Psi_0} \ket{\text{False}} + \sqrt{a} \ket{\Psi_1} \ket{\text{True}},

producing a superposition of orthogonal good and bad components where :math:`a \in [0,1]` is unknown.

The algorithm in question allows for us to establish an estimate :math:`\hat{a}` of the unknown :math:`a`. 

Mathematically speaking this means, given an error :math:`\epsilon` and a confidence level :math:`\alpha`, the Accelerated Quantum Amplitude Estimation finds an estimate :math:`\hat{a}` such that

.. math::

    \mathbb{P}\{|\hat{a} - a|\leq\epsilon\}\geq 1-\alpha 

A documentation explaining how to use the Qrisp implementation of this algorithm can found in the :ref:`IQAE <IQAE>` reference.


Example implementation
----------------------

Next up, we will step-by-step go through a example implementation of QMCI tailored to the example of integrating the function $f(x)=x^2$ w.r.t. the uniform distribution over the interval $[0,1]$,
i.e.,

$$\\int_0^1x^2\\mathrm dx$$

A general implementation for integration of multidimensional functions w.r.t. arbitrary probability distributions is provided by the :ref:`QMCI method <QMCI>`.

First, we define the ``function`` that we want to integrate, and a function for preparing the uniform distribution. 
Additionally, we define the variables repesenting the $x$-axis (``qf_x``) and $y$-axis (``qf_x``). 
Thereby, the QuantumFloat representing the $y$-axis must be chosen appropriately with respect to the values that ``function(qf_x)`` assumes.

In this example, we evaluate the function $f(x)$ at $2^3=8$ sampling points as specified by ``QuantumFloat(3,-3)``. 
The resulting values that the function assumes are represented by ``QuantumFloat(6,-6)``.
We also define a ``QuantumBool``, which will indicate the "points under the curve". 

::

    from qrisp import *

    def function(qf):
        return qf*qf

    def distribution(qf):
        h(qf)

    qf_x = QuantumFloat(3,-3)
    qf_y = QuantumFloat(6,-6)

    qbl = QuantumBool()

Now, we arrive at the heart of the algorithm, the definition of the ``state_function``:

::

    @auto_uncompute
    def state_function(qf_x, qf_y, qbl):

        distribution(qf_x)
        h(qf_y)

        with(qf_y < function(qf_x)):
            x(qbl)

It receives the ``@auto_uncompute`` :ref:`decorator <uncomputation>` ensuring that all intermediate variables are properly uncomputed. 
We apply the chosen distribution to ``qf_x``, which represents the :math:`x`-axis. 
As explained earlier, we also discretize the :math:`y`-axis by appling an ``h`` gate to ``qf_y``.

Within a :ref:`ConditionEnvironment`, we then evaluate in superposition which states in ``qf_y`` are smaller than the chosen function evaluated on ``qf_x``.
We store the result of the comparison in the QuantumBool ``qbl``, by applying an ``x`` gate on the previously mentioned QuantumBool if said condition is satisfied.

With everything in place, we can now execute the :ref:`Iterative QAE algorithm <IQAE>`, with a chosen error tolerance ``eps`` and a confidence level ``alpha``.

::

    a = IQAE([qf_x,qf_y,qbl], state_function, eps=0.01, alpha=0.01)

Aaaand that's it! The QMCI is complete! 

Let us now have a look at the result, and compare it to the expected result:

::

    print(a)
    # Yields: 0.27442553839756095

    N = 8
    print(sum((i/N)**2 for i in range(N))/N)
    # Yields: 0.2734375

Congratulations, in this tutorial you learned about the basic theory behind Quantum Monte Carlo Integration, as well as, how to implement it using the high-level concepts that Qrisp offers.
You witnessed the power of quantum computing that allows for evaluation of functions at exponentially many points all at once, 
but also experienced the intricacies of making the quantum computer reveal the solution by using Quantum Amplitude Estimation. 
By doing so, you're diving a step further into the world of quantum algorithms.