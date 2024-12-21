.. _QMCItutorial:

Quantum Monte Carlo Integration with Iterative QAE
==================================================

This tutorial will provide you with an introduction to Quantum Monte Carlo Integration within Qrisp.

For this purpose, we will first give you a theoretical overview of what this technique is about and where it is used. 
Then we will dive into the practical implemention within Qrisp. This also includes the usage of :ref:`Iterative Quantum Amplitude Estimation <IQAE>`. 
To finish of this tutorial, we investigate the full implementation of a simple example by integrating :math:`f(x)=x^2` over a uniform distribution in the interval :math:`\lbrack 0,1 \rbrack`.

The relevant literature can be found in the following papers: `A general quantum algorithm for numerical integration <https://www.nature.com/articles/s41598-024-61010-9>`_ and `Option pricing using Quantum computers <https://arxiv.org/pdf/1905.02666>`_ for QMCI and `Accelerated Quantum Amplitude Estimation
without QFT <https://arxiv.org/pdf/2407.16795>`_ for IQAE.

Theoretical overview of QMCI
----------------------------

QMCI tackels the same problems as its classical counterpart: Numerical integration of high-dimensional functions over probility distributions.

Mathemically speaking, we want to find an approximation for the following (general) integral

.. math::

    \int_{ { \lbrack 0,1 \rbrack }^n } f(x_1 , ... , x_n) \text{d} \mu (x_1 , ... , x_n)

As one does, we approximate an integral which cannot be solved analytically as a bunch of sums.

Integrals like these may appear in many different places, from chemistry, through many-body physics, to mathematical finance.

Implementation in Qrisp
-----------------------

The implementation in Qrisp requires the implementation of the function on a :ref:`QuantumFloat`. We will see how this can be done later in the example. 

There is multiple ways to implement Monte Carlo integration in a quantum fashion. Within Qrisp, we use an approach based on quantum counting. The idea is to discretize not only the :math:`x`-axis but the :math:`y`-axis as well. We use two ``QuantumFloats`` for this. 
One ``QuantumFloat`` will hold the discretized values of the distribution, i.e., the relevant support on the :math:`x`-axis, while the other one will hold the discretized values of the :math:`y`-axis which we will encode the function values.
We can then simply count, in quantum fashion, the number of points under the function curve, and divide it by the number of total points. This is facilitated by a ``QuantumBool`` which for each state $\ket{x}\ket{y}$ indicates whether the condition $y<f(x)$ is satisfied.

Don't give up just yet, the mathematical description will bring you more clarity!

We start with a ``state_function`` that, as a first step, encodes the ``QuantumFloats`` as follows:

.. math::

    \ket{0} \ket{0} \rightarrow \frac{1}{\sqrt{M \cdot N}} \sum^{N-1}_{x=0} \sum^{M-1}_{y=0} \ket{x} \ket{y}

As a second step, it applies the oracle that encodes the "points under the curve" condition as follows:

.. math::

    \ket{x} \ket{y} \ket{\text{False}} \rightarrow \mathbb{1}_{y \geq f(x)} \ket{x} \ket{y} \ket{\text{False}} + \mathbb{1}_{y < f(x)} \ket{x} \ket{y} \ket{\text{True}}

We now arrive at the central step of this algorithm, which is :ref:`Quantum Amplitude Estimation <QAE>`. We use it to find

.. math::

    p(\{ (x,y) \mid y < f(x) \}) = \frac{1}{N} \sum^{N-1}_{x=0} \frac{1}{M} \sum^{M-1}_{y=0}  \mathbb{1}_{y < f(x)} \approx \frac{1}{N} \sum^{N-1}_{x=0} \frac{f(x)}{M}

The last expression is then (up to a scaling factor) the approximation for the integral in question. 


Iterative Quantum Amplitude Estimation
--------------------------------------

In Qrisp we have the option of using a resource efficient amplitude estimation algorithm, namely `Accelerated Quantum Amplitude Estimation, see Algorithm 1 <https://arxiv.org/pdf/2407.16795>`_ , which iteratively applies :ref:`amplitude amplification <QAE>` to find an estimation for the amplitude of the good state.
The goal of the algorithm is as follows: 

We start with a unitary operator :math:`\mathcal{A}`, which acts on the input QuantumVariables as

.. math::

    \textbf{A} \ket{0}\ket{\text{False}} = \sqrt{1-a} \ket{\Psi_0} \ket{\text{False}} + \sqrt{a} \ket{\Psi_1} \ket{\text{True}},

where :math:`a \in [0,1]` is unknown.

The algorithm in question allows for us to establish an estimate :math:`\hat{a}` of the unknown :math:`a`. 

Mathematically speaking this means, given an error :math:`\epsilon` and a confidence level :math:`\alpha`, the Accelerated Quantum Amplitude Estimation finds an estimate :math:`\hat{a}` such that

.. math::

    \mathbb{P}\{|\hat{a} - a|\leq\epsilon\}\geq 1-\alpha 

A documentation explaining how to use the Qrisp implementation of this algorithm can found in the :ref:`IQAE <IQAE>` reference.


QMCI example implementation
---------------------------

Next up, we will step-by-step go through a full example implementation of QMCI tailored to the example of integrating the function $f(x)=x^2$ over the uniform distribution in the interval $[0,1]$.
A general implementation for integration of multidimensional functions is provided by the :ref:`QMCI <QMCI>` function.

First, we define the function that we want to integrate, and a function for preparing the uniform distribution. 
Additionally, we define a list of variables ``qargs`` repesenting the $x$-axis (``qargs[0]``) and $y$-axis (``qargs[1]``). 
Thereby, the QuantumVariable representing the $y$-axis has to be chosen appropriately with respect to the values that the result of ``function(qargs[0])`` assumes.

::

    from qrisp import *

    def function(qf):
        return qf*qf

    def distribution(qf):
        h(qf)

    qargs = [QuantumFloat(2,-2), QuantumFloat(4,-4)]

Second, we determine the correct scaling factor by calculating the volume of the hypercube spanned by the intervals for the $x$-axis and $y$-axis.

We also append a ``QuantumBool`` to our input ``qargs``, which will indicate the "points under the curve". 

::

    V0=1
    for arg in qargs:
        V0 *= 2**(arg.size+arg.exponent)
    
    qargs.append(QuantumBool())

Now, we arrive at the heart of the algorithm, the definition of the ``state_function``:

::

    @auto_uncompute
    def state_function(*args):
        qf_x = args[0]
        qf_y = args[1]
        tar = args[2]

        distribution(qf_x)
        h(qf_y)

        with(qf_y < function(qf_x)):
            x(tar)

It receives the ``@auto_uncompute`` :ref:`decorator <uncomputation>` ensuring that all intermediate variables are properly uncomputed. 
We apply the chosen distribution to ``qf_x``, which represents the :math:`x`-axes support. 
As explained earlier, we also discretize the :math:`y`-axis by appling an ``h`` gate to ``qf_y``.
We then evaluate in superposition which states in ``qf_y`` are smaller than the chosen function evaluated on ``qf_x``.

We store the result of the comparison in the QuantumBool ``tar``, by applying a ``cx`` gate on the previously mentioned QuantumBool.

With everything in place, we can now execute the :ref:`Iterative QAE algorithm <IQAE>`, with a chosen error tolerance ``eps`` and a confidence level ``alpha``.
We also have to rescale with the previously calculated volume ``V0``.

::

    a = IQAE(qargs, state_function, eps=0.01, alpha=0.01) 
    V = V0*a

Aaaand that's it! The QMCI is complete! 

Let us now have a look at the result, and compare it to the expected result:

::

    >>> V
    0.21855991519015455

    >>> (0+0.25**2+0.5**2+0.75**2)/4
    0.21855991519015455
