.. _shor_tutorial:

Factoring integers using Shor's Algorithm
=========================================

In the realm of quantum computing, where classical limitations are challenged and new horizons are explored, Shor's Algorithm stands as a testament to the transformative potential of quantum mechanics in the field of cryptography. Developed by mathematician Peter Shor in 1994, this groundbreaking algorithm has the power to revolutionize the world of cryptography by efficiently factoring large numbers, once considered an insurmountable task - at least for classical computers.

At its core, Shor's Algorithm addresses one of the cornerstones of modern asymmetric encryption: the difficulty of factoring the product of two large prime numbers. Many cryptographic protocols, including the widely used RSA algorithm, rely on the presumed computational complexity of this task to secure sensitive information. However, Shor's Algorithm exploits the unique properties of quantum computing to perform this factorization exponentially faster than the best-known classical algorithms.

This tutorial aims to demystify the intricacies of Shor's Algorithm by leveraging the programming abstractions that Qrisp provides to boil down the algorithm to only a few lines of code. Please note that this tutorial is not meant to teach the number-theoretical details, but is focussed on how the algorithm can be implemented and compiled. For an in-depth look, please consider the following resources:

* `Nielsen & Chuang textbook <https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview>`_
* `Wikipedia article  <https://en.wikipedia.org/wiki/Shor%27s_algorithm>`_
* `Tutorial paper <https://arxiv.org/pdf/quant-ph/0303175.pdf>`_
* `Original paper <https://arxiv.org/abs/quant-ph/9508027>`_

Our presentation of the theory below will follow the one given by the textbook from Nielsen & Chuang. 


The implementation that you find below is with 11 lines of quantum code not only relatively simple but also one of the :ref:`most efficient <shor_benchmark_plot>` that you can find (you don't find many!).

The general idea
----------------

Given the number to factorize $N \in \mathbb{N}$, the first step of Shor's algorithm is to find the `order <https://en.wikipedia.org/wiki/Multiplicative_order>`_ $r$ of a random number $a \in \mathbb{Z}/N\mathbb{Z}$. Here,

* $a$ is a random number $<N$, which is coprime to $N$ i.e., $\text{GCD}(a, N) = 1$ (otherwise $\text{GCD}(a, N)$ is a factor and we are done already).
* $r$ is the order of $a$ i.e. a number such that $a^r = 1(\text{mod} N)$.
* $a \in \mathbb{Z}/N\mathbb{Z}$ is the `Quotient Ring <https://en.wikipedia.org/wiki/Quotient_ring>`_ of integers modulo $N$.

Assume now, that for a given classical number $x$, we have access to an operator $U_x$, which acts as

.. math::
    U_x \ket{y} = \ket{(xy) \text{mod} N}
    
that is, $U_x$ performs an in-place modular multiplication with $x$ on any quantum number $\ket{y}$.

The next step of the construction is defining the state $\ket{u_s}$ for an arbitrary $s < r$

.. math::
    \ket{u_s} = \frac{1}{\sqrt{r}} \sum_{k = 0}^{r-1} \text{exp}\left(\frac{-2\pi i s k}{r}\right)\ket{x^k \text{mod} N}
    
Applying $U$ to such a state reveals, that it is indeed an eigenvector:

.. math::
    
    \begin{align}
    U \ket{u_s} &= \frac{1}{\sqrt{r}} \sum_{k = 0}^{r-1} \text{exp}\left(\frac{-2\pi i s k}{r}\right) U_x \ket{x^k \text{mod} N}\\
    &= \frac{1}{\sqrt{r}} \sum_{k = 0}^{r-1} \text{exp}\left(\frac{-2\pi i s k}{r}\right) \ket{x^{k+1} \text{mod} N}\\
    &= \text{exp}\left(\frac{2\pi i s}{r}\right) \ket{u_s}
    \end{align}


Therefore if we apply :ref:`quantum phase estimation <QPE>` (with sufficient precision) to $U_x$ and $\ket{u_1}$ we have basically solved the problem:

.. math::

    \text{QPE}_{U_x} \ket{u_1} \ket{0} = \ket{u_1} \ket{\phi_1}
  
Where $\ket{\phi_1}$ is a state that, if measured, is close to $\frac{1}{r}$ with high probability.

An important point here is that we don't need to call the operator $U_x$ for $2^j$ times to achieve $U_x^{2^j}$, which is required for the quantum phase estimation. Instead
we can classically precompute $x^{2^j} (\text{mod}) N$ and use $U_x^{2^j} = U_{x^{2^j}}$.

As you might have noticed however, there is a problem. We would need the solution $r$ to prepare $\ket{u_1}$, so this idea is not feasible. What we can do however
is preparing a superposition of different $\ket{u_s}$.

.. math::

    \frac{1}{\sqrt{r}}\sum_{s = 0}^{r-1} \ket{u_s} = \ket{1}
    
To get an intuition about this equation, it can be helpful to view the mapping $\ket{s} \rightarrow \ket{u_s}$ as the generalized quantum fourier transform over $\mathbb{F}_N$ (the regular $n$ qubit QFT would be $N = 2^n$). If you believe us that this Fourier transform is it's own inverse (maybe up to some signs), then the above equation should be no problem to understand. Otherwise feel free to investigate using one of the mentioned resources.

To commence with our factoring problem we now apply quantum phase estimation to $\ket{1}$.

.. math::

    \begin{align}
    \text{QPE}_{U_x} \ket{1} \ket{0} &= \frac{1}{\sqrt{r}}\sum_{s = 0}^{r-1} \text{QPE}_{U_x} \ket{u_s} \ket{0}\\
    &= \frac{1}{\sqrt{r}} \sum_{s = 0}^{r-1} \ket{u_s} \ket{\phi_s}
    \end{align}

In the next step, we measure the second register to acquire a value $h$ which is close to a number of the form $\frac{s}{r}$. Using the `continued fraction algorithm <https://en.wikipedia.org/wiki/Continued_fraction>`_ we can turn $h$ into a fraction $\frac{\tilde{s}}{\tilde{r}}$. $\tilde{r}$ is therefore a potential candidate for the solution $r$. Finally, we verify our potential solutin using $a^r = 1 (\text{mod}) N$ or (if neccessary) measure another $h$.

The next step is to transform the equation, defining $r$

.. math::
    
    a^r = 1 (\text{mod})N \Leftrightarrow a^r -1 = 0 (\text{mod}) N

Combining this with the definition of the modulus operation, we can see that $N$ must be a factor of $a^r - 1$ (written $N | (a^r - 1)$). If $r$ is even, we can write

.. math::
    a^r - 1 = (a^{r/2} - 1)(a^{r/2} + 1)

On the other hand, if $r$ is odd, the algorithm needs to restart picking a different $a$. Finally, we check whether one of these (for instance $a^{r/2} + 1$) has a common factor with $N$ and, if so, we are done. Otherwise the algorithm needs to restart.

To acquire the final result of the factorization $g$ we compute

.. math::

    g = \text{gcd}(a^{r/2} + 1, N)

Naturally, the other factor is found to be $N/g$.

Implementation
--------------

As you might know by now, the reason why this algorithm can be tricky to implement is because the operator $U_x$ is non-trivial to encode as a quantum circuit and a variety of ideas have been proposed in the past. A popular way of constructing this circuit is to start with a regular in-place adder, build a modular adder from that, build an out-of-place modular multiplyer from that, and finally combine two out-of-place multiplyers into one modular in-place multiplyer. If this wasn't already complicated enough, you also need the controlled version of this circuit. The `approach that we are using here <https://arxiv.org/abs/1801.01081>`_ might also be complicated but the Qrisp abstractions allow for a powerfull reduction in complexity for the user, while still retaining a degree of flexibility for customization and most importantly, PERFORMANCE. 🚀

.. note::
    To make sure you understand everything and get familiar with the concepts, we strongly recommend to execute the code yourself! Either with the Thebe server (might be slow) or on your own device!

The central concept of our implementation is the :ref:`QuantumModulus` class. This :ref:`quantum type<QuantumTypes>` can be used to represent and process elements of a `Quotient Ring <https://en.wikipedia.org/wiki/Quotient_ring>`_, which is basically just a fancy wording for "numbers that operate under modular arithmetic".

>>> from qrisp import *
>>> N = 13
>>> qg = QuantumModulus(N)
>>> qg[:] = 8

This snippet creates such a :ref:`QuantumVariable` with modulus $N = 13$. Subsequently we encode the value 8. The defining feature of this type is the fact that arithmetic is always modular.

>>> qg += 8
>>> print(qg)
{3: 1.0}

We can take a look at the quantum circuit:

>>> print(qg.qs)

As you can see under the hood, there is a lot of complexity, however due to systematic development and testing, a lot of it can be hidden from the user. Feel free to try out the in-place multiplication ``*=``!

Using the :ref:`QuantumModulus` class allows us to implement Shor's algorithm within a few lines of code! For a simple example, let us factor $N = 99$ using $a = 10$.

>>> N = 99
>>> a = 10
>>> qg = QuantumModulus(N)
>>> qg[:] = 1

According to `literature <https://www.cambridge.org/de/universitypress/subjects/physics/quantum-physics-quantum-information-and-quantum-computation/quantum-computation-and-quantum-information-10th-anniversary-edition>`_, a quantum phase estimation precision of $2n+1$ is sufficient, where $n$ is the bit-width of $N$.

>>> n = qg.size
>>> qpe_res = QuantumFloat(2*n+1, exponent = -(2*n+1))
>>> h(qpe_res)

Note the ``exponent`` keyword of the :ref:`QuantumFloat` constructor. It indicates that this :ref:`QuantumFloat` can represent numbers up to the precision $2^{-(2n+1)}$. After construction, we apply a Hadamard gate to each qubit as is customary in :ref:`quantum phase estimation <qpe_tutorial>`.

We can now code the main loop:

::
    
    x = a
    for i in range(len(qpe_res)):
        with control(qpe_res[i]):
            qg *= x
            x = (x*x)%N

Note that we have $x=a^{2^i}$ at the i-th iteration. Such a procedure is called repeated squaring and reduces the classical resources for the computation of $x$ in each iteration.

Finally, we conclude the phase estimation with the inverse quantum Fourier transformation and perform a measurement.

>>> QFT(qpe_res, inv = True)
>>> meas_res = qpe_res.get_measurement()
>>> print(meas_res)
{0.0: 0.5, 0.5: 0.5}

To perform the continued fraction step, we can use some `sympy tools <https://docs.sympy.org/latest/modules/ntheory.html>`_:

:: 

    from sympy import continued_fraction_convergents, continued_fraction_iterator, Rational
    
    def get_r_candidates(approx):
        rationals = continued_fraction_convergents(continued_fraction_iterator(Rational(approx)))
        return [rat.q for rat in rationals]

This function takes an approximation value ``approx`` and calculates fractionals of the form $\frac{p}{q}$ that are increasingly close to ``approx``. To extract our results for the $r$ values, we are interested in the $q$ part of each fractional.

>>> r_candidates = sum([get_r_candidates(approx) for approx in meas_res.keys()], [])

To find the correct $r$, we perform a classical search on our results

::
 
    for cand in r_candidates:  
        if (a**cand)%N == 1:
            r = cand
            break
    else:
        raise Exception("Please sample again")
    
    if r % 2:
        raise Exception("Please choose another a")


The final step in acquiring the factor is computing the greatest common divisor of $a^{r/2 + 1}$, which can be `done efficiently <https://w.wiki/znj>`_

>>> import numpy as np
>>> g = np.gcd(a**(r//2)+1, N)
>>> print(g)
11

Aaaaand we are done! ⏲️

To highlight once more how much Qrisp simplifies the construction, we summarize the code of the quantum subroutine in a single function:

::

    def find_order(a, N):
        qg = QuantumModulus(N)
        qg[:] = 1
        qpe_res = QuantumFloat(2*qg.size + 1, exponent = -(2*qg.size + 1))
        h(qpe_res)
        for i in range(len(qpe_res)):
            with control(qpe_res[i]):
                qg *= a
                a = (a*a)%N
        QFT(qpe_res, inv = True)
        return qpe_res.get_measurement()

11 lines - feel free to compare with other implementations!

To learn how to compile this algorithm optimized for fault-tolerant backends and deploy an exponentially faster adder, make sure to check out :ref:`the next tutorial<ft_compilation>`!