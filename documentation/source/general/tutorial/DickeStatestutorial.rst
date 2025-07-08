.. _DickeStatestutorial:

Preparation of Dicke States
===========================

In this tutorial we will show you different routines for the preparation of Dicke States and how to implement them in Qrisp! They are all JASP [REF] compatible, so you can seamlessly integrate them into your realtime computations as a subroutine. 

The two main approaches we will look at are [REF, YU] and [REF Bärtschi], as well as the more efficient [REF SCS] implementation.   

But before we jump into their preparation, let us first answer the questions: **What are Dicke states? And what do we use them for?**


Definition and utility of Dicke States
======================================

Formally, Dicke states are defined as follows


The preparation of Dicke states is an important and highly investigated topic in the field of quantum computation due to to their range of applications. 
This ranges from error correction techniques in the ISD algorithm to optimization tasks solved with DQI or QAOA. 
Furthermore, these states are native to quantum optics systems due to their relation to spin systems, 
and can be found quantum communication applications. 

Dicke states are characterized by there constant Hamming-weight. A Dicke state with :math:`n` qubits and Hamming-weight :math:`\text{wt}(x) =k` is defined as

.. math::

    \ket{D_{k}^{n}} = \binom{n}{k}^{-\frac{1}{2}} \sum_{x \in \{ 0,1 \}^{n}}, \. \text{wt}(x) = k } \ket{x} 

An example for this is the Dicke state :math:`\ket{D_{2}^{4}} = \frac{1}{\sqrt{6}} ( \ket{1100} + \ket{1010} + \ket{1001} +\ket{0110} + \ket{0101} + \ket{0011} )`, consisting of 4 qubits with Hamming-weight :math:`\text{wt}(x) =2`.  
It can also be defined in direct relation to a spin system, see p.e. the related wikipedia article [REF].


Preparation algorithms
======================

After the short overview on their definition and applications lets now dive into the established algorithms for Dicke state preparation! 

Probablistic approach
---------------------

The first algorithm we want to highlight here is `Efficient preparation of Dicke states (2024) <https://arxiv.org/abs/2411.03428>`_ by Jeffery Yu et al. from the University of Maryland. 
The algorithm utilizes mid-circuit Hamming-weight measurements and feedback to prepare Dicke states, by incorporating  adaptively-chosen global rotations. It is furthermore motivated by its nativity to a cavity QED context, where both the mid-circuit measurement and rotations 
are easily implemented in a physical system.

But as he knows best, we will let Jeffery explain it himself:

[Theoretical Intro and physical motivation]


--> Pseudo-Algo
--> physical intuition behind rotation in Bloch-Sphere, I.e. rotate towards desired overlap
--> quick idea of where the optimal rotation angle comes from 
--> Hamming weight measurement and short explanation of nativity to cavity QED system (maybe for this also further explanation of spin manipulations in this system?)

Implementation 
--------------

For the implementation we will resort back to circuit representation of the aforementioned components, namely the Hamming-weight measurements and the rotations around the Bloch sphere. 

As explained in the paper, one of the facilitators for this algorithm is the collective Hamming-weight measurement, so let us start by creating that. 
The paper `Shallow Quantum Circuit Implementation of Symmetric Functions with Limited Ancillary Qubits (2024) <https://arxiv.org/pdf/2404.06052>`_ gives a straight forward 
scheme. We first create the necessary amount of ancilla qubits, and then apply the routine based on controlled rotations and an inverse Quantum Fourier transformation [REF].

:: 

    def collective_hamming_measurement(qf, n):

        # create ancillas and put the in superposition
        n_anc = jnp.ceil(jnp.log2(n)+1).astype(int)
        ancillas = QuantumFloat(n_anc)
        h(ancillas)

        # controlled rz rotations 
        for i in jrange(n_anc):
            rz(2*n*jnp.pi/(2**(n_anc+1-i)),ancillas[i])
            
        for i in jrange(n_anc):
            for k in jrange(n):
                cx(ancillas[n_anc-i-1], qf[k])
            for k in jrange(n):
                rz(-2*jnp.pi/(2**(i+2)), qf[k])
            for k in jrange(n):
                cx(ancillas[n_anc-i-1], qf[k])

        # inverse QFT for correct representation
        QFT(ancillas,inv=True)
        
        return ancillas


With that out of the way, lets get back to the implementation of Jeffery's algorithm. The implementation is a straight forward code version of the pseudo code provided.
It is already jaspified [REF] and intended to be called with the ``@terminal_sampling`` decorator (or an adaption with makes use of the ``terminal_sampling`` simulator optimization).

To reiterate: the procedure performs iterative ``ry``-rotations, where the rotation angle is adaptively chosen based on the ``collective_hamming_measurement`` of previous iteration.
We stop, once we measure the correct Hamming-weight. 
In JASP-terms this is achieved by wrapping the "rotate-and-measure" procedure in a q_while_loop [REF]. This jaspified version of a quantum while-loop requires a condition function ``cond_fun`` with a ``bool`` (or ``QuantumBool``) return, and a body function ``body_fun``.
The ``cond_fun`` checks whether the "while" condition is still true, while the ``body_fun`` performs the iterative quantum operations.

Let us investigate the ``body_fun`` first. We indeed see that we directly translate what Jeffery proposes into code. First we perform some arithmetic to find the updated rotation angle.
Then, we apply the corrected ``ry``-rotations. And finally, we perform the ``collective_hamming_measurement`` to gather information about our Hamming-weight overlap. 

::

    def body_fun(val):
        # assign initial values
        m_t, qf1, theta, j, m = val
        # algebra from paper
        r_m = jnp.sqrt(j * (j+1) - m.astype(float) **2)
        theta = jnp.asin((m * r_mt - m_t.astype(float) * r_m) /r_0**2)

        # rotation towards desired state
        for t in jrange(j):
            ry(theta, qf1[t])

        # collective hamming weight measurement and uncomputation
        ancillas = collective_hamming_measurement(qf1,j)
        m = measure(ancillas)
        # delete ancillas
        ancillas.delete()

        return m_t, qf1,theta ,j, m 



The ``cond_fun`` is very simple. All it does, is to check whether the result from the Hamming-weight measurement (described by ``val[-1]``) 
is equivalenant to the one we are looking for (which is given by ``val[0]``). If yes, we stop the loop.

::

    def cond_fun(val):
        return val[0] != val[-1]

Putting it all together, the main function ``iterative_dicke_state_sampling`` reduces to five lines of code, with the ``q_while_loop`` [REF] being the central ingredient.

::

    def iterative_dicke_state_sampling(qf, m_t):
        
        j = qf.size 

        # algebra from paper for initial values
        r_mt = jnp.sqrt(j*(j+1)-m_t**2)
        r_0 = jnp.sqrt(j*(j+1))

        # insert jasp body_fun from above here
        
        # insert jasp cond_fun from above here

        thet_0 = 0
        
        m_t, qf1, thet_0, j, m  = q_while_loop(cond_fun, body_fun, (m_t, qf,thet_0 ,j,j))
        
        return qf1


To give an final example, this what the code looks like to create the aforementioned :math:`\ket{D_{2}^{4}}` state looks like:

::

    #We initiate a QuantumVariable with 4 qubits from this create the Dicke state with Hamming weight 2 in JASP mode.
    @terminal_sampling
    def main():
            
        n = 4
        k = 2
        qv_iter = QuantumFloat(n)
        qv_iter = iterative_dicke_state_sampling(qv_iter,k)

        return qv_iter

    dicke_qv = main()
    
And thats it! All you need to create a Dicke state in JASP mode. 

Let us now continue with the deterministic approach


Deterministic approach 
----------------------

The other algorithm of interest is `Deterministic Preparation of Dicke States (2024) <https://arxiv.org/abs/1904.07358>`_ and its more efficient variation `A Divide-and-Conquer Approach to Dicke State preparation (2021) <https://arxiv.org/abs/2112.12435>`_. 

The second algorithm mentioned is a divide-and-conquer adaption based on the first one, as the name would suggest. So let us start with the first paper. 

In it the authors make use of *split & cyclic shift* unitaries, which are then applied inductively in a cascade. In the following, we will show you how 
the basic components and how these unitaries are structed in terms of Qrisp code.

For an indepth explanation on how these unitaries emerge and their action on a quantum state, please refer to the original paper. 

The aforementioned unitary is given by the function ``split_cycle_shift``, which receives a QuantumVariable ``qv`` on which it is a applied. 
Additionally, two ``int``s ``highIndex`` and ``lowIndex`` indicate the preparation steps, as seen in original algorithm.

Some caveats: 

This implementation is JASP ready. It therefore makes use the ``jrange`` iterator. In the paper, the iteration is conducted in reverse, i.e. from the lowest to the highest index. 
In a normal ``range`` iterator you would just set ``step =-1`` for this behaviour, ``jrange`` does not allow for this. Instead we embed the whole construct in an ``invert()``-statement to reverse the loop.

Additionally you may notice some logic checks using the ``ctrl_bool``s. This replaces ``if``-statement usage in JASP mode, so make good use of that when **jaspifying** your Qrisp code! 

::

    def split_cycle_shift(qv, highIndex, lowIndex):

        with invert():
            # reversed jrange
            for i in jrange(lowIndex): 

                index = highIndex - i 
                param = 2 * jnp.arccos(jnp.sqrt((highIndex - index + 1 ) /(highIndex)) )

                ctrL_bool = index == highIndex
                ctrL_bool_false = index != highIndex

                # conditional application of the cx and c-ry rotations 
                with control(ctrL_bool):
                    cx(qv[highIndex - 2], qv[highIndex-1]) 
                    with control( qv[highIndex-1] ):
                        ry(param, qv[highIndex - 2])
                    cx(qv[highIndex - 2], qv[highIndex -1])
                
                with control(ctrL_bool_false):
                    cx(qv[index -2], qv[highIndex-1]) 
                    with control([qv[highIndex -1],qv[index -1]]):
                        ry(param, qv[index - 2])
                    cx(qv[index -2], qv[highIndex-1]) 


These *split & cyclic shift* unitaries are embedded in the main function **dicke_state**. It receives as inputs the QuantumVariable ``qv`` that we want to work on and an integer ``k`` which represents the desired Hamming-weight.
Here we again invert the ``jrange`` operator to represent the logic of the original paper.


::
        
    def dicke_state(qv,k):

        # jasp compatibility
        if check_for_tracing_mode():
            n = qv.size
        else:
            n = len(qv)

        # SCS cascade
        with invert():
            for index2 in jrange(k+1, n+1):
                split_cycle_shift(qv, index2, k,)
            #barrier(qv)
        with invert():
            for index in jrange(2,k+1):
                split_cycle_shift(qv, index, index-1, )
            #barrier(qv)

Usage
-----

To run this code and properly generate the desired Dicke state, we have to make sure that the input state already has the desired Hamming-weight ``k`` in its trailing ``k`` qubits.

So in other words, to receive :math:`\ket{D_{2}^{4}}` from calling ``dicke_state(qv,2)``, the ``qv`` has to in the :math:`\ket{0011}` state! 

We can therefore execute the following code:

::
    
    from qrisp import QuantumVariable, x, dicke_state
    # create the qv and put it in |0011> state
    qv = QuantumVariable(4)
    x(qv[2])
    x(qv[3])
    # call the dicke_state function
    dicke_state(qv, 2)
    # receive Dicke state with wt == 2

While this may be seen as an inhibition to the algorithm, this actually leads to some very useful behaviour;
The unitary which prepares :math:`\ket{D_{2}^{4}}` from :math:`\ket{0011}`, lets name it :math:`U_{2}^{4}`, also creates :math:`\ket{D_{1}^{4}}` from :math:`\ket{0001}`!

More generally, a unitary :math:`U_{k}^{n}`, which creates a given Hamming-weight :math:`k` state with :math:`n` total qubits, will also create any lower Hamming-weight state from the correct input state.

Mathematically speaking this means, with :math:`n` being a given number of qubits, :math:`k` a given Hamming-weight, and any other :math:`l \leq k`. 

[REF, fix math below --> cdot to tensor]
.. math::

    U_{k}^{n} \ket{0}^{n-k} \cdot \ket{1}^{k} = \ket{D_{k}^{n}} \text{ and } U_{k}^{n} \ket{0}^{n-l} \cdot \ket{1}^{l} = \ket{D_{l}^{n}}


This is particularly useful for creating superpositions of different Hamming-weight Dicke states (see for example the DQI algorithm [REF]). Consider the following example, where :math:`\alpha \in (0,1)` 

.. math::

    U_{2}^{4} ( \sqrt{\alpha} \ket{0011} + \sqrt{1- \alpha} \ket{0001}  = \sqrt{\alpha} \ket{D_{4}^{2}} + \sqrt{1-\alpha} \ket{D_{4}^{1}} 

Accordingly we can execute the function from above on a QuantumVariable in superposition to receive the dicke state in superposition!


::
    
    from qrisp import QuantumVariable, x, dicke_state
    # create the qv and put it in |0011> state
    qv = QuantumVariable(4)
    x(qv[2])
    h(qv[3])
    # call the dicke_state function
    dicke_state(qv, 2)
    # receive superposition of Dicke states with weight 1 and 2!


Divide-and-Conquer approach
---------------------------

For the final algorithm in this tutorial let us investigate the `divide-and-conquer approach from Bärtschi et al. <https://arxiv.org/abs/2112.12435>`_

The idea here is to divide the whole Dicke state preparation procedure as follows: 

First we separate the set of qubits into two.
Then a smart prepreparation is conducted, after which the ``dicke_state``-function is executed on each qubit set individually.
Finally, we fuse the qubit sets back together.

The main difficulty lays in choosing the correct weighting of states for the preparation step. For an indepth explanation please refer to the original paper.
We will also make use of the function ``comb``, a JAX [REF] compatible version of the binomial coeffient.

::

    @jax.jit
    def comb(N, k):
        integ = jnp.uint16(jnp.round(jnp.exp(gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1))))
        return integ

In the following we will keep it short. The ``dicke_divide_and_conquer_jasp`` function precomputes the correct weights, i.e. the ``ry``-gate angles to fan-out 
the amplitude information, and then applies a ``cx``-cascade. 
Afterwards we apply the ``dicke_state`` functions on the separted qubit set.
For the explanation of the ``ry``-angle calculation we refer to the original paper. 

::

    def dicke_divide_and_conquer_jasp(qv, k):

        # separate the QuantumVariable
        n = qv.size
        n_1 = jnp.floor(n/2)
        n_2 = n - n_1

        # divide step
        def dicke_divide(qv):
            l_xi = []
            rotation_angles = jnp.zeros(k)
            l_xi = jnp.zeros(k+1)

            # compute rotation angles
            for i1 in range(k+1):
                x_i = comb(n_1,i1)*comb(n_2,k-i1)
                l_xi = l_xi.at[i1].set(x_i)

            for i2 in range(k):
                temp_sum = jnp.sum(l_xi[i2:])
                rot_val = 2*jnp.acos(jnp.sqrt(l_xi[i2]/temp_sum))
                rotation_angles = rotation_angles.at[i2].set(rot_val)
            
            n_1h = n_1.astype(int)
            # apply the rotations
            ry(rotation_angles[0], qv[n_1h-1])
            # fan-out
            for i in range(1,k):
                with control(qv[n_1h-i]):
                    ry(rotation_angles[i], qv[n_1h-i-1])
            
            x(qv[n-k:n])
            for i in range(k):
                cx(qv[n_1h-k+i], qv[-(i+1)])

        # call the divide step and the two conquer (dicke_state) steps.
        dicke_divide(qv)
        #barrier(qv)
        n_1a = n_1.astype(int)
        n_2a = n_2.astype(int)
        dicke_state(qv[:n_1a], k)
        #barrier(qv)
        dicke_state(qv[n-n_2a:], k)
        #barrier(qv)    


An that's it! You have reached the end of tutorial and are now ready to prepare Dicke States with all of the state-of-the-art methodology!