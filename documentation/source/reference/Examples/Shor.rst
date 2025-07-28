.. _ShorExample:

Exploring Shor's algorithm
==========================

Shor’s algorithm, named after mathematician Peter Shor, is a quantum algorithm for integer factorization. It’s a way to find the prime factors of a composite number. This algorithm is particularly famous because it is expected solve this problem exponentially faster than the most efficient known classical factoring algorithm, providing a good enough quantum computer. 

This, as you shall see in the example below, has significant implications for cryptography, as many encryption systems rely on the difficulty of factoring really large numbers. So far only numbers up to 15 were crackable utiliziing an implementation of Shor's algorithm - this is where Qrisp comes in. We were able to utilize it's high level architecture and take advantage of it's features to significantly increase this. If provided with enough quality qubits this is the implementation to use to break encryption. 🔥🏦🔥🚒

Factorizing numbers
-------------------

Let's start with a simple example of using Shor's algorithm to factor a number. As stated above, up to our implementation, the highest factorized number was 15. So let's factor number 65!
::

    from qrisp.shor import shors_alg
    shors_alg(65)


Try running the code on the website yourself and feel free to try the algorithm out factorizing different numbers! The result we obtain is 5, which *checks notes* is indeed one of the factors! 

As we will see in the next example, number 65 is easy to crack in terms of the private and public key pairings, which is used for encryption. However, the bacis principles of encryption remain the same even with using much greater numbers.

A tale of encryption and decryption
-----------------------------------

Imagine a scenario where two characters, Alice and Bob, are trying to exchange a secure message. They decide to use RSA encryption, a popular method that uses the product of two prime numbers as part of the key. 
Alice chooses two prime numbers $p=5$ and $q=13$ and calculates their product $N=65$. She then chooses the public exponent $e=11$ and calculates the private exponent $d=35$ following the `key generation protocol <https://www.geeksforgeeks.org/computer-networks/rsa-algorithm-cryptography/>`_.
She publishes the pair $(e,N)=(11,65)$ as the public key, and keeps the pair $(d,N)=(35,65)$ as the private key.
::

    from qrisp.shor import rsa_encrypt_string
    rsa_encrypt_string(e = 11, N = 65 , message = "Qrisp is awesome!")

Enter our detective, let's call him Gadget, who manages to intercept the encrypted message using his highly advanced encrypted-message-interceptor tool. He knows that Alice and Bob have used RSA encryption, but he doesn’t know the private keys they used. "Aaaargh, so close!", he thought.

Luckily for the future of his career as a detective, he remembered that he has recently stumbled upon the website of Eclipse Qrisp where he read the :ref:`enlightening tutorial about Shor's algorithm <tutorial>`. Albeit thinking the text in the tutorial is bordering science fiction, he still decided to give the implementation a go.

His console read:
::

    intercepted_message = '01010000000101011001000101000010100011111101111110001101000010100011010001011001110000100100111010000100001101100010000010100100111110100001'

    from qrisp.shor import rsa_decrypt_string
    rsa_decrypt_string(e = 11, N = 65, ciphertext = intercepted_message)

He ran the command and simply smirked at the result and said "You've got that right, Alice and Bob... Well played!".

*fin*

New adder, no problem
---------------------

Stories like the one above are fun and exciting way to showcase the elegant approach of utilizing Eclipse Qrisp's high level structure. Learning from existing frameworks, however, it is also of utmost importance to ask ourselves the serious, hard hitting question of how to futureproof such an implementation. You've asked the question, we've got the answer - let's look under the hood and delve into the nitty-gritty!

As elaborated on in the :ref:`Fault-Tolerant compilation tutorial <tutorial>`, the Qrisp implementation of Shor's algorithm allows you to provide an arbitrary adder for the execution of the required arithmetic. With our Qrispy structure one can write ones own adder, or implement a shiny new one future research publications might bring, and test its performance claims.

As of right now, the following list of adders have been pre-implemented:

* The :meth:`fourier_adder <qrisp.fourier_adder>` (`paper <https://arxiv.org/abs/quant-ph/0008033>`__) requires minimal qubit overhead and has a very efficient :meth:`custom_control <qrisp.custom_control>` but uses a lot of parametized phase gates, which increases the T-depth. The low qubit count makes it suitable for simulation, which is why it is the default adder.

* The :meth:`cucarro_adder <qrisp.cuccaro_adder>` (`paper <https://arxiv.org/abs/quant-ph/0410184>`__) also requires minimal qubits but no parametrized phase gates. It doesn't have a custom controlled version.

* The :meth:`gidney_adder <qrisp.gidney_adder>` (`paper <https://arxiv.org/abs/1709.06648>`__) requires $n$ ancillae but uses the ``gidney`` Toffoli method described above, making it very fast in terms of T-depth but also economical in terms of T-count.

* The :meth:`qcla <qrisp.qcla>` (`paper <https://arxiv.org/abs/2304.02921>`__) requires quite a lot of ancillae but has only logarithmic scaling when it comes to T-depth. It is faster than the Gidney adder for any input size larger than 7.

Using a diffent adder is as easy as adding an ``inpl_adder`` keyword to the :ref:`QuantumModulus <QuantumModulus>` variable. Literally!

Let's provide an example of benchmarking the :meth:`gidney_adder <qrisp.gidney_adder>` and compare it to the :meth:`qcla <qrisp.qcla>` on the operation most relevant for Shor's algorithm: Controlled modular in-place multiplication.

::

    from qrisp import *
    N = 3295
    qg = QuantumModulus(N, inpl_adder = gidney_adder)
    
    ctrl_qbl = QuantumBool()
    
    with control(ctrl_qbl):
        qg *= 953
        
    gate_speed = lambda op : t_depth_indicator(op, epsilon = 2**-10)
     
    qc = qg.qs.compile(gate_speed = gate_speed, compile_mcm = True)
    print(qc.t_depth())
    # Yields 956
    print(qc.num_qubits())
    # Yields 79    
    
    
Now the :meth:`qcla <qrisp.qcla>`:

::

    qg = QuantumModulus(N, inpl_adder = qcla)
    
    ctrl_qbl = QuantumBool()
    
    with control(ctrl_qbl):
        qg *= 10
        
    qc = qg.qs.compile(workspace = 10, gate_speed = gate_speed, compile_mcm = True)
    
    print(qc.t_depth())s
    # Yields 784
    print(qc.num_qubits())
    # Yields 88   

We see that the T-depth is reduced by $\approx 20 \%$. Due to the logarithmic scaling of the adder, larger scales will profit even more! Note that we granted the compiler 10 qubits of :ref:`workspace <workspace>`, as this adder can profit a lot from this resource.

The comparison analysis is intriguing on its own, but here we wanted to emphasize the simplicity of improving the performance of Shor's algorithm by the means of implementing possible new shiny adders with the least amount of headaches. Future 👏🏻 proven 👏🏻



