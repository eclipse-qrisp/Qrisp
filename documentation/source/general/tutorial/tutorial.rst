.. _Qrisp101:

Getting familiar with Qrisp
---------------------------
Congratulations, you made it to your first Qrisp tutorial! This tutorial is designed to provide you with a hands-on understanding of how to use Qrisp for all things quantum. In this tutorial, you will create a QuantumVariable, solve a quadratic equation using Grover’s algorithm, and delve into the Quantum Phase Estimation algorithm. 

We’ll start by introducing you to the concept of a QuantumVariable and how it forms the basis of quantum computing in Qrisp. Next, we’ll explore Grover’s algorithm, a quantum algorithm that provides a quadratic speedup for unstructured search problems. You’ll learn how to construct an oracle and use it within Grover’s algorithm to solve a quadratic equation. Finally, we’ll visit the Quantum Phase Estimation algorithm, a key subroutine in many other quantum algorithms and an essential tool in quantum computing.

By the end of this tutorial, you’ll have a solid understanding of Qrisp's fundamental concepts and be well-equipped to tackle more complex problems and implement the Quantum Approximate Optimization Algorithm with some Qrisp exclusive mixers using our framework.

Creating a QuantumVariable
==========================
The central building block of Qrisp algorithms is the :doc:`/./reference/Core/QuantumVariable`. A QuantumVariable hides the qubit management from the user, enables human readable in- and outputs, strong typing via class inheritance, infix arithmetic syntax and much more. Creating a QuantumVariable is simple:

>>> from qrisp import QuantumVariable
>>> qv = QuantumVariable(5)

Here, 5 refers to the number of qubits the QuantumVariable represents.
QuantumVariables allow for convenient quantum function construction and evaluation, since a QuantumVariable carries all the information that is relevant for performing quantum operations on it.

QuantumVariables can be low-level manipulated by calling :ref:`gate application functions <gate_application_functions>` on them:
  
>>> from qrisp import h, z, cx
>>> h(qv[0])
>>> z(qv)
>>> cx(qv[0], qv[3])

Note that the Z gate is applied to all qubits of ``qv`` since there is no qubit specified.

In order to take a look at the generated circuit, we call ``print`` on the :ref:`QuantumSession` in which ``qv`` is registered:

::

    print(qv.qs)
    

::

    QuantumCircuit:
    ---------------
          ┌───┐┌───┐     
    qv.0: ┤ H ├┤ Z ├──■──
          ├───┤└───┘  │  
    qv.1: ┤ Z ├───────┼──
          ├───┤       │  
    qv.2: ┤ Z ├───────┼──
          ├───┤     ┌─┴─┐
    qv.3: ┤ Z ├─────┤ X ├
          ├───┤     └───┘
    qv.4: ┤ Z ├──────────
          └───┘          
    Live QuantumVariables:
    ----------------------
    QuantumVariable qv

Once the QuantumVariable is not needed anymore, we can call the :meth:`delete<qrisp.QuantumVariable.delete>` method, to tell the qubit manager, to free up the corresponding qubits. Calling :meth:`delete<qrisp.QuantumVariable.delete>` enables the qubits of ``qv`` to be reused at a later point for other purposes.

>>> qv.delete(verify = True)
Exception: Tried to delete QuantumVariable which is not in |0> state.

If given the keyword argument ``verify`` , Qrisp will check if the deleted qubits are properly disentangled by querying a simulator. Note that ``verify`` is set to ``False`` by default as the simulation can be resource costly for algorithms, which are creating and destroying alot of QuantumVariables.

In this case, the qubits are not ready to use for other purposes as they still are in a non-trivial state. If ``qv`` is entangled to other qubits, a simple reset would result in a non-unitary collapse of superposition. We would need to perform a procedure which is commonly called :ref:`uncomputation`.

QuantumVariables can be thought of as the abstract parent class of more special types. One example is the :doc:`/./reference/Quantum Types/QuantumChar`:

>>> from qrisp import QuantumChar
>>> qch = QuantumChar()

In order to initialize ``qch``, we use the slicing operator (which invokes the :doc:`encode</./reference/Core/generated/qrisp.QuantumVariable.encode>` method):

>>> qch[:] = "e"

We can check the content using a simple ``print`` call:

>>> print(qch)
{"e": 1.0}

This command queries a simulator which evaluates the compiled quantum circuit. The measurement results are returned as bitstrings, which are then converted to the corresponding outcome value. Here, the 1.0 corresponds to the probability of the outcome ``"e"``.
In order to bring some quantumness into the script, we can entangle it to our previously created QuantumVariable

>>> cx(qv[0], qch[0])
>>> print(qch)
{'e': 0.5, 'f': 0.5}

This brings the 0-th qubit of ``qch`` into a superposition and therefore ``"f"`` now appears with 50% probability.

If we want to apply further processing to the measurement results, we can retrieve them as a dictionary using the :meth:`get_measurement<qrisp.QuantumVariable.get_measurement>` method:
  
>>> results = qch.get_measurement()

To investigate the statevector, we call the :meth:`statevector <qrisp.QuantumSession.statevector>` method of the :ref:`QuantumSession`:

>>> qch.qs.statevector()
sqrt(2)*(|00000>*|e> - |10010>*|f>)/2

If you have Sympy's `pretty printing <https://docs.sympy.org/latest/tutorials/intro-tutorial/printing.html>`_ enabled in your console, you will even receive a nice LaTeX rendering:

.. image:: ./tutorial_statevector.png
   :width: 200
   :alt: Tutorial statevector
   :align: left

|

Qrisp has full compatibility to Qiskit featuring convenient :meth:`importing <qrisp.QuantumCircuit.from_qiskit>` and :meth:`exporting <qrisp.QuantumCircuit.to_qiskit>` of Qiskit circuits:

>>> qiskit_qc = qch.qs.compile().to_qiskit()

It is also possible to run Qrisp code directly on IBM Q hardware using a :ref:`VirtualQiskitBackend`.

>>> from qiskit_ibm_provider import IBMProvider
>>> provider = IBMProvider(YOUR_APITOKEN)
>>> kolkata_qiskit = provider.get_backend("ibm_lagos")
>>> from qrisp import VirtualQiskitBackend
>>> kolkata_qrisp = VirtualQiskitBackend(kolkata_qiskit)
>>> results = qch.get_measurement(backend = kolkata_qrisp)
>>> print(results)
{'e': 0.4544, 'f': 0.4492, 'g': 0.0269, 'h': 0.0261, 'm': 0.0173, 'n': 0.0142, 'a': 0.0037, 'b': 0.0035, 'u': 0.0012, 'v': 0.0012, 'p': 0.0008, 'o': 0.0006, 'd': 0.0002, 'j': 0.0002, 'x': 0.0002, 'c': 0.0001, 'i': 0.0001, '?': 0.0001}

And that's it - you're set with the basics and ready to build some algorithms!

Solving a quadratic equation using Grover's algorithm
-----------------------------------------------------

As a first example, we showcase how to solve the quadratic equation

.. math::

   x^2 = 0.25

using Grover's algorithm. The idea here is to prepare an oracle, that multiplies a :doc:`/./reference/Quantum Types/QuantumFloat` with itself and tags the desired value $c_{tag} = 0.25$. This oracle is then embedded into several Grover iterations to amplify the amplitude of the solution.

Oracle Construction
===================

We start with elaborating the oracle construction: ::

    from qrisp import auto_uncompute, z, h, QuantumFloat

    @auto_uncompute
    def sqrt_oracle(qf):
        temp_qbool = (qf*qf == 0.25)
        z(temp_qbool)
       


This oracle recieves a :ref:`QuantumFloat` ``qf`` and evaluates the square. Subsequently it determines wether the result is equal to 0.25, which returns the :ref:`QuantumBool` ``temp_qbool``. Finally, we perform a Z gate on ``temp_qbool``. Note the ``auto_uncompute`` decorator, which automatically uncomputes all temporary values of this function (ie. the result of the multiplication and ``temp_qbool``). You can find more information about Qrisps automatic uncomputation in  :ref:`uncomputation`.

To inspect the circuit, we create a :ref:`QuantumFloat`, evaluate the oracle and call ``print`` on the ``.qs`` attribute

>>> qf = QuantumFloat(3, -1, signed = True)
>>> sqrt_oracle(qf)
>>> print(qf.qs)

::

    QuantumCircuit:
    --------------
                 ┌───────────┐               ┌──────────────┐
         qf_0.0: ┤0          ├───────────────┤0             ├
                 │           │               │              │
         qf_0.1: ┤1          ├───────────────┤1             ├
                 │           │               │              │
         qf_0.2: ┤2          ├───────────────┤2             ├
                 │           │               │              │
         qf_0.3: ┤3          ├───────────────┤3             ├
                 │           │               │              │
    mul_res_0.0: ┤4          ├──■─────────■──┤4             ├
                 │           │  │         │  │              │
    mul_res_0.1: ┤5          ├──o─────────o──┤5             ├
                 │           │  │         │  │              │
    mul_res_0.2: ┤6          ├──o─────────o──┤6             ├
                 │           │  │         │  │              │
    mul_res_0.3: ┤7          ├──o─────────o──┤7             ├
                 │           │  │         │  │              │
    mul_res_0.4: ┤8  __mul__ ├──o─────────o──┤8  __mul___dg ├
                 │           │  │         │  │              │
    mul_res_0.5: ┤9          ├──o─────────o──┤9             ├
                 │           │  │         │  │              │
    mul_res_0.6: ┤10         ├──o─────────o──┤10            ├
                 │           │  │         │  │              │
    sbp_anc_0.0: ┤11         ├──┼─────────┼──┤11            ├
                 │           │  │         │  │              │
    sbp_anc_1.0: ┤12         ├──┼─────────┼──┤12            ├
                 │           │  │         │  │              │
    sbp_anc_2.0: ┤13         ├──┼─────────┼──┤13            ├
                 │           │  │         │  │              │
    sbp_anc_3.0: ┤14         ├──┼─────────┼──┤14            ├
                 │           │  │         │  │              │
    sbp_anc_4.0: ┤15         ├──┼─────────┼──┤15            ├
                 │           │  │         │  │              │
    sbp_anc_5.0: ┤16         ├──┼─────────┼──┤16            ├
                 └───────────┘┌─┴─┐┌───┐┌─┴─┐└──────────────┘
    eq_cond_0.0: ─────────────┤ X ├┤ Z ├┤ X ├────────────────
                              └───┘└───┘└───┘                
    Live QuantumVariables:
    ---------------------
    QuantumFloat qf_0

We can see how the multiplication is evaluated into a new QuantumFloat called ``mul_res_0`` using some ancilla qubits. Subsequently, a multi-controlled X-gate evaluates the condition of it to be equal to 0.25 into a qubit called ``eq_cond_0``. The ancilla qubits and ``eq_cond`` will be recycled for each other during :meth:`compilation <qrisp.QuantumSession.compile>`, implying there is 0 qubit overhead for the ancillae:

>>> qf.qs.compile().num_qubits()
12

We perform the Z-gate and :ref:`uncompute <uncomputation>`. The uncomputation is necessary here because the state the :meth:`Grover diffuser <qrisp.grover.diffuser>` acts on needs to be disentangled.

.. note::
   QuantumVariables can be named independently of their name as a Python variable. If no name is provided, Qrisp tries to infer the name of the Python variable but in many cases there is ambiguity, meaning there is no guaranteed relation between the naming of the qubits and the name of the Python variable.

Grover's algorithm
==================

The code for embedding the constructed oracle into Grover's algorithm is: ::


    from qrisp.grover import diffuser

    qf = QuantumFloat(3, -1, signed = True)

    n = qf.size
    iterations = int((2**n/2)**0.5)

    h(qf)

    for i in range(iterations):
        sqrt_oracle(qf)
        diffuser(qf)


>>> print(qf)
{0.5: 0.4727, -0.5: 0.4727, 0.0: 0.0039, 1.0: 0.0039, 1.5: 0.0039, 2.0: 0.0039, 2.5: 0.0039, 3.0: 0.0039, 3.5: 0.0039, -4.0: 0.0039, -3.5: 0.0039, -3.0: 0.0039, -2.5: 0.0039, -2.0: 0.0039, -1.5: 0.0039, -1.0: 0.0039}
   

First we create the :doc:`/./reference/Quantum Types/QuantumFloat` which will contain the solution. Note that the QuantumFloat constructor creates unsigned floats by default. We determine the number of iterations according to the formula given `here <https://arxiv.org/abs/quant-ph/9909040>`_, taking into consideration that we expect two solutions ($S = \{0.5, -0.5\}$). The next step is then to bring ``qf`` into uniform superposition, followed by the Grover iterations and finalized by a :meth:`measurement<qrisp.QuantumVariable.get_measurement>` (which is called by ``print``).

Quantum Phase Estimation
------------------------

`Quantum phase estimation <https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm>`_ is an important subroutine in many quantum algorithms. If you are not familiar with this algorithm, we recommend that you first read about it `elsewhere <https://qiskit.org/textbook/ch-algorithms/quantum-phase-estimation.html>`_ and then come back here. We shortly summarize the problem this algorithm solves:
Given is a unitary $U$ and quantum state $\ket{\psi}$ which is an eigenvector of $U$:

.. math::

   U \ket{\psi} = \text{exp}(i 2 \pi \phi)\ket{\psi}

Applying quantum phase estimation to $U$ and $\ket{\psi}$ returns a quantum register containing an estimate for the value of $\phi$.

.. math::

   \text{QPE}_{U} \ket{\psi} \ket{0} = \ket{\psi} \ket{\phi}

It can be implemented within a few lines of code in Qrisp:

::

    from qrisp import QuantumFloat, control, QFT, h    

    def QPE(psi, U, precision):
       
        res = QuantumFloat(precision, -precision)

        h(res)

        for i in range(precision):
            with control(res[i]):
                for j in range(2**i):
                    U(psi)
       
        return QFT(res, inv = True)

The first step here is to create the :meth:`QuantumFloat <qrisp.QuantumFloat>` ``res`` which will contain the result. The first argument specifies the amount of mantissa qubits, the QuantumFloat should contain and the second argument specifies the exponent. Having $n$ mantissa qubits and and exponent of $-n$ means that this QuantumFloat can represent the values between 0 and 1 with a granularity of $2^{-n}$. Subsequently we apply an Hadamard gate to all qubits of ``res`` and continue by performing controlled evaluations of $U$. This is achieved by using the ``with control(res[i]):`` statement. This statement enters a :ref:`ControlEnvironment` such that every quantum operation inside the indented code block will be controlled on the i-th qubit of ``res``. We conclude the algorithm by performing an inverse quantum fourier transformation of ``res``.

Note that compared to the `Qiskit implementation <https://qiskit.org/documentation/stubs/qiskit.circuit.library.PhaseEstimation.html>`_ the Qrisp version comes with the convenience that $U$ can be given as a Python function (instead of a Circuit object) allowing for slim and elegant evaluations. Furthermore, the line ``with control(res[i]):`` invokes a :meth:`ControlEnvironment <qrisp.ControlEnvironment>`, which can yield significant gains in performance if ``qpe`` is called within another ``ControlEnvironments`` (compared to the Qiskit approach of simply synthesizing the double controlled version for every participating gate).

We test our code with a simple example:

::

    from qrisp import p, QuantumVariable, multi_measurement
    import numpy as np

    def U(psi):
        phi_1 = 0.5
        phi_2 = 0.125

        p(phi_1*2*np.pi, psi[0])
        p(phi_2*2*np.pi, psi[1])
       
    psi = QuantumVariable(2)

    h(psi)

    res = QPE(psi, U, 3)

In this code snippet, we define a function ``U`` which applies a phase gate onto the first two qubits of its input. We then create the :ref:`QuantumVariable` ``psi`` and bring it into uniform superposition by applying Hadamard gates onto each qubit. Subsequently, we evaluate ``qpe`` on ``U`` and ``psi`` with the precision 3.

The quantum state is now:

.. math::
   
   \frac{1}{2} \text{QPE}_{U}(\ket{00} + \ket{01} + \ket{10} + \ket{11})\ket{0} = \frac{1}{2} (\ket{00}\ket{0} + \ket{10}\ket{\phi_1} + \ket{01}\ket{\phi_2} +\ket{11}\ket{\phi_1 + \phi_2})

We verify by measuring ``psi`` :meth:`together<qrisp.multi_measurement>` with ``res``:

>>> print(multi_measurement([psi, res]))
{('00', 0.0): 0.25,
 ('10', 0.5): 0.25,
 ('01', 0.125): 0.25,
 ('11', 0.625): 0.25}
