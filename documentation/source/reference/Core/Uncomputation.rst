.. _uncomputation:

Uncomputation
===============

Why uncomputation?
------------------

Uncomputation is an important aspect of quantum information processing because it allows for the efficient use of quantum resources. In classical computing this can be achieved by deleting information and reusing the deleted bits for other purposes. Deleting (or resetting) a qubit is however not a reversible process and is usually performed by measuring the qubit in question and performing a bitflip based on the outcome. This measurement in turn collapses the superposition of other entangled qubits, which are supposed to be unaffected. In many cases this collapse interferes with the quantum algorithm, such that the resulting state can no longer be used.
In some situations, uncomputation is not only relevant as a way to manage quantum resources but is required in order for the quantum algorithm to function properly. One such example is Grover's algorithm. Assume we have two QuantumVariables, of which one is in a state of uniform superposition


.. math::

   \ket{\psi_0} = \sum_{i = 0}^{2^n-1} \ket{i} \ket{0}


An oracle now calculates some boolean value $f(i)$, which is required in order to perform the phase tag. 

.. math::

   \ket{\psi_1} = \sum_{i = 0}^{2^n-1} \ket{i} \ket{f(i)}

After performing the phase tag, the state is

.. math::

   \ket{\psi_2} = Z_{\ket{f(i)}} \ket{\psi_1}
   = \sum_{i = 0}^{2^n-1} (-1)^{f(i)} \ket{i} \ket{f(i)}

In order for the math of diffuser the diffuser to work out, we need the state to be disentangled, ie.

.. math::

   \ket{\psi_3} = \sum_{i = 0}^{2^n-1} (-1)^{f(i)} \ket{i} \ket{0}
   
Therefore we need to uncompute the variable containing $f(i)$.

How-To uncompute
----------------

In many cases, uncomputing a QuantumVariable can be achieved by the inverse of the steps required for the computation. While this seems like a simple recipe, it can be ambiguous (was the Z gate a part of the computation?). In any case it is a tedious amount of extra programming work to be done, which should be automated.
Fortunately an `algorithm for automatic uncomputation <https://github.com/eth-sri/Unqomp>`_  has been developed at ETH Zürich.
An important advantage of Unqomp is, that it does not follow the philosophy of simply reverting the computation. This feature enables the algorithm to skip the "un-uncomputation" of values which would be required to recompute in order to perform the uncomputation.

Calling Unqomp in Qrisp
-----------------------

Unqomp has been implemented within Qrisp and we provide two ways to call this procedure:
The first is the *auto_uncompute* decorator, which automatically uncomputes all local :ref:`QuantumVariables <QuantumVariable>` of a function.
To demonstrate the functionality, we create a function which returns a :ref:`QuantumBool` containing the AND value of its three inputs. In order to do this, this function creates a local QuantumBool, which stores the temporary result of the AND value of the first two inputs. ::

   from qrisp import QuantumBool, mcx

   def triple_AND(a, b, c):

      local = QuantumBool()
      result =  QuantumBool()
      
      mcx([a, b], local)
      mcx([local, c], result)
      
      return result

   a = QuantumBool()
   b = QuantumBool()
   c = QuantumBool()

   result = triple_AND(a, b, c)
   
>>> print(result.qs)

::

    QuantumCircuit:
    --------------
         a.0: ──■───────
                │       
         b.0: ──■───────
                │       
         c.0: ──┼────■──
              ┌─┴─┐  │  
     local.0: ┤ X ├──■──
              └───┘┌─┴─┐
    result.0: ─────┤ X ├
                   └───┘
    Live QuantumVariables:
    ---------------------
    QuantumBool a
    QuantumBool b
    QuantumBool c
    QuantumBool local
    QuantumBool result

We will now redefine this function with the *auto_uncompute* decorator ::

   from qrisp import auto_uncompute
   
   @auto_uncompute
   def triple_AND(a, b, c):

      local = QuantumBool()
      result = QuantumBool()
      
      mcx([a, b], local)
      mcx([local, c], result)
      
      return result

   a = QuantumBool()
   b = QuantumBool()
   c = QuantumBool()

   result = triple_AND(a, b, c)
   
>>> print(result.qs)

::

    QuantumCircuit:
    --------------
              ┌────────┐     ┌────────┐
         a.0: ┤0       ├─────┤0       ├
              │        │     │        │
         b.0: ┤1       ├─────┤1       ├
              │  pt2cx │     │  pt2cx │
         c.0: ┤        ├──■──┤        ├
              │        │  │  │        │
     local.0: ┤2       ├──■──┤2       ├
              └────────┘┌─┴─┐└────────┘
    result.0: ──────────┤ X ├──────────
                        └───┘          
    Live QuantumVariables:
    ---------------------
    QuantumBool a
    QuantumBool b
    QuantumBool c
    QuantumBool result

We see that the multi-controlled X-gate acting on the local :ref:`QuantumBool` has been replaced by a gate called ``pt2cx`` which stands for phase tolerant two controlled X. For the case of two controls, this is the so called `Margolus gate <https://arxiv.org/abs/quant-ph/0312225>`_. This gate performs the logical operation of a Toffoli gate at only 3 CNOT gates, but introduces an extra phase for each input. Since the inputs here stay unchanged, this extra phase is reversed, once the second inverted Margolus gate is performed.

The second way of calling uncomputation is the :meth:`uncompute<qrisp.QuantumVariable.uncompute>` method of the :ref:`QuantumVariable` class. We demonstrate the use with our established example ::

   def triple_AND(a, b, c):

      local = QuantumBool()
      result =  QuantumBool()
      
      mcx([a, b], local)
      mcx([local, c], result)
      
      local.uncompute()
      
      return result

   a = QuantumBool()
   b = QuantumBool()
   c = QuantumBool()

   result = triple_AND(a, b, c)
   
>>> print(result.qs)

::

    QuantumCircuit:
    --------------
              ┌────────┐     ┌────────┐
         a.0: ┤0       ├─────┤0       ├
              │        │     │        │
         b.0: ┤1       ├─────┤1       ├
              │  pt2cx │     │  pt2cx │
         c.0: ┤        ├──■──┤        ├
              │        │  │  │        │
     local.0: ┤2       ├──■──┤2       ├
              └────────┘┌─┴─┐└────────┘
    result.0: ──────────┤ X ├──────────
                        └───┘          
    Live QuantumVariables:
    ---------------------
    QuantumBool a
    QuantumBool b
    QuantumBool c
    QuantumBool result

.. note::
   The :meth:`uncompute <qrisp.QuantumVariable.uncompute>` method and the ``auto_uncompute`` decorator automatically call the :meth:`delete <qrisp.QuantumVariable.delete>` method after successfull uncomputation.

In some cases, the entanglement structure of a set of QuantumVariables only allows uncomputation if all of them are uncomputed together. In this situation, setting ``do_it = False`` marks a QuantumVariable for uncomputation but does not actually perform it. On the next call with ``do_it = True``, the whole batch is uncomputed together ::

   from qrisp import gate_wrap, cx

   @gate_wrap
   def fanout(a, b, c):
       cx(a,b)
       cx(a,c)

   a = QuantumBool()
   b = QuantumBool()
   c = QuantumBool()

   fanout(a,b,c)

>>> print(a.qs)

::

    QuantumCircuit:
    --------------
         ┌─────────┐
    c.0: ┤0        ├
         │         │
    b.0: ┤1 fanout ├
         │         │
    a.0: ┤2        ├
         └─────────┘
    Live QuantumVariables:
    ---------------------
    QuantumBool a
    QuantumBool b
    QuantumBool c

>>> b.uncompute()
Exception: Uncomputation failed because gate "fanout" needs to be uncomputed but is also targeting qubits [Qubit(c.0)] which are not up for uncomputation

In this example, the :meth:`gate_wrap <qrisp.gate_wrap>` decorator makes sure, the quantum gates inside of the ``fanout`` function are bundled into a single gate object. Since it acts on ``b`` and ``c`` alike, we would also uncompute ``c`` if we uncomputed ``b``.

We now queue *b* for uncomputation and perform the algorithm once *c* is also up for uncomputation. ::
   
   a = QuantumBool()
   b = QuantumBool()
   c = QuantumBool()

   fanout(a,b,c)

>>> b.uncompute(do_it = False)
>>> c.uncompute()

::

    QuantumCircuit:
    --------------
         ┌─────────┐┌────────────┐   
    a.0: ┤0        ├┤0           ├
         │         ││            │
    b.0: ┤1 fanout ├┤1 fanout_dg ├
         │         ││            │
    c.0: ┤2        ├┤2           ├
         └─────────┘└────────────┘
    Live QuantumVariables:
    ---------------------
    QuantumBool a

This problem might seem a bit constructed, because the ``fanout`` gate could in principle be decomposed into a sequence of CNOT gates, which would face no such issue. Not decomposing gates during uncomputation however allows a feature which will be highlighted in the next section.

Uncomputing synthesized gates
-----------------------------

Even though the Unqomp algorithm provides a very convenient way of solving automatic uncomputation, it comes with a few restrictions. We won't go into these too deep here because they are well documented in their publication - the most important one can be overcome using the Qrisp implementation of this algorithm. This restriction imposes that only a certain class of gates can be uncomputed, which the authors of Unqomp call ``qfree``. A quantum gate is ``qfree`` if it neither introduces nor destroys states of superposition.
In more mathematical terms, this implies that the unitary matrix of a ``qfree``  gate can only have a single non-zero entry per column.
This is a serious restriction, since many quantum functions make use of non-qfree gates such as the Hadamard, even though their net-effect is ``qfree``. An example of such a situation is Fourier arithmetic (of which Qrisps arithmetic module makes heavy use). Even though the multiplication function

.. math::

   U_{mul}\ket{a}\ket{b}\ket{0} = \ket{a}\ket{b}\ket{a \cdot b}

itself is ``qfree``, it makes use of Hadamard gates, which are not ``qfree`` .
In order to overcome this major restriction, the Qrisp implementation of Unqomp will not decompose gate objects but instead check the combined gate for ``qfree``-ness.

This feature (in combination with the :meth:`gate_wrap decorator <qrisp.gate_wrap>`) can be used to create quantum functions that can be successfully uncomputed event hough their inner workings contain non-qfree gates.

Permeability
------------

Permeability is a concept, that is introduced within Qrisps implementation of Unqomp, that generalizes the notion of a "control knob". The permeability status of a gate object on a certain input qubit $q_0$ decides how this gate is treated, when $q_0$ is uncomputed.
A gate is called permeable on qubit i, if it commutes with the Z operator on this qubit.

.. math::

   \text{U} \text{ is permeable on qubit i} \Leftrightarrow \text{U} \text{Z}_i = \text{Z}_i \text{U}

This implies that any controlled gate is permeable on its control qubit because

.. math::
   :nowrap:

   \begin{align*}
   \text{Z}_0 \text{cU} &= \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & U \end{pmatrix}\\
   &= \begin{pmatrix} 1 & 0 \\ 0 & -U \end{pmatrix}\\
   &= \text{cU} \text{Z}_0
   \end{align*}

However, not every permeable unitary is equal to a controlled gate (example: $\text{Z}_0 \text{CX}_{01}$).

Qrisps uncomputation algorithm automatically determines ``qfree``-ness and permeability of given gate objects. For gate objects with a lot of qubits, this can however significally impair compilation performance since this calculation requires investigation of the unitary. To overcome this problem, the :meth:`gate_wrap decorator <qrisp.gate_wrap>` supports developer specification of qfree-ness and permeability. Note that the given information won't be verified by default, since this would again require the calculation of the unitary and therefore defy the purpose of this feature. Furthermore we would like to make you aware of the :meth:`lifted <qrisp.lifted>` decorator, which is a shorthand of :meth:`gate_wrap <qrisp.gate_wrap>` with the corresponding keyword arguments set up to support automatic uncomputation.


Recomputation
-------------
.. _recomputation:

Recomputation is a phenomenon that can happen if a function ``f`` that uncomputes a local value ``u`` itself is uncomputed. If ``f`` is simply inverted, ``u`` will be recomputed within the inverted version of ``f``. Within Unqomp, recomputation never happens:

::

   from qrisp import z

   def f(a, b, c):

      u = QuantumBool()
      result = QuantumBool()
      
      mcx([a, b], u)
      mcx([u, c], result)
      
      u.uncompute()
      
      return result
   
   
   a = QuantumBool()
   b = QuantumBool()
   c = QuantumBool()
   
   result = f(a, b, c)
   
   z(result)
   
   result.uncompute()

>>> print(result.qs)

::

    QuantumCircuit:
    --------------
              ┌────────┐                         ┌────────┐
         a.0: ┤0       ├─────────────────────────┤0       ├
              │        │                         │        │
         b.0: ┤1       ├─────────────────────────┤1       ├
              │  pt2cx │┌────────┐     ┌────────┐│  pt2cx │
         c.0: ┤        ├┤1       ├─────┤1       ├┤        ├
              │        ││        │     │        ││        │
         u.0: ┤2       ├┤0 pt2cx ├─────┤0 pt2cx ├┤2       ├
              └────────┘│        │┌───┐│        │└────────┘
    result.0: ──────────┤2       ├┤ Z ├┤2       ├──────────
                        └────────┘└───┘└────────┘          
    Live QuantumVariables:
    ---------------------
    QuantumBool a
    QuantumBool b
    QuantumBool c
      
We see that the uncomputation of ``u`` is awaited until the uncomputation of ``result`` finished, even though within the code, the :meth:`uncompute<qrisp.QuantumVariable.uncompute>` call of ``u`` came before the one of ``result``. In many situations this is a really efficient and helpfull feature of the algorithm, since there is less overhead in gates because ``u`` doesn't need to be recomputed.
There are however situations, where this can yield to a significant qubit overhead. To understand how this happens, consider the following simplified situation:

::
   
   def quadra_AND(a,b,c,d):
      
      u = QuantumBool()
      v = QuantumBool()
      
      result = QuantumBool()
      
      mcx([a,b], u)
      mcx([u, c], v)
      
      u.uncompute()
      
      mcx([v,d], result)
      
      v.uncompute()
      
      return result
   
   a = QuantumBool()
   b = QuantumBool()
   c = QuantumBool()
   d = QuantumBool()
   
   result = quadra_AND(a,b,c,d)
   
   z(result)
   
   result.uncompute()

>>> print(result.qs)

::

    QuantumCircuit:
    --------------
              ┌────────┐                                             ┌────────┐
         a.0: ┤0       ├─────────────────────────────────────────────┤0       ├
              │        │                                             │        │
         b.0: ┤1       ├─────────────────────────────────────────────┤1       ├
              │        │┌────────┐                         ┌────────┐│        │
         c.0: ┤  pt2cx ├┤1       ├─────────────────────────┤1       ├┤  pt2cx ├
              │        ││        │┌────────┐     ┌────────┐│        ││        │
         d.0: ┤        ├┤        ├┤1       ├─────┤1       ├┤        ├┤        ├
              │        ││  pt2cx ││        │     │        ││  pt2cx ││        │
         u.0: ┤2       ├┤0       ├┤        ├─────┤        ├┤0       ├┤2       ├
              └────────┘│        ││  pt2cx │     │  pt2cx ││        │└────────┘
         v.0: ──────────┤2       ├┤0       ├─────┤0       ├┤2       ├──────────
                        └────────┘│        │┌───┐│        │└────────┘          
    result.0: ────────────────────┤2       ├┤ Z ├┤2       ├────────────────────
                                  └────────┘└───┘└────────┘                    
    Live QuantumVariables:
    ---------------------
    QuantumBool a
    QuantumBool b
    QuantumBool c
    QuantumBool d

In the above code snippet, ``result`` only holds a value at times, where ``u`` is deallocated, implying there is a qubit overhead because the qubit containing ``u`` could be recycled to be used for ``result``. However because the uncomputation of ``u`` is delayed until the uncomputation of ``result`` such a recycling is not possible. Therefore the whole point of the uncomputation (efficient qubit resource management) is gone. We circumvent this problem with the ``recompute`` keyword.

::
   
   def quadra_AND(a,b,c,d):
      
      u = QuantumBool()
      v = QuantumBool()
      
      result = QuantumBool()
      
      mcx([a,b], u)
      mcx([u, c], v)
      
      u.uncompute(recompute = True)
      
      mcx([v,d], result)
      
      v.uncompute()
      
      return result
   
   a = QuantumBool()
   b = QuantumBool()
   c = QuantumBool()
   d = QuantumBool()
   
   result = quadra_AND(a,b,c,d)
   
   z(result)
   
   result.uncompute()

>>> print(result.qs)

::

    QuantumCircuit:
    --------------
              ┌────────┐          ┌────────┐          ┌────────┐          »
         a.0: ┤0       ├──────────┤0       ├──────────┤0       ├──────────»
              │        │          │        │          │        │          »
         b.0: ┤1       ├──────────┤1       ├──────────┤1       ├──────────»
              │        │┌────────┐│        │          │        │          »
         c.0: ┤  pt2cx ├┤1       ├┤  pt2cx ├──────────┤  pt2cx ├──────────»
              │        ││        ││        │┌────────┐│        │┌────────┐»
         d.0: ┤        ├┤        ├┤        ├┤1       ├┤        ├┤1       ├»
              │        ││  pt2cx ││        ││        ││        ││        │»
         u.0: ┤2       ├┤0       ├┤2       ├┤        ├┤2       ├┤        ├»
              └────────┘│        │└────────┘│  pt2cx │└────────┘│  pt2cx │»
         v.0: ──────────┤2       ├──────────┤0       ├──────────┤0       ├»
                        └────────┘          │        │  ┌───┐   │        │»
    result.0: ──────────────────────────────┤2       ├──┤ Z ├───┤2       ├»
                                            └────────┘  └───┘   └────────┘»
    «                    ┌────────┐
    «     a.0: ──────────┤0       ├
    «                    │        │
    «     b.0: ──────────┤1       ├
    «          ┌────────┐│        │
    «     c.0: ┤1       ├┤  pt2cx ├
    «          │        ││        │
    «     d.0: ┤        ├┤        ├
    «          │  pt2cx ││        │
    «     u.0: ┤0       ├┤2       ├
    «          │        │└────────┘
    «     v.0: ┤2       ├──────────
    «          └────────┘          
    «result.0: ────────────────────
    «                              
    Live QuantumVariables:
    ---------------------
    QuantumBool a
    QuantumBool b
    QuantumBool c
    QuantumBool d

We see that the uncomputation of ``u`` is no longer delayed but performed instantly. Once ``result`` is uncomputed, ``u`` is once again recomputed. To reap our gains in qubit count, we call the :meth:`compile <qrisp.QuantumSession.compile>` method of the :ref:`QuantumSession`. This method performs an allocation algorithm to reduce the required qubit count (if possible)

>>> compiled_qc = result.qs.compile()
>>> print(compiled_qc)

::

                 ┌────────┐          ┌────────┐                         ┌────────┐»
            a.0: ┤0       ├──────────┤0       ├─────────────────────────┤0       ├»
                 │        │          │        │                         │        │»
            b.0: ┤1       ├──────────┤1       ├─────────────────────────┤1       ├»
                 │        │┌────────┐│        │                         │        │»
            c.0: ┤  pt2cx ├┤1       ├┤  pt2cx ├─────────────────────────┤  pt2cx ├»
                 │        ││        ││        │┌────────┐     ┌────────┐│        │»
            d.0: ┤        ├┤        ├┤        ├┤1       ├─────┤1       ├┤        ├»
                 │        ││  pt2cx ││        ││        │┌───┐│        ││        │»
    workspace_0: ┤2       ├┤0       ├┤2       ├┤2 pt2cx ├┤ Z ├┤2 pt2cx ├┤2       ├»
                 └────────┘│        │└────────┘│        │└───┘│        │└────────┘»
    workspace_1: ──────────┤2       ├──────────┤0       ├─────┤0       ├──────────»
                           └────────┘          └────────┘     └────────┘          »
    «                       ┌────────┐
    «        a.0: ──────────┤0       ├
    «                       │        │
    «        b.0: ──────────┤1       ├
    «             ┌────────┐│        │
    «        c.0: ┤1       ├┤  pt2cx ├
    «             │        ││        │
    «        d.0: ┤        ├┤        ├
    «             │  pt2cx ││        │
    «workspace_0: ┤0       ├┤2       ├
    «             │        │└────────┘
    «workspace_1: ┤2       ├──────────
    «             └────────┘          

>>> compiled_qc.num_qubits()
6

We can see how ``u`` is calculated into ``workspace_0`` and then uncomputed. Subsequently, ``result`` is computed into the recycled qubit and uncomputed afterwards. Finally ``u`` is recomputed, used to uncompute ``v`` and finally uncomputed for good. Performing the recomputation therefore gave us a circuit with one less qubit at the cost of two additional Margolus gates. This example is of course trivial but depending on the amount of qubits occupied by ``u`` and the amount of extra gates to perform a recomputation, this can be really beneficial (especially when working with a simulator, where qubits are a more costly resource than gates).