.. _SessionMerging:

Session Merging
===============

Session merging describes the process of merging two QuantumSessions into one. This construct is necessary because we want to ensure that the user can

* write multiple quantum algorithms in parallel inside a single Python instance, and
* construct circuits that only include the qubits, which are relevant for that quantum algorithm.

While the first point in principle could be achieved by having a global quantum circuit per Python instance, the second point discourages this approach, as this would imply that each time we want to query a backend, a circuit analyzer would have to pinpoint which qubits of the global quantum circuit are relevant and which are not.

We therefore decided to structure separate algorithms into separate QuantumSessions.

This however leaves the user with the inconvience of specifying the QuantumSession each time a QuantumVariable is created. We approach this by implementing the mechanism of session merging.

Session merging allows the user to delay specifying the QuantumSession of a QuantumVariable until it is actually integrated into the algorithm. By default QuantumVariables are created in their own new session and once it is required this session can then be merged into the main algorithms QuantumSession.

>>> from qrisp import QuantumVariable
>>> a = QuantumVariable(1)
>>> b = QuantumVariable(1)
>>> a.qs == b.qs
False

To manually merge the sessions we can use the merge function

>>> from qrisp import merge
>>> merge(a, b)
>>> a.qs == b.qs
True

**Automated session merging**

It is however rarely the case that this is actually required as the neccesity for merging can be automatically inferred from the context in many cases. For instance executing any multiqubit gate on two QuantumVariables with disjoint QuantumSessions results in a merge.

>>> from qrisp import cx
>>> a = QuantumVariable(1)
>>> b = QuantumVariable(1)
>>> cx(a, b)
>>> a.qs == b.qs
True

**Automated session merging in QuantumEnvironments**

Another situation where automatic merging happens is in QuantumEnvironments ::

   from qrisp import QuantumEnvironment, x
   a = QuantumVariable(1)
   b = QuantumVariable(1)
   
   with QuantumEnvironment():
      x(a[0])
      x(b[0])


>>> a.qs == b.qs
True
   
Note that there has been no explicit entangling operation. The quantum variables get merged into the enviroments session as soon as any quantum operation is applied to them.
This is however not the case for QuantumSessions that get created inside this environment. This ensures that QuantumSessions which are created only temporarily, do not get merged ::

   
   
   a = QuantumVariable(1, name = "a")
   
   def some_function_creating_a_quantum_session():
      b = QuantumVariable(1)
      x(b)
      return b.qs.to_op(name = "gate_name")
   
   with QuantumEnvironment():
      temp_op = some_function_creating_a_quantum_session()
      a.qs.append(temp_op, a.reg)

>>> print(a.qs)

::

    QuantumCircuit:
    --------------
         ┌───────────┐
    a.0: ┤ gate_name ├
         └───────────┘
    Live QuantumVariables:
    ---------------------
    QuantumVariable a

In this snipped we mimic the situation, that a submodule is creating a gate object. In the defined function, the QuantumVariable creates a QuantumSession and performs some (trivial) operations. After that, this quantum session is turned into a gate object and returned. We see that the QuantumSession b was registered in, is not merged into a. 
