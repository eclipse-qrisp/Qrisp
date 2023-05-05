Core
====

.. toctree::
   :maxdepth: 2
   :hidden:

   QuantumVariable
   QuantumSession
   QuantumArray
   QuantumDictionary
   Uncomputation
   Session Merging

The core module of Qrisp provides fundamental functionality that is essential for building quantum applications. It includes features such as data structures, algorithms, and utility functions that can be used across multiple components of quantum algorithm development. Additionally, the core module includes interfaces for :meth:`input <qrisp.QuantumVariable.encode>` and :meth:`output <qrisp.QuantumVariable.get_measurement>` operations, :meth:`compilation <qrisp.QuantumSession.compile>` to quantum circuits, and uncomputation. This module serves as the foundation for the entire framework and provides developers with the necessary tools to build reliable and scalable applications.

:ref:`QuantumVariable <QuantumVariable>`
----------------------------------------

QuantumVariables are the central building block of Qrisp algorithms. Using QuantumVariables, many of the nitty-gritty management tasks such as choosing a suited set of qubits within a complex algorithm are automated.
Furthermore the class provides features such as human readable in- and outputs, quantum typing via class inheritance or infix arithmetics.
A :ref:`QuantumFloat<QuantumFloat>` is an example for a subclass of QuantumVariable, which can represent numbers.


:ref:`QuantumSession <QuantumSession>`
--------------------------------------

Each QuantumVariable is registered in exactly one QuantumSession. The QuantumSession manages the lifetime cycle of the QuantumVariables being assigned to that Session and enables features such as automatic uncomputation.
The QuantumSession a QuantumVariable is assigned to, is stored in the ``qs`` attribute of a QuantumVariable.

.. code-block:: python

   from qrisp import QuantumVariable
   qv = QuantumVariable(5)
   qs = qv.qs


:ref:`QuantumArray <QuantumArray>`
----------------------------------

Multiple QuantumVariables with the same type can be handled within a QuantumArray. As an inheritor of the `numpy array <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_, QuantumArrays can be handled with a lot of flexibility since many of the established numpy methods work for QuantumArrays too. For QuantumArrays with type QuantumFloat, matrix multiplication and tensor contraction is available.


:ref:`QuantumDictionary <QuantumDictionary>`
--------------------------------------------

A QuantumDictionary extends a classical python dictionary, by allowing QuantumVariables as keys as well. Using this mechanism it is possible to load classical data relations into the quantum computer via logic synthesis.


:ref:`Uncomputation <uncomputation>`
---------------------------------------
This page elaborates on the ways automatic uncomputation can be achieved in Qrisp. Mainly there are two ways to call this procedure:

#. The :meth:`uncompute <qrisp.QuantumVariable.uncompute>` method, which uncomputes the QuantumVariable (if possible) and calls :meth:`delete <qrisp.QuantumVariable.delete>` afterwards.
#. The ``auto_uncompute`` decorator, which uncomputes any local QuantumVariable of Python function.


:ref:`Session Merging <SessionMerging>`
---------------------------------------

As the title indicates, this page elaborates how QuantumSessions are merged (manually and automatically). Roughly speaking, two sessions are automatically merged if an entangling operation is executed between two of the contained QuantumVariables. For manual merging we provide the merge function

.. code-block:: python
   
   from qrisp import QuantumVariable, merge, cx
   
   #Create QuantumVariables (each registered in their own session)
   qv_0 = QuantumVariable(1)
   qv_1 = QuantumVariable(1)
   qv_2 = QuantumVariable(1)

   #Merge their QuantumSessions manually using the merge function
   merge(qv_0.qs, qv_1.qs)
   
   #Merge automatically by executing an entangling operation
   cx(qv_1, qv_2)
   