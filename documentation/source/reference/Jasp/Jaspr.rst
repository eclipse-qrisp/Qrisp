.. _jaspr:

Jaspr
=====

.. currentmodule:: qrisp.jasp
.. autoclass:: Jaspr

Methods
=======

Manipulation
^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   
   Jaspr.inverse
   Jaspr.control
   

Evaluation
^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   
   Jaspr.qjit
   Jaspr.to_qc
   Jaspr.extract_post_processing   
   Jaspr.to_qasm
   Jaspr.to_mlir
   Jaspr.to_catalyst_jaxpr
   Jaspr.to_catalyst_mlir
   Jaspr.to_qir
   

Construction
============

.. autofunction:: make_jaspr
   :noindex:


Advanced details
================

This section elaborates how Jaspr objects are embedded into the Jax infrastructure. If you just want to accelerate your code you can (probably) skip this. It is recommended to first get a solid understanding of `Jax primitives <https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html>`_ and how to create a Jaxpr out of them.

Jasp is designed to model dynamic quantum computations with a minimal set of primitives.

For that, there are 3 new Jax abstract data types defined:
        
* ``QuantumState``, which represents an object that tracks what kind of manipulations are applied to the quantum state.
* ``QubitArray``, which represents an array of qubits that can have a dynamic number of qubits.
* ``Qubit``, which represents individual qubits.

Before we describe how quantum computations are realized, we list some "administrative" primitives and their semantics.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Primitive
     - Semantics
   * - ``jasp.create_qubits``
     - Creates new qubits. Takes a (dynamic) integer representing the size and a ``QuantumState``, then returns a ``QubitArray`` and a new ``QuantumState``. 
   * - ``jasp.get_qubit``
     - Extracts a ``Qubit`` from a ``QubitArray``. Takes a ``QubitArray`` and a dynamic integer (indicating the position) and returns a ``Qubit``.
   * - ``jasp.get_size``
     - Retrieves the size of a ``QubitArray``. Takes a ``QubitArray`` and returns an integer (the size).
   * - ``jasp.delete_qubits``
     - Deallocates a ``QubitArray``. Takes a ``QubitArray`` and a ``QuantumState``, returns a new ``QuantumState``.
   * - ``jasp.reset``
     - Resets qubits in a ``QubitArray`` to the ``|0⟩`` state. Takes a ``QubitArray`` and a ``QuantumState``, returns a new ``QuantumState``.

Quantum Operations
^^^^^^^^^^^^^^^^^^

Quantum gates are represented by the ``jasp.quantum_gate`` primitive. Here's an example:

::

	from qrisp import *
	from qrisp.jasp import *
			
	def test_function(i):
	    qv = QuantumVariable(i)
	    cx(qv[0], qv[1])
	    bl = measure(qv[1])
	    return qv, bl
	   
	print(make_jaspr(test_function)(2))

::

	{ lambda ; a:i64[] b:QuantumState. let
	    c:QubitArray d:QuantumState = jasp.create_qubits a b
	    e:Qubit = jasp.get_qubit c 0:i64[]
	    f:Qubit = jasp.get_qubit c 1:i64[]
	    g:QuantumState = jasp.quantum_gate[gate=cx] e f d
	    h:bool[] i:QuantumState = jasp.measure f g
	  in (c, h, i) }

The line starting with ``g:`` describes how quantum gates are represented in a Jaspr: The gate name is specified in the parameters (``gate=cx``), followed by the ``Qubit`` arguments, and finally the ``QuantumState``. This structure closely mirrors how quantum computations are modeled mathematically: as a unitary applied to a tensor at certain indices. You can think of ``QuantumState`` objects as tensors, ``Qubit`` objects as integer indices, and ``QubitArray`` objects as arrays of indices.

The ``jasp.measure`` primitive takes a special role: Unlike other quantum operations, it not only returns a new ``QuantumState`` but also a measurement outcome. When measuring a single ``Qubit``, it returns a boolean value. When measuring a ``QubitArray``, it returns an integer:

::
	
	def test_function(i):
	    qv = QuantumVariable(i)
	    cx(qv[0], qv[1])
	    a = measure(qv)
	    return a

	print(make_jaspr(test_function)(2))

::
	
	{ lambda ; a:i64[] b:QuantumState. let
	    c:QubitArray d:QuantumState = jasp.create_qubits a b
	    e:Qubit = jasp.get_qubit c 0:i64[]
	    f:Qubit = jasp.get_qubit c 1:i64[]
	    g:QuantumState = jasp.quantum_gate[gate=cx] e f d
	    h:i64[] i:QuantumState = jasp.measure c g
	  in (h, i) }

Both variants return values (``bool[]`` or ``i64[]``) that other Jax modules understand, highlighting the seamless embedding of quantum computations into the Jax ecosystem.

QuantumEnvironments
^^^^^^^^^^^^^^^^^^^

:ref:`QuantumEnvironment` objects in Jasp can be represented in two forms: unflattened (where environments appear as ``jasp.q_env`` primitives) or flattened (where environment transformations are applied directly).

**Unflattened form** (``flatten_envs=False``):

::

	def test_function(i):
	    qv = QuantumVariable(i)
		
	    with invert():
	        t(qv[0])
	        cx(qv[0], qv[1])
		
	    return qv

	jaspr = make_jaspr(test_function, flatten_envs=False)(2)
	print(jaspr)

::
	
	{ lambda ; a:i64[] b:QuantumState. let
	    c:QubitArray d:QuantumState = jasp.create_qubits a b
	    e:QuantumState = jasp.q_env[
	      jaspr={ lambda ; c:QubitArray f:QuantumState. let
	          g:Qubit = jasp.get_qubit c 0:i64[]
	          h:QuantumState = jasp.quantum_gate[gate=t] g f
	          i:Qubit = jasp.get_qubit c 1:i64[]
	          j:QuantumState = jasp.quantum_gate[gate=cx] g i h
	        in (j,) }
	      type=InversionEnvironment
	    ] c d
	  in (c, e) }

Here, the body of the :ref:`InversionEnvironment` is collected into a nested Jaspr within the ``jasp.q_env`` primitive. This representation preserves the environment structure and reflects how :ref:`QuantumEnvironments <QuantumEnvironment>` describe `higher-order quantum functions <https://en.wikipedia.org/wiki/Higher-order_function>`_ that transform quantum operations.

**Flattened form** (``flatten_envs=True``, default):

::

	jaspr = make_jaspr(test_function, flatten_envs=True)(2)
	print(jaspr)

::
	
	{ lambda ; a:i64[] b:QuantumState. let
	    c:QubitArray d:QuantumState = jasp.create_qubits a b
	    e:Qubit = jasp.get_qubit c 0:i64[]
	    f:Qubit = jasp.get_qubit c 1:i64[]
	    g:QuantumState = jasp.quantum_gate[gate=cx] e f d
	    h:QuantumState = jasp.quantum_gate[gate=t_dg] e g
	  in (c, h) }

In the flattened form, the :ref:`InversionEnvironment` transformation has been applied: the order of the ``cx`` and ``t`` gates has been reversed, and the ``t`` gate has been transformed into ``t_dg`` (T-dagger). This is the default behavior as it produces more optimized Jaspr representations suitable for execution.

For more detailed information about the Jasp primitives and their semantics, see the :ref:`MLIR Interface <mlir_interface>` documentation.