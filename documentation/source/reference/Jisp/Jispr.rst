.. _jispr:

Jispr
=====

.. currentmodule:: qrisp.jisp
.. autoclass:: Jispr

Methods
=======

Manipulation
^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   
   Jispr.inverse
   Jispr.control
   Jispr.flatten_environments
   
   
Evaluation
^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   
   Jispr.to_qc
   Jispr.to_qir
   Jispr.to_mlir
   Jispr.to_catalyst_jaxpr


Advanced details
================

This section elaborates how Jispr objects are embedded into the Jax infrastructure. If you just want to accelerate your code you can (probably) skip this. It is recommended to first get a solid understanding of `Jax primitives <https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html>`_ and how to create a Jaxpr out of them.

Jisp is designed to model dynamic quantum computations with a minimal set of primitives.

For that, there are 3 new Jax data types defined:
        
* ``QuantumCircuit``, which represents an object that tracks what kind of manipulations are applied to the quantum state.
* ``QubitArray``, which represents an array of qubits, that can have a dynamic amount of qubits
* ``Qubit``, which represents individual qubits.

Before we describe how quantum computations are realized we list some "administrative" primitives and their semantics.

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - Primitive
     - Semantics
   * - ``create_qubits``
     - Can be used to create new qubits. Takes a ``QuantumCircuit`` and a (dynamic) integer and returns a new ``QuantumCircuit`` and a ``QubitArray``. 
   * - ``get_qubit``
     - Extracts a ``Qubit`` from a ``QubitArray``. Takes a ``QubitArray`` and a dynamic integer (indicating the position) and returns a ``Qubit``.
   * - ``get_size``
     - Retrieves the size of a ``QubitArray``. Takes a ``QubitArray`` and returns an integer (the size).

To instruct a quantum computation, the :ref:`Operation` class is elevated to a Jax primitive:

::

	from qrisp import *
	from qrisp.jisp import *
			
	def test_function(i):
	    qv = QuantumVariable(i)
	    cx(qv[0], qv[1])
	    bl = measure(qv[1])
	    return qv, bl
	   
	print(make_jispr(test_function)(2))

::

	{ lambda ; a:QuantumCircuit b:i32[]. let
        c:QuantumCircuit d:QubitArray = create_qubits a b
        e:Qubit = get_qubit d 0
        f:Qubit = get_qubit d 1
        g:QuantumCircuit = cx c e f
        h:QuantumCircuit i:bool[] = measure g f
    in (h, d, i) }

The line starting with ``g:`` describes how an :ref:`Operation` can be plugged into a Jispr: The first argument is always a ``QuantumCircuit``, and the following arguments are ``Qubit`` objects. With this kind of structure, Jisp is very close to how quantum computations are modelled mathematically: As a unitary that is applied to a tensor on certain indices. Indeed you can view the defined object as precisely that (if it helps you programming/understanding): ``QuantumCircuit`` objects represent tensors, ``Qubit`` object represent integer indices and ``QubitArray`` object represent arrays of indices.

The ``measure`` primitive takes a special role here: Compared to the other quantum operations, it not only returns a new ``QuantumCircuit`` but also a boolean value (the measurement outcome). It is also possible to call the ``measure`` on a ``QubitArray``:

::
	
	def test_function(i):
	    qv = QuantumVariable(i)
	    cx(qv[0], qv[1])
	    a = measure(qv)
	    return a

	print(make_jispr(test_function)(2))

::
	
    { lambda ; a:QuantumCircuit b:i32[]. let
        c:QuantumCircuit d:QubitArray = create_qubits a b
        e:Qubit = get_qubit d 0
        f:Qubit = get_qubit d 1
        g:QuantumCircuit = cx c e f
        h:QuantumCircuit i:i32[] = measure g d
    in (h, i) }

In this case, an integer is returned instead of a boolean value. Both variants return values (bool/int32) that other Jax modules understand, highlighting the seamless embedding of quantum computations into the Jax ecosystem.

QuantumEnvironments
^^^^^^^^^^^^^^^^^^^

:ref:`QuantumEnvironment` in Jisp are also represented by a dedicated primitive:

::

	def test_function(i):
	    qv = QuantumVariable(i)
		
	    with invert():
	        t(qv[0])
	        cx(qv[0], qv[1])
		
	    return qv

	jispr = make_jispr(test_function)(2)
	print(jispr)

::
	
    { lambda ; a:QuantumCircuit b:i32[]. let
        c:QuantumCircuit d:QubitArray = create_qubits a b
        e:QuantumCircuit = q_env[
        jispr={ lambda ; f:QuantumCircuit d:QubitArray. let
            g:Qubit = get_qubit d 0
            h:QuantumCircuit = t f g
            i:Qubit = get_qubit d 1
            j:QuantumCircuit = cx h g i
            in (j,) }
        type=InversionEnvironment
        ] c d
      in (e, d) }

You can see how the body of the :ref:`InversionEnvironment` is _collected_ into another Jispr. This reflects the fact that at their core, :ref:`QuantumEnvironments <QuantumEnvironment>` describe `higher-order quantum functions <https://en.wikipedia.org/wiki/Higher-order_function>`_ (ie. functions that operate on functions). In order to apply the transformations induced by the QuantumEnvironment, we can call ``Jispr.flatten_environments``:

>>> print(jispr.flatten_environments)
{ lambda ; a:QuantumCircuit b:i32[]. let
    c:QuantumCircuit d:QubitArray = create_qubits a b
    e:Qubit = get_qubit d 0
    f:Qubit = get_qubit d 1
    g:QuantumCircuit = cx c e f
    h:QuantumCircuit = t_dg g e
in (h, d) }

We see that as expected, the order of the ``cx`` and the ``t`` gate has been switched and the ``t`` gate has been turned into a ``t_dg``.