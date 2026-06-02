.. _qrisp_typing:

Typing
======

.. currentmodule:: qrisp

Type aliases for use in function signatures and type annotations throughout
Qrisp. These can be imported directly from the top-level package:

.. code-block:: python

    from qrisp.typing import QubitLike, ClbitLike   # etc.

----

.. py:data:: QubitLike
   :type: TypeAlias
   :value: Qubit | int | Sequence[Qubit | int]

   Accepted as a qubit specifier in circuit methods and gate functions.
   A single qubit can be identified either by its :ref:`Qubit` object or by its
   integer index within the circuit. A sequence of either represents multiple
   qubits.

.. py:data:: ClbitLike
   :type: TypeAlias
   :value: Clbit | int | Sequence[Clbit | int]

   Accepted as a classical bit specifier in circuit methods.
   A single classical bit can be identified either by its :ref:`Clbit` object or
   by its integer index within the circuit. A sequence of either represents
   multiple classical bits.

.. py:data:: ScalarLike
   :type: TypeAlias
   :value: int | float | complex | bool | np.generic | jax.core.Tracer

   A Python, NumPy, or JAX scalar value. Covers Python built-in scalars, all
   NumPy scalar types, and JAX tracers.

.. py:data:: NDArrayLike
   :type: TypeAlias
   :value: np.ndarray | jax.Array | jax.core.Tracer

   A multi-dimensional array value. Covers NumPy arrays, JAX arrays, and JAX
   tracers. Useful when a parameter is expected to be an array, 
   and the function should be compatible with both NumPy and JAX. 

.. py:data:: ArrayLike
   :type: TypeAlias
   :value: ScalarLike | NDArrayLike

   Union of :data:`ScalarLike` and :data:`NDArrayLike`. Useful when a
   parameter accepts either scalars or arrays. This is similar in spirit to
   ``jax.typing.ArrayLike``, but is defined within Qrisp for consistency with
   the rest of this typing module.

.. py:data:: Param
   :type: TypeAlias
   :value: float | int | complex | np.number | sympy.Expr | jax.Array | jax.core.Tracer

   A gate parameter value.

   Covers all types accepted as gate parameters throughout Qrisp: Python
   numeric scalars, NumPy numeric scalars, symbolic expressions,
   concrete JAX arrays, and JAX tracers.
