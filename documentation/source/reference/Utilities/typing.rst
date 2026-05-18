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
   :value: int | float | complex | bool | np.generic

   A Python or NumPy scalar value. Covers Python built-in scalars and all
   NumPy scalar types (``np.float64``, ``np.int32``, etc. via ``np.generic``).

.. py:data:: NDArrayLike
   :type: TypeAlias
   :value: np.ndarray | jax.Array | jax.core.Tracer

   A multi-dimensional array value. Covers NumPy arrays, JAX arrays, and JAX
   tracers. Use this when a parameter is expected to be an array, 
   and the function should be compatible with both NumPy and JAX. 

.. py:data:: ArrayLike
   :type: TypeAlias
   :value: ScalarLike | NDArrayLike

   Union of :data:`ScalarLike` and :data:`NDArrayLike`. Use this when a
   parameter accepts either scalars or arrays. Use the narrower aliases when
   only one kind is expected, to avoid spurious Pylance warnings about missing
   attributes (such as ``.shape``, etc.).

.. py:data:: Param
   :type: TypeAlias
   :value: float | int | complex | np.number | sympy.Expr | jax.Array

   A gate parameter value.

   Covers all types accepted as gate parameters throughout Qrisp: Python
   numeric scalars (``float``, ``int``, ``complex``), NumPy numeric scalars
   (``np.float64``, ``np.int32``, etc. via ``np.number``), symbolic
   expressions (``sympy.Symbol``, ``sympy.Expr``, and any SymPy expression),
   and JAX arrays (``jax.Array``, which covers both concrete arrays and tracers).
