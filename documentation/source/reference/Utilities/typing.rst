.. _qrisp_typing:

Typing
======

.. currentmodule:: qrisp

Type aliases for use in function signatures and type annotations throughout
Qrisp. Import them directly from the top-level package:

.. code-block:: python

    from qrisp import QubitLike, ClbitLike, ScalarLike, NDArrayLike, ArrayLike

----

.. py:data:: QubitLike
   :type: TypeAlias
   :value: Qubit | int | list

   Accepted as a qubit specifier in circuit methods and gate functions.
   A single qubit can be identified either by its
   :class:`~qrisp.circuit.Qubit` object or by its integer index within the
   circuit. A list of either represents multiple qubits.

.. py:data:: ClbitLike
   :type: TypeAlias
   :value: Clbit | int | list

   Accepted as a classical bit specifier in circuit methods.
   A single classical bit can be identified either by its
   :class:`~qrisp.circuit.Clbit` object or by its integer index within the
   circuit. A list of either represents multiple classical bits.

.. py:data:: ScalarLike
   :type: TypeAlias
   :value: int | float | complex | bool | np.generic

   A Python or NumPy scalar value. Covers Python built-in scalars and all
   NumPy scalar types (``np.float64``, ``np.int32``, etc. via ``np.generic``).
   Variables typed as ``ScalarLike`` do not have a ``.shape`` attribute.

.. py:data:: NDArrayLike
   :type: TypeAlias
   :value: np.ndarray | jax.Array | jax.core.Tracer

   A multi-dimensional array value. Covers NumPy arrays, JAX arrays, and JAX
   tracers. All types in this alias expose a ``.shape`` attribute, so Pylance
   will not warn on attribute access when a parameter is typed as
   ``NDArrayLike``.

.. py:data:: ArrayLike
   :type: TypeAlias
   :value: ScalarLike | NDArrayLike

   Union of :data:`ScalarLike` and :data:`NDArrayLike`. Use this when a
   parameter accepts either scalars or arrays. Use the narrower aliases when
   only one kind is expected, to avoid spurious Pylance warnings about missing
   attributes such as ``.shape``.
