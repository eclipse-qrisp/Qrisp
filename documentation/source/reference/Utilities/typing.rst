.. _qrisp_typing:

Typing
======

.. currentmodule:: qrisp

Type aliases for use in function signatures and type annotations throughout
Qrisp. Import them directly from the top-level package:

.. code-block:: python

    from qrisp import QubitLike, ClbitLike

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

.. py:data:: ArrayLike
   :type: TypeAlias
   :value: int | float | complex | bool | np.ndarray | np.generic | jax.Array | jax.core.Tracer

   A type for all array-like numeric data accepted by Qrisp. Covers Python
   scalars, NumPy arrays and scalars, JAX arrays, and JAX tracers. JAX tracers
   appear whenever Qrisp code runs inside a Jasp-traced function (e.g. under
   ``@jaspify`` or ``jax.jit``).

   Because this alias contains only concrete types, ``isinstance`` checks work
   at runtime:

   .. code-block:: python

       from qrisp import ArrayLike
       import numpy as np

       isinstance(3.14, ArrayLike)           # True
       isinstance(np.array([1, 2, 3]), ArrayLike)  # True
       isinstance(np.float32(1.0), ArrayLike)      # True
