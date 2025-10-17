.. _scalable_integer_type:

Scalable Integer Type
=====================

.. currentmodule:: qrisp

JAX, and by extension Jasp, while highly performant for numerical computing, is limited to 64-bit integers. This becomes problematic for applications that require integer representations beyond that range—such as cryptography, arbitrary precision arithmetic, or certain simulations.

The ``BigInteger`` class solves this by implementing a scalable integer type that supports arbitrary bit-widths using arrays of smaller integers (32-bit integers). This allows for precise and efficient computation with large integers.

Example
-------

Here's a basic example that demonstrates the use of ``BigInteger``:

.. code-block:: python

    from qrisp import BigInteger

    # Create a BigInteger
    # The size is the amount of 32-bit integers, size=4 yields a 128-bit integer
    x = BigInteger.create(128, size=4)

    print(x())
    # Output: 128.0

Note that we use the ``__call__`` method (``x()``) to approximate the ``BigInteger`` as a nicely-readable float.

Class Documentation
-------------------

.. autoclass:: BigInteger

Static Methods
--------------

.. autosummary::
   :toctree: generated/

   BigInteger.create
   BigInteger.create_static

Operator Overloads
------------------

The ``BigInteger`` class supports standard arithmetic and comparison operators through Python's special methods (dunder methods). This allows you to use ``BigInteger`` instances much like native integers.

The following operators are implemented:

- Arithmetic: ``+`` (``__add__``), ``-`` (``__sub__``), ``*`` (``__mul__``), ``//`` (``__floordiv__``), ``%`` (``__mod__``), ``**`` (``__pow__``)
- Bitwise: ``&`` (``__and__``), ``|`` (``__or__``), ``^`` (``__xor__``), ``~`` (``__invert__``), ``<<`` (``__lshift__``), ``>>`` (``__rshift__``)
- Comparison: ``==`` (``__eq__``), ``!=`` (``__ne__``), ``<`` (``__lt__``), ``<=`` (``__le__``), ``>`` (``__gt__``), ``>=`` (``__ge__``)

This enables expressions like:

.. code-block:: python

    a = BigInteger.create(42)
    b = BigInteger.create(100)

    print((a + b)())     # 142.0
    print((b - a)())     # 58.0
    print((a < b))       # True   
    print((a << 2)())    # 168.0
