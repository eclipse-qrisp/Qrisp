.. _static_vs_dynamic:

Standard Python Execution Mode vs Jasp Mode
============================================

Qrisp supports two execution modes that share the same high-level code but differ in how that code is processed.

**Static mode** (Standard Python Execution Mode)
    :doc:`QuantumVariable </reference/Core/QuantumVariable>` allocations, gate applications, and classical conditionals execute
    immediately as Python encounters them, building a circuit instruction by instruction.

    No special decorators or functions are required; this is the standard way Qrisp
    operates.

**Dynamic mode (Jasp mode)**
    In dynamic mode, the same code is traced through `JAX <https://jax.readthedocs.io/>`_.
    Instead of running with concrete values, JAX sends *tracer* objects through the function
    and records every operation into a `Jaspr <https://qrisp.eu/reference/Jasp/Jaspr.html>`_,
    a JAX-compatible intermediate representation. The resulting IR can be lowered for
    scalable compilation and real-time classical computation.

    Dynamic mode is entered via :doc:`@jaspify </reference/Jasp/Simulation Tools/Jaspify>`, :doc:`make_jaspr </reference/Jasp/Simulation Tools/Jaspify>`
    decorator.

The same code can be used in both modes under certain circumstances. However, Python's dynamic semantics include several patterns that **cannot** be traced - they either raise an exception at trace time or silently produce incorrect results.

This page works through common Python patterns and shows which are portable and which require adjustment.


1. Your First Program
---------------------

The simplest Python program in static mode writes a string to output. Printing a string involves no quantum state, so it is a purely classical operation.

.. code-block:: python

    # Static mode: print executes immediately
    def hello_world():
        print("Hello, World")

    hello_world()
    # Output:

.. code-block:: text

    # Output:
    Hello, World

The same function works in dynamic mode because the string is a concrete value, not a tracer:

.. code-block:: python

    # Dynamic mode: same code, traced through Jasp
    from qrisp.jasp import make_jaspr

    def hello_world():
        print("Hello, World")

    jaspr = make_jaspr(hello_world)()
    jaspr()
    # Output:

.. code-block:: text

    # Output:
    Hello, World

The return value of ``make_jaspr`` is a :doc:`Jaspr </reference/Jasp/Jaspr>` --- a JAX-compatible intermediate representation that captures the traced function. The Jaspr can be inspected, converted to other representations (`QIR <https://qrisp.eu/reference/Jasp/generated/qrisp.jasp.Jaspr.to_qir.html#qrisp.jasp.Jaspr.to_qir>`_, `MLIR <https://qrisp.eu/reference/Jasp/generated/qrisp.jasp.Jaspr.to_mlir.html#qrisp.jasp.Jaspr.to_mlir>`_, `QuantumCircuit <https://qrisp.eu/reference/Jasp/generated/qrisp.jasp.Jaspr.to_qc.html#qrisp.jasp.Jaspr.to_qc>`_), or executed. Calling ``print(jaspr)`` displays the internal Jaxpr IR.

Entry points into dynamic mode:

* ``make_jaspr(func)(*args)`` --- traces a function and returns a :doc:`Jaspr </reference/Jasp/Jaspr>` object that you can inspect, convert, or execute manually. Best when you need access to the Jaspr for analysis or conversion.
* ``@jaspify`` --- a decorator that handles tracing, caching, and execution transparently. Best for production code where you just want the result. The function is compiled on the first call and cached for subsequent calls.

In Jasp mode, when ``print()`` receives a tracer instead of a concrete string, it prints the tracer's representation, not the runtime value:

.. code-block:: python

    # Dynamic mode: print of a tracer shows repr, not value
    from qrisp.jasp import make_jaspr
    import jax.numpy as jnp

    def show_tracer(x):
        print("x is:", x)   # prints tracer repr during tracing
        return x + 1

    jaspr = make_jaspr(show_tracer)(5)   # trace: returns a Jaspr
    print("Result:", jaspr(5))            # execute: runs the compiled Jaspr
    # Output:

.. code-block:: text

    # Output:
    x is: JitTracer<int64[]>
    Result: 6

.. note::

   ``JitTracer<int64[]>`` is the string representation of a JAX tracer. ``int64[]`` means a scalar 64-bit signed integer, which is the default integer precision in JAX when x64 mode is enabled. JAX also supports ``float64``, ``float32``, and ``int32`` dtypes, but Jasp operations typically use 64-bit precision. For cases requiring integers beyond 64 bits, see the :doc:`BigInteger <Scalable Integer Type>` type.

When to use ``print()`` vs ``jax.debug.print``:

* ``print()`` executes at **trace time**. For concrete (non-traced) values it shows the actual value. For traced values it shows the tracer repr (e.g., ``JitTracer<int64[]>``). Use it for debugging the trace itself or for purely classical code.
* `jax.debug.print() <https://docs.jax.dev/en/latest/_autosummary/jax.debug.print.html>`_ executes at **runtime** inside the compiled function. It shows the actual computed value of a tracer during execution. Use it when you need to inspect runtime values inside a traced function.

The following example uses ``@jaspify`` to demonstrate both:

.. code-block:: python

    # jaspify compiles the function through JAX, activating jax.debug.print
    from qrisp.jasp import jaspify
    import jax
    import jax.numpy as jnp

    @jaspify
    def demo_print(x):
        print("Trace time:", x)             # shows tracer repr
        jax.debug.print("Runtime: {}", x)   # shows actual value
        return x + 1

    result = demo_print(42)
    print("Returned:", result)
    # Output:

.. code-block:: text

    # Output:
    Trace time: JitTracer<int64[]>
    Runtime: 42
    Returned: 43



2. Standard Math Operations
----------------------------

Standard Python operators (``+``, ``-``, ``*``, ``==``, ``<``, etc.) are available in Jasp mode for most quantum data types. Support varies by type: see the respective :doc:`QuantumFloat </reference/Quantum Types/QuantumFloat>`, :doc:`QuantumModulus </reference/Quantum Types/QuantumModulus>`, :doc:`QuantumBool </reference/Quantum Types/QuantumBool>`, and related documentation for details on which operators are supported in dynamic mode.

2.1 Using ``jnp`` Instead of ``numpy``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When writing Jasp code, use `jax.numpy <https://docs.jax.dev/en/latest/jax.numpy.html>`_ (``jnp``) instead of standard `numpy <https://numpy.org/doc/stable/>`_ (``np``) or the `math <https://docs.python.org/3/library/math.html>`_ module for operations on traced values. ``numpy`` functions expect concrete values and fail on tracers, while ``jnp`` functions are JAX-aware and work transparently in dynamic mode.

Calling ``numpy`` functions on a traced value fails:

.. code-block:: python

    # numpy functions fail on tracers
    from qrisp.jasp import make_jaspr
    import numpy as np

    def compute_bad(x):
        return np.sin(x)   # expects concrete value

    try:
        jaspr = make_jaspr(compute_bad)(1.0)
    except Exception as e:
        print(f"{type(e).__name__}: numpy on tracer breaks")


.. code-block:: text

    # Output:
    TracerBoolConversionError: numpy on tracer breaks

As expected, using ``jnp`` instead works. Here we evaluate the trigonometric identity :math:`\sin^2 x + \cos^2 x = 1` on a traced value:

.. code-block:: python

    # Works with jnp instead
    from qrisp.jasp import make_jaspr
    import jax.numpy as jnp

    def compute(x):
        return jnp.sin(x)**2 + jnp.cos(x)**2

    jaspr = make_jaspr(compute)(1.0)
    print("Result:", jaspr(1.0))


.. code-block:: text

    # Output:
    Result: 1.0

2.2 Boolean Operators
~~~~~~~~~~~~~~~~~~~~~~

The ``and``, ``or``, and ``not`` operators require concrete Python bools. On traced values they raise ``TracerBoolConversionError``.

Here we define a function to verify whether the provided year is a leap year using ``and`` and ``or``:

.. code-block:: python

    # and/or/not on traced values - breaks
    from qrisp.jasp import make_jaspr

    def leap_year_bad(year):
        return (year % 400 == 0) or ((year % 4 == 0) and (year % 100 != 0))

    try:
        jaspr = make_jaspr(leap_year_bad)(2024)
    except Exception as e:
        print(f"{type(e).__name__}: traced bool in `or`/`and` breaks")


.. code-block:: text

    # Output:
    TracerBoolConversionError: traced bool in `or`/`and` breaks

Fix the leap year function by using ``jnp.logical_or`` and ``jnp.logical_and``, which operate on traced bools and compile to the correct runtime logic:

.. code-block:: python

    # Fix: use jnp.logical_and / jnp.logical_or for traced values
    from qrisp.jasp import make_jaspr
    import jax.numpy as jnp

    def leap_year_fixed(year):
        return jnp.logical_or(
            year % 400 == 0,
            jnp.logical_and(year % 4 == 0, year % 100 != 0)
        )

    jaspr = make_jaspr(leap_year_fixed)(2024)
    print("2024 is a leap year:", jaspr(2024))


.. code-block:: text

    # Output:
    2024 is a leap year: True