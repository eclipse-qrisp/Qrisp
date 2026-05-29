.. _static_vs_dynamic:

Standard Python Execution Mode vs Jasp Mode
============================================

Qrisp supports two execution modes that share the same high-level code but differ in how that code is processed.

**Static mode** (Standard Python Execution Mode)
    QuantumVariable allocations, gate applications, and classical conditionals execute
    immediately as Python encounters them, building a circuit instruction by instruction.

    No special decorators or functions are required; this is the standard way Qrisp
    operates.

**Dynamic mode (Jasp)**
    In dynamic mode, the same code is traced through `JAX <https://jax.readthedocs.io/>`_.
    Instead of running with concrete values, JAX sends *tracer* objects through the function
    and records every operation into a `Jaspr <https://qrisp.eu/reference/Jasp/Jaspr.html>`_,
    a JAX-compatible intermediate representation. The resulting IR can be lowered for
    scalable compilation and real-time classical computation.

    Dynamic mode is entered via ``@jaspify``, ``make_jaspr``, or the ``QuantumKernel``
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

When ``print()`` receives a JAX tracer instead of a concrete string, it prints the tracer's representation, not the runtime value:

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

