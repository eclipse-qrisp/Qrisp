.. _static_vs_dynamic:

Static vs. Dynamic Mode
========================

Qrisp supports two execution modes that share the same high-level code but differ in how that code
is processed.

**Static mode** (Standard Python Execution Mode)
    :doc:`QuantumVariable </reference/Core/QuantumVariable>` allocations, gate applications,
    and classical conditionals execute immediately as Python encounters them, building a
    circuit instruction by instruction.

    No special decorators or functions are required; this is the standard way Qrisp operates.

**Dynamic mode (Jasp mode)**
    In dynamic mode, the same code is traced through `JAX <https://jax.readthedocs.io/>`_.
    Instead of running with concrete values, JAX sends *tracer* objects through the function
    and records every operation into a `Jaspr <https://qrisp.eu/reference/Jasp/Jaspr.html>`_, a
    JAX-compatible intermediate representation. The resulting IR can be lowered for scalable
    compilation and real-time classical computation.

    Dynamic mode is entered via :doc:`@jaspify </reference/Jasp/Simulation Tools/Jaspify>` or
    :doc:`make_jaspr </reference/Jasp/Simulation Tools/Jaspify>`.

The same code can work in both modes under certain circumstances. However, Python's dynamic
semantics include several patterns that **cannot** be traced --- they either raise an exception at
trace time or silently produce incorrect results.

This page catalogues those patterns and shows what to use instead.

Entry points into dynamic mode:

* ``make_jaspr(func)(*args)`` --- traces a function and returns a :doc:`Jaspr
  </reference/Jasp/Jaspr>` object that you can inspect, convert, or execute manually. Best when
  you need access to the Jaspr for analysis or conversion.
* ``@jaspify`` --- a decorator that handles tracing, caching, and execution transparently. Best
  for production code where you just want the result. The function is compiled on the first call
  and cached for subsequent calls.

The return value of ``make_jaspr`` is a :doc:`Jaspr </reference/Jasp/Jaspr>` --- a JAX-compatible
intermediate representation that captures the traced function. The Jaspr can be inspected,
converted to other representations (`QIR
<https://qrisp.eu/reference/Jasp/generated/qrisp.jasp.Jaspr.to_qir.html#qrisp.jasp.Jaspr.to_qir>`_,
`MLIR <https://qrisp.eu/reference/Jasp/generated/qrisp.jasp.Jaspr.to_mlir.html#qrisp.jasp.Jaspr.to_
mlir>`_, `QuantumCircuit
<https://qrisp.eu/reference/Jasp/generated/qrisp.jasp.Jaspr.to_qc.html#qrisp.jasp.Jaspr.to_qc>`_),
or :doc:`executed </reference/Jasp/Simulation Tools/Jaspify>`.


Side effects and ``print``
------------------------------

If your function has side effects, JAX's tracer can cause unexpected behaviour. A common gotcha is
``print``:

.. code-block:: python

    from qrisp.jasp import make_jaspr
    import jax.numpy as jnp

    def show_tracer(x):
        print("x is:", x)   # prints tracer repr during tracing
        return x + 1

    jaspr = make_jaspr(show_tracer)(5)
    print("Result:", jaspr(5))

.. code-block:: text

    # Output:
    x is: JitTracer<int64[]>
    Result: 6

``print()`` executes at **trace time**. For concrete values it shows the actual value; for traced
values it shows the tracer repr. To inspect runtime values inside a traced function, use
`jax.debug.print() <https://docs.jax.dev/en/latest/_autosummary/jax.debug.print.html>`_:

.. code-block:: python

    from qrisp.jasp import jaspify
    import jax

    @jaspify
    def demo_print(x):
        print("Trace time:", x)             # shows tracer repr
        jax.debug.print("Runtime: {}", x)   # shows actual value
        return x + 1

    result = demo_print(42)
    print("Returned:", result)

.. code-block:: text

    # Output:
    Trace time: JitTracer<int64[]>
    Runtime: 42
    Returned: 43

.. note::

   ``JitTracer<int64[]>`` is the string representation of a JAX tracer. ``int64[]`` means a
   scalar 64-bit signed integer, the default integer precision in JAX when x64 mode is enabled.
   For cases requiring integers beyond 64 bits, see the :doc:`BigInteger <Scalable Integer
   Type>` type.


Pure functions --- globals, iterators, and mutation
-------------------------------------------------------

JAX transformations require functionally pure Python functions: all inputs arrive as parameters,
all outputs leave as return values. The ``print`` side effect was shown above. Here are three
additional patterns that break purity.

**Global variables** --- captured once at trace time, changes ignored afterwards

.. code-block:: python

    scale = 2.0

    def impure_global(x):
        return x * scale

    from qrisp.jasp import make_jaspr
    import jax.numpy as jnp

    jaspr = make_jaspr(impure_global)(jnp.float64(3.0))
    print("first:", jaspr(jnp.float64(3.0)))

    scale = 100.0  # change the global
    print("second:", jaspr(jnp.float64(3.0)))  # still uses old scale!

.. code-block:: text

    # Output:
    first: 6.0
    second: 6.0

The value of ``scale`` is baked into the Jaspr at trace time. Later mutations are invisible. Pass
the value as an explicit parameter instead:

.. code-block:: python

    from qrisp.jasp import make_jaspr
    import jax.numpy as jnp

    def pure_param(x, scale):
        return x * scale

    jaspr = make_jaspr(pure_param)(jnp.float64(3.0), jnp.float64(2.0))
    print(make_jaspr(pure_param)(jnp.float64(3.0), jnp.float64(100.0))(jnp.float64(3.0), jnp.float64(100.0)))

.. code-block:: text

    # Output:
    300.0

**Python iterators** --- stateful, incompatible with tracing

Iterators carry internal state (which element comes next) and therefore break referential
transparency:

.. code-block:: python

    from qrisp.jasp import make_jaspr
    import jax.numpy as jnp

    def impure_iterator():
        it = iter([10, 20, 30])
        return next(it) + next(it)

    jaspr = make_jaspr(impure_iterator)()

.. code-block:: text

    # Output:
    TypeError: 'ListIterator' object is not a valid JAX type

Replace iterators with array indexing or ``jax.lax`` primitives:

.. code-block:: python

    from qrisp.jasp import make_jaspr
    import jax.numpy as jnp

    def pure_array():
        arr = jnp.array([10, 20, 30])
        return arr[0] + arr[1]

    jaspr = make_jaspr(pure_array)()
    print("result:", jaspr())

.. code-block:: text

    # Output:
    result: 30

**Mutating external state** --- writes to outer scope are lost

Writing to a variable defined outside the function (e.g., appending to a global list or
incrementing a counter) also breaks purity:

.. code-block:: python

    from qrisp.jasp import make_jaspr
    import jax.numpy as jnp

    side_effects = []

    def impure_append(x):
        side_effects.append("called")  #  mutates outer list
        return x + 1

    jaspr = make_jaspr(impure_append)(jnp.int64(5))

.. code-block:: text

    # Output:
    Jax semantics changed during jrange iteration

The mutation is detected and raises an error at trace time. If it did succeed silently, the side
effect would happen during tracing only --- not during execution --- leading to subtle bugs. Keep
all mutation inside the function scope, using local variables and JAX's functional update patterns.


``jnp`` instead of ``numpy``
-------------------------------

When writing Jasp code, use `jax.numpy <https://docs.jax.dev/en/latest/jax.numpy.html>`_ (``jnp``)
instead of standard `numpy <https://numpy.org/doc/stable/>`_ (``np``) or the `math
<https://docs.python.org/3/library/math.html>`_ module for operations on traced values. ``numpy``
functions expect concrete values and fail on tracers:

.. code-block:: python

    from qrisp.jasp import make_jaspr
    import numpy as np

    def compute_bad(x):
        return np.sin(x)

    try:
        jaspr = make_jaspr(compute_bad)(1.0)
    except Exception as e:
        print(f"{type(e).__name__}: numpy on tracer breaks")

.. code-block:: text

    # Output:
    TracerBoolConversionError: numpy on tracer breaks

Use ``jnp`` instead:

.. code-block:: python

    from qrisp.jasp import make_jaspr
    import jax.numpy as jnp

    def compute(x):
        return jnp.sin(x)**2 + jnp.cos(x)**2

    jaspr = make_jaspr(compute)(1.0)
    print("Result:", jaspr(1.0))

.. code-block:: text

    # Output:
    Result: 1.0

Standard Python operators (``+``, ``-``, ``*``, ``==``, ``<``, etc.) are available in Jasp mode for
most quantum data types. Support varies by type: see the respective :doc:`QuantumFloat
</reference/Quantum Types/QuantumFloat>`, :doc:`QuantumModulus </reference/Quantum
Types/QuantumModulus>`, :doc:`QuantumBool </reference/Quantum Types/QuantumBool>`, and related
documentation for details.


Boolean operators
---------------------

The Python ``and``, ``or``, and ``not`` operators require concrete Python bools. On traced values
they raise ``TracerBoolConversionError``:

.. code-block:: python

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

Use ``jnp.logical_and`` / ``jnp.logical_or`` / ``jnp.logical_not`` instead:

.. code-block:: python

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


Conditionals and loops
==========================

This is where static and dynamic mode most strongly diverge. In static mode, all Python control
flow works normally. In dynamic mode, Jasp (via JAX) must record a single static computation graph
during tracing, so any construct whose branch or iteration count depends on a traced value fails at
trace time.

A **predicate** is a boolean expression used as a control-flow condition:

- **Classical predicate** --- all operands are plain Python values, known at trace time.
- **Traced predicate** --- depends on a JAX tracer (e.g. a quantum measurement result), unknown
  until execution.

Jasp wrappers vs. JAX primitives
-------------------------------------

Jasp provides two layers of control-flow replacements:

* **JAX primitives** (``jax.lax.cond``, ``jnp.where``): purely classical branching on traced
  values. Branches cannot contain quantum operations.
* **Jasp wrappers** (:func:`~qrisp.jasp.q_cond`, :func:`~qrisp.jasp.q_switch`,
  :func:`~qrisp.jasp.q_while_loop`, :func:`~qrisp.jasp.q_fori_loop`): extend the JAX primitives
  with quantum state tracking. Branches can contain quantum operations, but all must have the
  same output signature.

All branches must have the same output signature. Variables used inside a branch must appear in its
argument list. Unlike ``jrange``, ``q_while_loop`` and ``q_fori_loop`` support carry values; their
condition function must not contain quantum operations.


``if``/``else`` with a classical predicate
-------------------------------------------

``if`` / ``else`` with a classical predicate works in both modes:

.. code-block:: python

   # Static mode
   x = 5
   if x > 0:
       print("Static mode: positive")

   # Dynamic mode
   from qrisp.jasp import jaspify

   @jaspify
   def example():
       x = 5
       if x > 0:
           print("Dynamic mode: positive")

   example()

.. code-block:: text

   # Output:
   Static mode: positive
   Dynamic mode: positive


``if``/``else`` with a traced predicate
-------------------------------------------

``if`` / ``else`` with a **traced** predicate fails in dynamic mode. Choose the replacement based
on what the branches do.

**Classical branches** (no quantum operations): use `jax.lax.cond
<https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.cond.html>`_. Both branches compile
ahead of time; the correct result is selected at execution.

.. code-block:: python

   from jax import lax
   from qrisp.jasp import make_jaspr
   import jax.numpy as jnp

   def safe_abs(x):
       return lax.cond(x > 0, lambda: x, lambda: -x)

   jaspr = make_jaspr(safe_abs)(jnp.array(-3))
   print("Absolute value of -3:", jaspr(jnp.array(-3)))

.. code-block:: text

   # Output:
   Absolute value of -3: 3

**Quantum branches** (branches contain quantum operations): use :func:`~qrisp.jasp.q_cond`.

.. code-block:: python

   from qrisp import QuantumBool, h, measure, x
   from qrisp.jasp import make_jaspr, q_cond

   def q_cond_demo():
       qb = QuantumBool()
       h(qb)
       pred = measure(qb)
       qb = q_cond(pred,
                   lambda qb: x(qb),   # True: flip qubit
                   lambda qb: qb,      # False: leave as-is
                   qb)
       return measure(qb)

   jaspr = make_jaspr(q_cond_demo)()
   print("Result:", jaspr())

Since the X gate undoes the True outcome (:math:`|1\rangle` back to :math:`|0\rangle`), the final
measurement is always ``False``.

.. code-block:: text

   # Output:
   Result: False

**Quantum-controlled gates**: use :ref:`with control(...) <ControlEnvironment>`. Every quantum
operation inside the block is conditioned on the control qubit(s) being in the :math:`|1\rangle`
state.

.. code-block:: python

   from qrisp import QuantumBool, h, measure, x, control
   from qrisp.jasp import jaspify

   def control_example():
       qbl = QuantumBool()
       h(qbl)
       with control(qbl):
           x(qbl)
       return measure(qbl)

   print(jaspify(control_example)())

.. code-block:: text

   # Output:
   0.0

**Classical control on measured values**: use :ref:`ClControlEnvironment <ClControlEnvironment>`
with the same ``with control(...)`` syntax.

.. code-block:: python

   from qrisp import QuantumFloat, x, measure, control
   from qrisp.jasp import make_jaspr

   def classical_control(i):
       a = QuantumFloat(3)
       a[:] = i
       b = measure(a)
       with control(b == 4):
           x(a[0])
       return measure(a)

   jaspr = make_jaspr(classical_control)(1)
   print("Input 1:", jaspr(1))
   print("Input 4:", jaspr(4))

With input ``1``, ``b == 4`` is ``False`` and the X gate is skipped. With input ``4``, the X gate
flips qubit 0, changing the value from ``4`` to ``5``.

.. code-block:: text

   # Output:
   Input 1: 1
   Input 4: 5


``while`` loops
-------------------

``while`` with a plain Python expression (no traced values) works in both modes:

.. code-block:: python

   # Static mode
   i = 0
   while i < 4:
       i += 1
   print("Static mode:", i)

   # Dynamic mode
   from qrisp.jasp import jaspify

   @jaspify
   def while_demo():
       i = 0
       while i < 4:
           i += 1
       return i

   print("Dynamic mode:", while_demo())

.. code-block:: text

   # Output:
   Static mode: 4
   Dynamic mode: 4

``while`` with a **traced** condition fails because ``while`` needs a concrete Python ``bool``:

.. code-block:: python

   from qrisp import QuantumFloat
   from qrisp.jasp import make_jaspr

   def bad_while():
       qf = QuantumFloat(4)
       i = 0
       while i < 4:    # i is traced
           i += 1

   jaspr = make_jaspr(bad_while)()

.. code-block:: text

   # Output:
   TracerBoolConversionError: Attempted to convert a tracer to a bool

Use one of the following replacements:

``jrange`` **(no carry)** --- each iteration starts fresh; you cannot pass a value from one
iteration to the next. Good for simple counters. See :ref:`jrange restrictions
<jrange_restrictions>` for limitations.

.. code-block:: python

   from qrisp import QuantumFloat, measure
   from qrisp.jasp import make_jaspr, jrange

   def good_while_jrange():
       qf = QuantumFloat(4)
       for _ in jrange(4):
           pass
       return measure(qf)

   jaspr = make_jaspr(good_while_jrange)()
   print("result:", jaspr())

.. code-block:: text

   # Output:
   result: 0.0

:func:`~qrisp.jasp.q_while_loop` **(with carry)** --- values are passed across iterations. Use when
you need to track state (e.g. a :class:`QuantumFloat` or a counter) as the loop runs.

.. code-block:: python

   from qrisp import QuantumFloat, measure
   from qrisp.jasp import make_jaspr, q_while_loop

   def good_while():
       qf = QuantumFloat(4)
       def cond(val):
           i, _ = val
           return i < 4
       def body(val):
           i, qf = val
           return i + 1, qf
       _, qf = q_while_loop(cond, body, (0, qf))
       return measure(qf)

   jaspr = make_jaspr(good_while)()
   print("result:", jaspr())

.. code-block:: text

   # Output:
   result: 0.0


``for`` loops
-----------------

``for i in range(n)`` with a plain Python ``int`` works in both modes:

.. code-block:: python

   # Static mode
   for i in range(4):
       pass
   print("Static mode:", i)

   # Dynamic mode
   from qrisp.jasp import jaspify

   @jaspify
   def range_demo():
       for i in range(4):
           pass
       return i

   print("Dynamic mode:", range_demo())

.. code-block:: text

   # Output:
   Static mode: 3
   Dynamic mode: 3

``for i in range(traced_n)`` fails because Python's ``range`` requires a concrete integer:

.. code-block:: python

   from qrisp import QuantumFloat, h, measure
   from qrisp.jasp import make_jaspr

   def bad_range(n):
       for i in range(n):   # n is traced
           pass

   jaspr = make_jaspr(bad_range)(3)

.. code-block:: text

   # Output:
   TracerBoolConversionError: Attempted to convert a tracer to a bool

Replace with ``jrange`` (no carry) or :func:`~qrisp.jasp.q_fori_loop` (with carry):

.. code-block:: python

   from qrisp import QuantumFloat, h, measure
   from qrisp.jasp import make_jaspr, jrange

   def good_range(n):
       qf = QuantumFloat(4)
       h(qf[0])
       for i in jrange(n):
           pass
       return measure(qf)

   jaspr = make_jaspr(good_range)(3)
   print("result:", jaspr(3))

.. code-block:: text

   # Output:
   result: 1.0


Unsupported control flow in dynamic mode
--------------------------------------------

``break`` and ``continue`` work in static mode but fail in dynamic mode because they alter control
flow at runtime in a way JAX cannot represent in a single static trace. Inside a ``jrange`` loop,
``break`` exits the loop before the second required trace completes:

.. code-block:: python

   from qrisp.jasp import make_jaspr, jrange
   from qrisp import QuantumFloat, x, measure

   def bad_break(n):
       qf = QuantumFloat(4)
       for i in jrange(10):
           x(qf[0])
           break
       return measure(qf)

   jaspr = make_jaspr(bad_break)(3)

.. code-block:: text

   # Output:
   KeyError: Var(id=...):QuantumState

Replace with ``jrange`` and let the iteration count express the bound directly:

.. code-block:: python

   from qrisp import QuantumFloat, x, measure
   from qrisp.jasp import make_jaspr, jrange

   def good_break(n):
       qf = QuantumFloat(4)
       for i in jrange(n):
           x(qf[0])
       return measure(qf)

   jaspr = make_jaspr(good_break)(3)
   print("result:", jaspr(3))

.. code-block:: text

   # Output:
   result: 1.0

``try`` / ``except`` always fails in dynamic mode. The ``except`` branch is never entered at trace
time (no error occurs with tracer values), so JAX cannot record it. The code below silently
ignores the ``except`` branch — at runtime ``n = 0`` returns ``inf`` instead of ``0``:

.. code-block:: python

   from qrisp.jasp import make_jaspr

   def bad_try(n):
       try:
           result = 1 / n
       except ZeroDivisionError:
           result = 0
       return result

   jaspr = make_jaspr(bad_try)(3)
   print("n = 3:", jaspr(3))
   print("n = 0:", jaspr(0))

.. code-block:: text

   # Output:
   n = 3: 0.3333333333333333
   n = 0: inf    # except was NOT compiled

No replacement exists; restructure the code to avoid exception handling inside traced functions.

``match`` / ``case`` fails in dynamic mode on traced values because Python's ``match`` requires
concrete patterns to compare against. The code below matches a traced value against integer cases,
which produces a traced boolean:

.. code-block:: python

   from qrisp.jasp import make_jaspr

   def bad_match(n):
       match n:
           case 0:
               result = "zero"
           case 1:
               result = "one"
           case _:
               result = "other"
       return result

   jaspr = make_jaspr(bad_match)(3)

.. code-block:: text

   # Output:
   TracerBoolConversionError: Attempted boolean conversion of traced array

Use :func:`~qrisp.jasp.q_switch` instead. It takes a traced index, a list of case functions, and the
quantum variables they operate on. The function at the matching index is compiled and executed.

The example below measures a 2-qubit register to get a random index (0-3), then applies the
matching case function. Only one of the four cases applies ``h(qf)``, so the output depends on
which index was measured:

.. code-block:: python

   from qrisp import QuantumFloat, h, measure
   from qrisp.jasp import make_jaspr, q_switch
   import jax.numpy as jnp

   def f0(qf): return qf          # case 0: no gate
   def f1(qf): h(qf); return qf   # case 1: apply H
   def f2(qf): return qf          # case 2: no gate
   def f3(qf): return qf          # case 3: no gate

   def good_switch():
       qf = QuantumFloat(2); h(qf)                 # prepare superposition
       idx = jnp.int32(measure(qf))                # random index 0-3
       q_switch(idx, [f0, f1, f2, f3], qf)         # run matching case
       return measure(qf)

   jaspr = make_jaspr(good_switch)()
   print(jaspr())

.. code-block:: text

   # Output:
   (result depends on which index was measured)



Extended ``jrange`` example
-------------------------------

``jrange`` replaces ``range(n)``, ``break``/``continue``, and convergence loops where each
iteration is independent (no carry values). Each example below shows a different use.

**Quantum gate repetition.** Apply a gate ``n`` times inside a traced function. Each application is
independent, so ``jrange`` works:

.. code-block:: python

   from qrisp import QuantumFloat, x, measure
   from qrisp.jasp import make_jaspr, jrange

   def flip_n(n):
       qf = QuantumFloat(4)
       for i in jrange(n):
           x(qf[0])
       return measure(qf)

   jaspr = make_jaspr(flip_n)(3)
   print("flip_n :", jaspr(3))

.. code-block:: text

   # Output:
   flip_n : 1.0

**Fixed-iteration algorithms.** When the number of iterations is a traced value but each step is
self-contained, use ``jrange``. The example below applies a Hadamard to one qubit per iteration:

.. code-block:: python

   from qrisp import QuantumFloat, h, measure
   from qrisp.jasp import make_jaspr, jrange

   def apply_h(n):
       qf = QuantumFloat(4)
       for i in jrange(n):
           h(qf[i])
       return measure(qf)

   jaspr = make_jaspr(apply_h)(3)
   print("apply_h :", jaspr(3))

.. code-block:: text

   # Output:
   apply_h : 3.0

**Convergence loops.** If the number of iterations is fixed and no carry values are needed, use
``jrange`` with a literal count. The example below applies an X gate 20 times inside a traced
function:

.. code-block:: python

   from qrisp import QuantumFloat, x, measure
   from qrisp.jasp import make_jaspr, jrange

   def repeat_gate():
       qf = QuantumFloat(4)
       for _ in jrange(20):
           x(qf[0])
       return measure(qf)

   jaspr = make_jaspr(repeat_gate)()
   print("repeat_gate :", jaspr())

.. code-block:: text

   # Output:
   repeat_gate : 0.0

For examples that need to thread state across iterations (such as accumulating a result), use
:func:`~qrisp.jasp.q_fori_loop` or
:func:`~qrisp.jasp.q_while_loop` instead.


.. _jrange_restrictions:

``jrange`` restrictions
---------------------------

``jrange`` traces the loop body **exactly twice**, compares the traces to determine which inputs
change per iteration, and compiles the result into a single ``jax.lax.while_loop``. This imposes
several restrictions:

* No external carry values (the loop index is the only value passed between iterations).
  Variables modified in the body must not be used outside.
* No iteration-dependent branching (the same JAX IR must result from both traces, so
  ``if``/``else`` on the loop index is not allowed).
* No ``continue``, ``break``, or ``return`` (these alter control flow mid-trace and cause the two
  traces to diverge).
* Dynamic stop/step is not supported (the bounds are baked in when ``__iter__`` is called).
* No conditional :class:`QuantumVariable` allocation (qubit bookkeeping must be consistent across
  traces).
* No nested ``jrange`` with a data-dependent inner bound (the inner loop would need re-tracing
  per outer iteration).
* No negative step (use index arithmetic instead, as shown in the reverse iteration example
   above).


Lists and arrays in dynamic mode
========================================

Dynamic mode traces through JAX, which replaces all values with tracer objects. Python lists
and standard NumPy operations expect concrete values and fail when they encounter tracers.
Use ``jax.numpy`` (``jnp``) and JAX's array primitives instead.

**List indexing with a traced index**

Python lists require a concrete integer index. A traced index raises
``TracerIntegerConversionError``:

.. code-block:: python

   from qrisp.jasp import make_jaspr

   def bad_index(n):
       items = [10, 20, 30]
       return items[n]       # n is traced

   jaspr = make_jaspr(bad_index)(1)

.. code-block:: text

   # Output:
   TracerIntegerConversionError: The __index__() method was called on traced array

Use a ``jnp`` array and ``jax.lax.dynamic_index_in_dim`` or ``jnp.take`` instead:

.. code-block:: python

   from qrisp.jasp import make_jaspr
   import jax.numpy as jnp

   def good_index(n):
       items = jnp.array([10, 20, 30])
       return jnp.take(items, n)

   jaspr = make_jaspr(good_index)(1)
   print("result:", jaspr(1))

.. code-block:: text

   # Output:
   result: 20

**The** ``in`` **operator on a list**

Python's ``in`` operator tries to compare the traced value against each list element,
producing a traced boolean that Python cannot evaluate:

.. code-block:: python

   from qrisp.jasp import make_jaspr

   def bad_in(n):
       items = [1, 2, 3, 4, 5]
       return n in items

   jaspr = make_jaspr(bad_in)(3)

.. code-block:: text

   # Output:
   TracerBoolConversionError: Attempted boolean conversion of traced array

Use ``jnp.any`` with an element-wise comparison instead:

.. code-block:: python

   from qrisp.jasp import make_jaspr
   import jax.numpy as jnp

   def good_in(n):
       items = jnp.array([1, 2, 3, 4, 5])
       return jnp.any(items == n)

   jaspr = make_jaspr(good_in)(3)
   print("result:", jaspr(3))

.. code-block:: text

   # Output:
   result: True

**Python max/min on traced values**

``max`` and ``min`` compare values using ``<`` and ``>``, which produce traced booleans
when applied to tracers. Python cannot evaluate the comparison:

.. code-block:: python

   from qrisp.jasp import make_jaspr

   def bad_max(n):
       return max(n, n + 5, n + 2)

   jaspr = make_jaspr(bad_max)(3)

.. code-block:: text

   # Output:
   TracerBoolConversionError: Attempted boolean conversion of traced array

Use ``jnp.maximum`` / ``jnp.minimum`` instead:

.. code-block:: python

   from qrisp.jasp import make_jaspr
   import jax.numpy as jnp

   def good_max(n):
       return jnp.maximum(jnp.maximum(n, n + 5), n + 2)

   jaspr = make_jaspr(good_max)(3)
   print("result:", jaspr(3))

.. code-block:: text

   # Output:
   result: 8

**Boolean (mask) indexing on jnp arrays**

Boolean indexing on a ``jnp`` array requires a concrete mask at trace time; a traced mask
raises ``NonConcreteBooleanIndexError``:

.. code-block:: python

   from qrisp.jasp import make_jaspr
   import jax.numpy as jnp

   def bad_mask(n):
       arr = jnp.array([1, 2, 3, 4, 5])
       mask = arr > n
       return arr[mask]       # mask is traced

   jaspr = make_jaspr(bad_mask)(2)

.. code-block:: text

   # Output:
   NonConcreteBooleanIndexError: Array boolean indices must be concrete

Use ``jnp.where`` to select between two arrays based on a traced condition:

.. code-block:: python

   from qrisp.jasp import make_jaspr
   import jax.numpy as jnp

   def good_mask(n):
       arr = jnp.array([1, 2, 3, 4, 5])
       mask = arr > n
       return jnp.where(mask, arr, 0)

   jaspr = make_jaspr(good_mask)(2)
   print("result:", jaspr(2))

.. code-block:: text

   # Output:
   result: [0 0 3 4 5]

**Dynamic slicing**

Python / NumPy slice syntax requires static start/stop/step values; traced bounds raise an
error:

.. code-block:: python

   from qrisp.jasp import make_jaspr
   import jax.numpy as jnp

   def bad_slice(n):
       arr = jnp.array([1, 2, 3, 4, 5])
       return arr[n:]         # n is traced

   jaspr = make_jaspr(bad_slice)(2)

.. code-block:: text

   # Output:
   IndexError: Array slice indices must have static start/stop/step

Use `jax.lax.dynamic_slice <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dynamic_slice.html>`_ or `jax.lax.slice_in_dim <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.slice_in_dim.html>`_ instead:

.. code-block:: python

   from qrisp.jasp import make_jaspr
   import jax.lax as lax
   import jax.numpy as jnp

   def good_slice(n):
       arr = jnp.array([1, 2, 3, 4, 5])
       return lax.dynamic_slice(arr, (n,), (3,))

   jaspr = make_jaspr(good_slice)(2)
   print("result:", jaspr(2))

.. code-block:: text

   # Output:
   result: [3 4 5]
