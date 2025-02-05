.. _jasp_tutorial:

How to think in Jasp
====================

What is Jasp and why do we need it?
-----------------------------------


Jasp is a submodule of Qrisp that allows you to scale up your Qrisp code to to practically relevant problem sizes. The fundamental problem that many Python based quantum frameworks face is that the Python interpreter is slow compared to what is possible with compiled languages. As an example:

::

    from qrisp import *
    N = 19326409253
    qm = QuantumModulus(N, inpl_adder = gidney_adder)
    qm[:] = 293587334
    qm *= 2345747462
    
This snippet demonstrates a 35 bit modular in-place multiplication which already takes ~20 seconds to compile. Considering typical RSA key sizes contain up to 2000 bits, compiling a circuit addressing practically relevant problem scales therefore seems unlikely. Note that this issue is not restricted to Qrisp. We can also observe the same in Qiskit.

::

    from qiskit.circuit.library import RGQFTMultiplier
    
    n = 50
    multiplication_circuit = RGQFTMultiplier(n, 2*n)
    
This snippet compiles a 50 bit (non-modular) multiplication circuit using Qiskit and also takes approximately 20 seconds. Using classical compilation infrastructure, a classical 64 bit multiplication can be compiled within micro-seconds, which hints at the large disparity in compilation speed.

Real-time computations
^^^^^^^^^^^^^^^^^^^^^^

Apart from the compilation scaling issues, many frameworks (Qrisp included) suffer from the inability to integrate classical real-time computations. What is a real-time computation? A classical computation that happens during the quantum computation, while the quantum computer stays in superposition. This computation has to happen much faster than the coherence time, so performing that computation by waiting for the Python interpreter is impossible. Real-time computations are essential for many techniques in error correction, such as `syndrom decoding <https://thesis.library.caltech.edu/2900/2/THESIS.pdf>`_ or `magic state distillation <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.020341>`_. On the algorithmic level, real-time computations also become more popular since they are so much cheaper than the quantum equivalent. Examples are `Gidney's adder <https://arxiv.org/abs/1709.06648>`_ or repeat until success protocols like `HHL <https://arxiv.org/abs/0811.3171>`_.

Within Qiskit, real-time computations are in principle achievable by preparing a look-up table using the `c_if <https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.Instruction#c_if>`_ feature. Next to the clunkiness there are however scalability questions: For an error correction code that can extract $2^{100}$ possible syndroms, a single round of error correction would require a look-up table with $2^{100}$ entries, which is clearly not feasible. The `OpenQASM 3 specification <https://arxiv.org/abs/2104.14722>`_ elaborates the importance of real-time computations and defines the ``extern`` keyword. So far however neither Qiskit nor the OpenQASM demonstrate how the ``extern`` keyword can be used for executing classically established functions.

A more promising approach is the `QIR specification <https://www.qir-alliance.org/>`_ which integrates quantum related data-types into the widely used `LLVM IR <https://en.wikipedia.org/wiki/LLVM>`_. Compiling QIR not only has the advantage of incorporating a wide ecosystem of classically established code but also leveraging highly optimized compilation libraries.

With Jasp, we therefore aim to tackle both problems - compilation speed and lack of real-time computations - by targeting the established LLVM toolchain.

Ideally we want you to keep all your Qrisp code the same and simply enable the Jasp.

.. note::

    Does this make "regular" Qrisp old news? Absolutely not! 
    
    As with many Pythonic ``jit`` compilation approaches, a vitally important feature is to execute the code in non-jitted mode for development/maintainance. This is because code introspection is much harder in ``jit`` mode since it is difficult to investigate intermediate results for bug-fixing purposes. The same applies to Jasp: Most Jasp and Qrisp code is interoperable, such that it is very easy to identify bugs using Qrisp mode. The relationship between the two compilation pipelines is therefore:
    
    .. list-table::
       :widths: 25 25
       :header-rows: 1

       * - **Qrisp**
         - **Jasp**
       * - Development
         - Scalability
       * - Maintainance
         - Compilation performance
       * - Fast prototyping
         - Real-time computation embedding
    

What is Jax?
^^^^^^^^^^^^

To understand how to fully leverage the Jasp module, you need a basic understanding of `Jax <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html>`_. Jax is a framework developed by Google, which aims to address a similar set of problems as described above but in the context of machine learning. Essentially Jax makes Python code for ML applications run faster, by leveraging a mechanism called tracing. Tracing means that instead of executing a Python function with actual values, Jax sends so called Tracers through the function, which keep a "record" of what would have happened to the values. This record is a mini functional programming language called `Jaxpr <https://jax.readthedocs.io/en/latest/_tutorials/jaxpr.html>`_. Creating a Jaxprs can be achieved by calling the ``make_jaxpr`` function.

::

    import jax.numpy as jnp
    
    def test_f(x):
        y = x + 2
        z = jnp.sin(y)
        return y*z
        
    from jax import make_jaxpr
    x = jnp.array([1.,2.,3.])
    print(make_jaxpr(test_f)(x))
    
This gives the output

.. code-block::

    { lambda ; a:f32[3]. let
    b:f32[3] = add a 2.0
    c:f32[3] = sin b
    d:f32[3] = mul b c
    in (d,) }

Note the ``jax.numpy`` import! This module is a ``numpy`` clone that is Jax traceable. This module is particularly helpful for Jasp algorithms, since it exposes the power of numpy for classical real-time computations in the middle of a quantum algorithm.

Jax not only allows us to represent (classical) computations in a more simplified and easier-to-process form but also provides a `matured ecosystem <https://www.educative.io/courses/intro-jax-deep-learning/awesome-jax-libraries>`_ of libraries. 

What is Jasp?
^^^^^^^^^^^^^

Jasp is a module that provides Jax primitives for Qrisp syntax and therefore makes Qrisp Jax-traceable. How does this work in practice? The central class here is the :ref:`Jaspr`, which is a subtype of the Jaxpr. Similarly to Jaxprs, Jasprs can be created using the ``make_jaspr`` function.

::

    from qrisp import *

    def main(i):
        qf = QuantumFloat(i)
        h(qf[0])
        cx(qf[0], qf[1])

        meas_float = measure(qf)

        return meas_float
        

    jaspr = make_jaspr(main)(5)

    print(jaspr)

.. code-block::

    { lambda ; a:QuantumCircuit b:i64[]. let
        c:QuantumCircuit d:QubitArray = jasp.create_qubits a b
        e:Qubit = jasp.get_qubit d 0
        f:QuantumCircuit = jasp.h c e
        g:Qubit = jasp.get_qubit d 1
        h:QuantumCircuit = jasp.cx f e g
        i:QuantumCircuit j:i64[] = jasp.measure h d
        k:QuantumCircuit = jasp.reset i d
        l:QuantumCircuit = jasp.delete_qubits k d
      in (l, j) }

Jasp programs can be executed with the Jasp interpreter by calling them like a function
::

    print(jaspr(5))
    # Yields: 0 or 3

A quicker way to do this is to use the :meth:`jaspify <qrisp.jasp.jaspify>` decorator. This decorator automatically transforms the function into a Jaspr and calls the simulator

::
    
    @jaspify
    def main(i):
        qf = QuantumFloat(i)
        h(qf[0])
        cx(qf[0], qf[1])

        meas_float = measure(qf)

        return meas_float

    print(main(5))
    # Yields: 0 or 3


Jasp programs can be compiled to `QIR <https://github.com/qir-alliance/qir-spec>`_, which is one of the most popular low-level representations for quantum computers. This possible because Jasp has a deeply integrated support for `Catalyst <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`_. In order to compile to QIR please install the package (only on Mac & Linux).

::

    try:
        import catalyst
    except:
        !pip install pennylane-catalyst


::

    qir_string = jaspr.to_qir()
    print(qir_string[:2500])


Here we printed only the first "few" lines since the entire string is very long.

The Qache decorator
-------------------

One of the most powerful features of Jasp is that it is fully dynamic, allowing many functions to be cached and reused. For this we have the :meth:`qache <qrisp.jasp.qache>` decorator. Qached functions are only excuted once (per calling signature) and otherwise retrieved from cache.

::

    import time

    @qache
    def inner_function(qv, i):
        cx(qv[0], qv[1])
        h(qv[i])
        # Complicated compilation, that takes a lot of time
        time.sleep(1)

    def main(i):
        qv = QuantumFloat(i)

        inner_function(qv, 0)
        inner_function(qv, 1)
        inner_function(qv, 2)

        return measure(qv)


    t0 = time.time()
    jaspr = make_jaspr(main)(5)
    print(time.time()- t0)
    # Yields:
    # 1.0196595191955566


If a cached function is called with a different type (classical or quantum) the function will not be retrieved from cache but instead retraced. If called with the same signature, the appropriate implementation will be retrieved from the cache.

::

    @qache
    def inner_function(qv):
        x(qv)
        time.sleep(1)

    def main():
        qf = QuantumFloat(5)
        qbl = QuantumBool(5)

        inner_function(qf)
        inner_function(qf)
        inner_function(qbl)
        inner_function(qbl)

        return measure(qf)

    t0 = time.time()
    jaspr = make_jaspr(main)()
    print(time.time()- t0)
    # Yields:
    # 2.044877767562866


We see 2 seconds now because the ``inner_function`` has been traced twice: Once for the :ref:`QuantumFloat` and once for the :ref:`QuantumBool`.

Another important concept are dynamic values. Dynamic values are values that are only known at runtime (i.e. when the program is actually executed). This could be because the value is coming from a quantum measurement. Every QuantumVariable and it's ``.size`` attribute are dynamic. Furthermore classical values can also be dynamic. For classical values, we can use the Python native ``isinstance`` check for the ``jax.core.Tracer`` class, whether a variable is dynamic. Note that even though ``QuantumVariables`` behave dynamic, they are not tracers themselves.
::
    
    from jax.core import Tracer

    def main(i):
        print("i is dynamic?: ", isinstance(i, Tracer))
        
        qf = QuantumFloat(5)
        j = qf.size
        print("j is dynamic?: ", isinstance(i, Tracer))
        
        h(qf)
        k = measure(qf)
        print("k is dynamic?: ", isinstance(k, Tracer))

        # Regular Python integers are not dynamic
        l = 5
        print("l is dynamic?: ", isinstance(l, Tracer))

        # Arbitrary Python objects can be used within Jasp
        # but they are not dynamic
        import networkx as nx
        G = nx.DiGraph()
        G.add_edge(1,2)
        print("G is dynamic?: ", isinstance(l, Tracer))
        
        return k

    jaspr = make_jaspr(main)(5)
    # Yields:
    # i is dynamic?:  True
    # j is dynamic?:  True
    # k is dynamic?:  True
    # l is dynamic?:  False
    # G is dynamic?:  False


What is the advantage of dynamic values? Dynamical code is **scale invariant**! For this we can use the :meth:`jrange <qrisp.jasp.jrange>` iterator, which allows you to execute a dynamic amount of loop iterations. Some restrictions apply however (check the docs to see which).
::

    @jaspify
    def main(k):

        a = QuantumFloat(k)
        b = QuantumFloat(k)

        # Brings a into uniform superposition via Hadamard
        h(a)

        c = measure(a)

        # Excutes c iterations (i.e. depending the measurement outcome)
        for i in jrange(c):

            # Performs a quantum incrementation on b based on the measurement outcome
            b += c//5

        return measure(b)

    print(main(5))


It is possible to execute a multi-controlled X gate with a dynamic amount of controls.
::

    @jaspify
    def main(i, j, k):

        a = QuantumFloat(5)
        a[:] = i
        
        qbl = QuantumBool()

        # a[:j] is a dynamic amount of controls
        mcx(a[:j], qbl[0], ctrl_state = k)

        return measure(qbl)


This function encodes the integer ``i`` into a ``QuantumFloat`` and subsequently performs an MCX gate with control state ``k``. Therefore, we expect the function to return ``True`` if ``i == k`` and ``j > 5``.
::

    print(main(1, 6, 1))
    print(main(3, 6, 1))
    print(main(2, 1, 1))
    # Yields:
    # True
    # False
    # False


Classical control flow
----------------------

Jasp code can be conditioned on classically known values. For that we simply use the :ref:`control <ControlEnvironment>` feature from base-Qrisp but with dynamical, classical bools. Some restrictions apply (check the docs for more details).
::

    @jaspify
    def main():

        qf = QuantumFloat(3)
        h(qf)

        # This is a classical, dynamical int
        meas_res = measure(qf)

        # This is a classical, dynamical bool
        ctrl_bl = meas_res >= 4
        
        with control(ctrl_bl):
            qf -= 4

        return measure(qf)

    for i in range(5):
        print(main())
    # Yields
    # 2
    # 0
    # 2
    # 2
    # 3


The Repeat-Until-Success (RUS) decorator
----------------------------------------

RUS stands for Repeat-Until-Success and is an essential part for many quantum algorithms such as HHL or Linear Combination of Unitaries (LCU). As the name says the RUS component repeats a certain subroutine until a measurement yields ``True``. The RUS decorator should be applied to a ``trial_function``, which returns a classical bool as the first return value and some arbitrary other values. The trial function will be repeated until the classical bool is ``True``.

To demonstrate the RUS behavior, we initialize a GHZ state 

$\ket{\psi} = \frac{1}{\sqrt{2}} (\ket{00000} + \ket{11111})$

and measure the first qubit into a boolean value. This will be the value to cancel the repetition. This will collapse the GHZ state into either $\ket{00000}$ (which will cause a new repetition) or $\ket{11111} = \ket{31}$, which cancels the loop. After the repetition is canceled we are therefore guaranteed to have the latter state.
::

    from qrisp.jasp import RUS, make_jaspr
    from qrisp import QuantumFloat, h, cx, measure

    def init_GHZ(qf):
        h(qf[0])
        for i in jrange(1, qf.size):
            cx(qf[0], qf[i])

    @RUS
    def rus_trial_function():
        qf = QuantumFloat(5)

        init_GHZ(qf)
        
        cancelation_bool = measure(qf[0])
        
        return cancelation_bool, qf

    @jaspify
    def main():

        qf = rus_trial_function()

        return measure(qf)

    print(main())
    # Yieds:
    # 31.0 (the decimal equivalent of 11111)


Terminal sampling
-----------------

The :meth:`jaspify <qrisp.jasp.jaspify>` decorator executes one "shot". For many quantum algorithms we however need the distribution of shots. In principle we could execute a bunch of "jaspified" function calls, which is however not as scalable. For this situation we have the :meth:`terminal_sampling <qrisp.jasp.terminal_sampling>` decorator. To use this decorator we need a function that returns a ``QuantumVariable`` (instead of a classical measurement result). The decorator will then perform a (hybrid) simulation of the given script and subsequently sample from the distribution at the end.
::

    
    @RUS
    def rus_trial_function():
        qf = QuantumFloat(5)

        init_GHZ(qf)
        
        cancelation_bool = measure(qf[0])
        
        return cancelation_bool, qf

    @terminal_sampling
    def main():

        qf = rus_trial_function()
        h(qf[0])

        return qf

    print(main())
    # Yields:
    # {30.0: 0.5, 31.0: 0.5}


The ``terminal_sampling`` decorator requires some care however. Remember that it only samples from the distribution at the end of the algorithm. This distribution can depend on random chances that happened during the execution. We demonstrate faulty use in the following example.
::

    from qrisp import QuantumBool, measure, control

    @terminal_sampling
    def main():

        qbl = QuantumBool()
        qf = QuantumFloat(4)

        # Bring qbl into superposition
        h(qbl)

        # Perform a measure
        cl_bl = measure(qbl)

        # Perform a conditional operation based on the measurement outcome
        with control(cl_bl):
            qf[:] = 1
            h(qf[2])

        return qf

    for i in range(5):
        print(main())
    # Yields either {0.0: 1.0} or {1.0: 0.5, 5.0: 0.5} (with a 50/50 probability)


Boolean simulation
------------------
The tight Jax integration of Jasp enables some powerful features such as a highly performant simulator of purely boolean circuits. This simulator works by transforming Jaspr objects that contain only X, CX, MCX etc. into boolean Jax logic. Subsequently this is inserted into the Jax pipeline, which yields a highly scalable simulator for purely classical Jasp functions.

To call this simulator, we simply use the ``boolean_simulation`` decorator like we did with the ``jaspify`` decorator.
::

    from qrisp import *
    from qrisp.jasp import *

    def quantum_mult(a, b):
        return a*b

    @boolean_simulation(bit_array_padding = 2**10)
    def main(i, j, iterations):

        a = QuantumFloat(10)
        b = QuantumFloat(10)

        a[:] = i
        b[:] = j

        c = QuantumFloat(30)

        for i in jrange(iterations): 

            # Compute the quantum product
            temp = quantum_mult(a,b)

            # add into c
            c += temp

            # Uncompute the quantum product
            with invert():
                # The << operator "injects" the quantum variable into
                # the function. This means that the quantum_mult
                # function, which was originally out-of-place, is
                # now an in-place function operating on temp.

                # It can therefore be used for uncomputation
                # Automatic uncomputation is not yet available within Jasp.
                (temp << quantum_mult)(a, b)

            # Delete temp
            temp.delete()

        return measure(c)


The first call needs some time for compilation
::

    import time
    t0 = time.time()
    main(1, 2, 5)
    print(time.time()-t0)
    # Yields:
    # 8.607563018798828


Any subsequent call is super fast
::

    t0 = time.time()
    print(main(3, 4, 120)) # Expected to be 3*4*120 = 1440
    print(f"Took {time.time()-t0} to simulate 120 iterations")
    # Yields:
    # 1440.0
    # Took 0.006011247634887695 to simulate 120 iteration


Compile and simulate A MILLION QFLOPs!
::

    print(main(532, 233, 1000000))


Letting a classical neural network decide when to stop
-------------------------------------------------------

The following example showcases how a simple neural network can decide (in real-time) whether to go on or break the RUS iteration. For that we create a simple binary classifier and train it on dummy data (disclaimer: ML code by ChatGPT). This is code is not really useful in anyway and the classifier is classifying random data, but it shows how such an algorithm can be constructed and evaluated.
::

    import jax
    import jax.numpy as jnp
    from jax import grad, jit
    import optax

    # Define the model
    def model(params, x):
        W, b = params
        return jax.nn.sigmoid(jnp.dot(x, W) + b)

    # Define the loss function (binary cross-entropy)
    def loss_fn(params, x, y):
        preds = model(params, x)
        return -jnp.mean(y * jnp.log(preds) + (1 - y) * jnp.log(1 - preds))

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    W = jax.random.normal(key, (2, 1))
    b = jax.random.normal(key, (1,))
    params = (W, b)

    # Create optimizer
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(params)

    # Define training step
    @jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Generate some dummy data
    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (1000, 2))
    y = jnp.sum(X > 0, axis=1) % 2

    # Training loop
    for epoch in range(100):
        params, opt_state, loss = train_step(params, opt_state, X, y)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    # Make predictions
    predictions = model(params, X)
    accuracy = jnp.mean((predictions > 0.5) == y)
    print(f"Final accuracy: {accuracy}")


We can now use the ``model`` function to evaluate the classifier. Since this function is Jax-based it integrates seamlessly into Jasp.
::

    from qrisp.jasp import *
    from qrisp import *
    
    @RUS
    def rus_trial_function(params):

        # Sample data from two QuantumFloats.
        # This is a placeholder for an arbitrary quantum algorithm.
        qf_0 = QuantumFloat(5)
        h(qf_0)

        qf_1 = QuantumFloat(5)
        h(qf_1)

        meas_res_0 = measure(qf_0)
        meas_res_1 = measure(qf_1)

        # Turn the data into a Jax array
        X = jnp.array([meas_res_0,meas_res_1])/2**qf_0.size

        # Evaluate the model
        model_res = model(params, X)

        # Determine the cancelation
        cancelation_bool = (model_res > 0.5)[0]
        
        return cancelation_bool, qf_0

    @jaspify
    def main(params):

        qf = rus_trial_function(params)
        h(qf[0])

        return measure(qf)

    print(main(params))
    


Summary
=======
This marks the end of this tutorial! Here is a little summary of what you learned:
Jasp is a module in Qrisp that provides Jax primitives, making quantum programming Jax-traceable. It offers dynamic and flexible quantum computation features.

Core Components
---------------

**Jaspr Class**

- Subtype of Jaxpr.

- Created using the :ref:`make_jaspr <Jaspr>` function.

- Enables Jax-traceable quantum programming.

**Key Decorators**

1. :ref:`jaspify <jaspify>`

- Transforms functions into Jasprs.

- Simplifies quantum circuit creation and execution.

2. :ref:`qache <qache>`

- Enables function caching.

- Traces functions once per calling signature.

- Supports dynamic retracing for different types.

3. :ref:`RUS <RUS>` (Repeat-Until-Success).

- Repeats quantum subroutines until a condition is met.

- Used in algorithms like HHL and Linear Combination of Unitaries.

Dynamic Values
--------------

- Support for runtime-determined values.

- Enables scale-invariant code execution.

- Uses :ref:`jrange` for dynamic iteration.

Compilation Capabilities
------------------------

- Compilation to QIR (Quantum Intermediate Representation).

- Requires Catalyst package installation.

Simulation Modes
----------------

- :ref:`terminal_sampling` for distribution-level quantum simulation.

- :ref:`boolean_simulation` for high-performance classical circuit simulation.

- Other simulation modes in the :ref:`docs <jasp_simulators>`.

Technical Highlights
--------------------

- Full Jax integration.

- Dynamic quantum programming.

- Advanced caching mechanisms.

- Support for hybrid quantum-classical computations.


