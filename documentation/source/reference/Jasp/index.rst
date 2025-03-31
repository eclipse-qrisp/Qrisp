.. _jasp:

.. image:: /_static/hybrid_realtime.png
    :align: center
    :width: 300

Jasp
====

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

Within Qiskit, real-time computations are in principle achievable by preparing a look-up table using the `c_if <https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.Instruction#c_if>`_ feature. Next to the clunkiness there are however scalability questions: For an error correction code that can extract $2^{100}$ syndroms, a single round of error correction would require a look-up table with $2^{100}$ entries, which is clearly not feasible. The `OpenQASM 3 specification <https://arxiv.org/abs/2104.14722>`_ elaborates the importance of real-time computations and defines the ``extern`` keyword. So far however neither Qiskit nor the OpenQASM demonstrate how the ``extern`` keyword can be used for executing classically established functions.

A more promising approach is the `QIR specification <https://www.qir-alliance.org/>`_ which integrates quantum related data-types into the widely used `LLVM IR <https://en.wikipedia.org/wiki/LLVM>`_. Compiling QIR not only has the advantage of incorporating a wide ecosystem of classically established code but also leveraging highly optimized compilation libraries.

With Jasp, we therefore aim to tackle both problems - compilation speed and lack of real-time computations - by targeting the established LLVM toolchain.

Ideally we want you to keep all your Qrisp code the same and simply enable the Jasp feature. So if you simply want to speed up your code, you can stop reading here. Otherwise get ready for more background knowledge (yay)!

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

Jax not only allows us to represent (classical) computations in a more simplified and easier-to-process form but also provides a `matured ecosystem <https://www.educative.io/courses/intro-jax-deep-learning/awesome-jax-libraries>`_ of libraries. On top of that, Jax exposes the means to `create new primitives <https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html>`_, which allows quantum developers to use the Jax infrastructure for their purposes.

What is Jasp?
^^^^^^^^^^^^^

Jasp is a module that provides Jax primitives for Qrisp syntax and therefore makes Qrisp Jax-traceable. How does this work in practice? The central class here is the ``Jaspr``, which is a subtype of the Jaxpr. Similarly to Jaxprs, Jasprs can be create using the ``make_jaspr`` function.

::
    
    from qrisp import *
    from qrisp.jasp import make_jaspr
    
    def test_fun(i):
        
        qv = QuantumFloat(i, -1)
        x(qv[0])
        cx(qv[0], qv[i-1])
        meas_res = measure(qv)
        return meas_res
        
    
    jaspr = make_jaspr(test_fun)(4)
    print(jaspr)
    
This will give you the following output:

.. code-block::

    { lambda ; a:QuantumCircuit b:i32[]. let
        c:QuantumCircuit d:QubitArray = create_qubits a b
        e:Qubit = get_qubit d 0
        f:QuantumCircuit = x c e
        g:Qubit = get_qubit d 0
        h:i32[] = sub b 1
        i:Qubit = get_qubit d h
        j:QuantumCircuit = cx f g i
        k:QuantumCircuit l:i32[] = measure j d
        m:f32[] = convert_element_type[new_dtype=float32 weak_type=True] l
        n:f32[] = mul m 0.5
      in (k, n) }
      
Assuming you already have some `understanding of the Jaxpr language <https://jax.readthedocs.io/en/latest/_tutorials/jaxpr.html>`_ you see a function, that receives a ``QuantumCircuit`` and an integer, does some processing and then returns a ``QuantumCircuit`` and a float. Here you can see one of the defining features of jasprs: They always receive and return a ``QuantumCircuit`` within their signature.

Furthermore it is interesting to note, that you can already see some real-time computation happening there: The result of the measurement is an integer (compared to a `ClBit as in Qiskit <https://docs.quantum.ibm.com/api/qiskit/circuit#clbit>`_) and is decoded according to the decoder by multiplying with $0.5$. In subsequent parts of the program, this float could be processed by literally any other Jax component.

jasprs can be simulated using the built-in real-time simulator. You achieve this by calling the jaspr like a function:

>>> print(jaspr(4))
4.5
>>> print(jaspr(8))
64.5

How is this different from the regular Qrisp features? The essential point is that because jaspr objects are embedded into the Jaxpr IR, they allow more advanced compilation tools to process the algorithm. In our case it is possible to convert jaspr objects to the `Catalyst <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`__ representation, which targets `established classical compilation infrastructure <https://mlir.llvm.org/>`_ to lower the (hybrid) algorithm into `QIR <https://www.qir-alliance.org/>`_.
If you are interested in how the QIR representation looks like, try calling

>>> jaspr.to_qir(8)

.. toctree::
   :maxdepth: 2
   :hidden:
   
   Jaspr
   Quantum Kernel
   qache
   Control Flow/index
   Sampling
   Expectation Value
   Resource Estimation
   Simulation Tools/index
   Ported Features
