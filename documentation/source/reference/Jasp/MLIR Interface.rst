.. _mlir_interface:

MLIR Interface
==============

Overview
--------

`MLIR (Multi-Level Intermediate Representation) <https://mlir.llvm.org/>`_ is a compiler infrastructure that enables the design and implementation of code generators, translators, and optimizers at different levels of abstraction. MLIR provides a flexible type system and extensible operation set that allows for the representation of various domain-specific abstractions.

Qrisp leverages MLIR to provide a standardized intermediate representation for quantum programs through the Jasp dialect. This enables interoperability with the broader MLIR ecosystem and facilitates advanced compiler optimizations and code transformations.

The MLIR representation is generated using `xDSL <https://xdsl.dev/>`_, a Python-based framework for building MLIR dialects and transformations. This allows Qrisp to seamlessly integrate quantum-classical hybrid computations into the MLIR infrastructure.

Basic Usage
-----------

Converting a Jasp-traced function to MLIR is straightforward using the :meth:`to_mlir` method on a :ref:`Jaspr <jaspr>` object:

::

    from qrisp import QuantumVariable, h, cx, measure, make_jaspr
    
    def quantum_circuit():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        result = measure(qv)
        return result
    
    # Create Jaspr object and convert to MLIR
    jaspr = make_jaspr(quantum_circuit)()
    mlir_module = jaspr.to_mlir()
    
    # The MLIR module is an xDSL ModuleOp object
    # Convert to string for inspection
    mlir_str = str(mlir_module)
    print(mlir_str)

This will produce MLIR code using the Jasp dialect operations such as ``jasp.create_qubits``, ``jasp.quantum_gate``, and ``jasp.measure``.

.. note::

    Extracting the Catalyst MLIR dialect is still possible using ``jaspr.to_catalyst_mlir()``.


Advanced Example: QAOA
-----------------------

The MLIR interface supports complex quantum-classical hybrid algorithms. Here's an example with QAOA (Quantum Approximate Optimization Algorithm):

::

    from qrisp import QuantumVariable, make_jaspr
    from qrisp.qaoa import QAOAProblem, RX_mixer, create_maxcut_cost_operator, create_maxcut_sample_array_post_processor
    import networkx as nx
    
    def qaoa_maxcut():
        # Create a graph for the MaxCut problem
        G = nx.erdos_renyi_graph(6, 0.7, seed=133)
        
        # Set up QAOA problem
        cl_cost = create_maxcut_sample_array_post_processor(G)
        qarg = QuantumVariable(G.number_of_nodes())
        
        qaoa = QAOAProblem(
            cost_operator=create_maxcut_cost_operator(G),
            mixer=RX_mixer,
            cl_cost_function=cl_cost
        )
        
        # Run QAOA optimization
        results = qaoa.run(qarg, depth=5, max_iter=50, optimizer="SPSA")
        return results
    
    # Generate MLIR representation
    jaspr = make_jaspr(qaoa_maxcut)()
    mlir_module = jaspr.to_mlir()

The generated MLIR will include quantum operations from the Jasp dialect as well as control flow constructs (such as ``scf.while`` for the optimization loop) that have been lowered from JAX's representation.


JASP Dialect Specification
---------------------------

The Jasp dialect defines a set of operations and types specifically designed for expressing quantum computations with classical control flow. Below is the complete specification of all operations and types in the dialect.

Types
^^^^^

The Jasp dialect defines three core types:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Type
     - Description
   * - ``QuantumState``
     - An opaque type describing the quantum state of the machine. This object is passed around the program to capture quantum computations in a functional style. Each quantum operation takes a ``QuantumState`` as input and produces a new ``QuantumState`` as output.
   * - ``Qubit``
     - A type describing an individual qubit. Qubit objects are semantically identical to integers as they simply index into the ``QuantumState``. This especially implies that it is semantically well-defined to copy a qubit reference.
   * - ``QubitArray``
     - A type describing a dynamically sized collection of qubits. QubitArrays enable expression of dynamically sized quantum programs. They are semantically equivalent to immutable arrays of integers representing qubit indices.


Operations
^^^^^^^^^^

Quantum State Management
"""""""""""""""""""""""""

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Operation
     - Description
   * - ``create_quantum_kernel``
     - Creates a quantum state from nothing. Indicates to the execution environment that a quantum computation will start.
       
       **Arguments:** None
       
       **Results:** ``QuantumState``
   * - ``consume_quantum_kernel``
     - Destroys the quantum state. Indicates to the execution environment that the quantum computation has concluded.
       
       **Arguments:** ``QuantumState``
       
       **Results:** ``tensor<i1>`` (success indicator)


Qubit Allocation and Management
""""""""""""""""""""""""""""""""

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Operation
     - Description
   * - ``create_qubits``
     - Allocates a ``QubitArray`` containing *n* qubits. The number of qubits can be dynamically sized.
       
       **Arguments:** ``tensor<i64>`` (amount), ``QuantumState`` (input state)
       
       **Results:** ``QubitArray``, ``QuantumState`` (output state)
   * - ``delete_qubits``
     - Deallocates qubits from a ``QubitArray``. Indicates to the execution environment that the corresponding qubits can be reused.
       
       **Arguments:** ``QubitArray``, ``QuantumState`` (input state)
       
       **Results:** ``QuantumState`` (output state)


Qubit Array Operations
""""""""""""""""""""""

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Operation
     - Description
   * - ``get_qubit``
     - Retrieves a single qubit from a given ``QubitArray`` at a specified position.
       
       **Arguments:** ``QubitArray``, ``tensor<i64>`` (position)
       
       **Results:** ``Qubit``
   * - ``get_size``
     - Gets the size of a ``QubitArray``. Returns the number of qubits in the array.
       
       **Arguments:** ``QubitArray``
       
       **Results:** ``tensor<i64>`` (size)
   * - ``slice``
     - Slices a ``QubitArray`` to extract a subset of qubits using start and end indices.
       
       **Arguments:** ``QubitArray``, ``tensor<i64>`` (start), ``tensor<i64>`` (end)
       
       **Results:** ``QubitArray``
   * - ``fuse``
     - Concatenates two qubits or qubit arrays. Fuses two ``QubitArrays``, ``Qubits``, or combinations thereof to create a larger ``QubitArray``.
       
       **Arguments:** ``Qubit`` or ``QubitArray`` (operand1), ``Qubit`` or ``QubitArray`` (operand2)
       
       **Results:** ``QubitArray``


Quantum Operations
""""""""""""""""""

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Operation
     - Description
   * - ``quantum_gate``
     - The quantum gate operation. Enables quantum processing of quantum states with (parametric) gates. The gate type is specified as a string attribute, and operands can include both parameters (``tensor<f64>``) and qubits (``Qubit``).
       
       **Arguments:** ``string`` (gate_type), variadic ``tensor<f64>`` or ``Qubit`` (gate_operands), ``QuantumState`` (input state)
       
       **Results:** ``QuantumState`` (output state)
       
       **Example gate types:** "h", "x", "y", "z", "rx", "ry", "rz", "cx", "cz", "rzz", etc.
   * - ``measure``
     - Performs a measurement of a given quantum state on a specified qubit or qubit array.
       
       **Arguments:** ``Qubit`` or ``QubitArray`` (qubits to measure), ``QuantumState`` (input state)
       
       **Results:** ``tensor<i1>`` or ``tensor<i64>`` (measurement result), ``QuantumState`` (output state)
   * - ``reset``
     - Resets qubits to the :math:`\ket{0}` state. Performs a reset operation on a single qubit or qubit array, returning them to the ground state.
       
       **Arguments:** ``Qubit`` or ``QubitArray`` (qubits to reset), ``QuantumState`` (input state)
       
       **Results:** ``QuantumState`` (output state)


Implementation Notes
--------------------

StableHLO Dialect for Classical Computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All classical computations in the generated MLIR are represented through the `StableHLO dialect <https://github.com/openxla/stablehlo>`_, which serves as the MLIR equivalent of JAX. StableHLO is a portability layer for machine learning frameworks, providing a stable set of operations that can be compiled to various hardware backends.

When Jasp-traced functions are converted to MLIR, the JAX computations are lowered to StableHLO operations, which include:

- Arithmetic operations (``stablehlo.add``, ``stablehlo.multiply``, etc.)
- Tensor operations (``stablehlo.reshape``, ``stablehlo.broadcast_in_dim``, etc.)
- Comparison and logical operations (``stablehlo.compare``, ``stablehlo.and``, etc.)

This approach ensures seamless integration between classical JAX code and quantum operations within a unified MLIR representation.

Control Flow Rewriting
^^^^^^^^^^^^^^^^^^^^^^

While StableHLO provides control flow operations, these constructs are designed exclusively for classical tensor types and cannot directly handle quantum types (``QuantumState``, ``Qubit``, ``QubitArray``). To address this limitation, the MLIR generation process includes automatic rewriting of StableHLO control flow operations to SCF (Structured Control Flow) dialect operations when they involve quantum types.

The SCF dialect provides generic control flow constructs that can work with arbitrary MLIR types, making it suitable for quantum-classical hybrid computations. The rewriting process preserves the semantics of the control flow while enabling proper representation of quantum operations.

For example:
- ``stablehlo.while`` → ``scf.while`` (when loop carries quantum-typed values)
- ``stablehlo.case`` → ``scf.if`` (when branches involve quantum-typed conditionals)

This transformation ensures that quantum operations are properly integrated within standard control flow constructs, enabling complex quantum-classical hybrid algorithms to be expressed naturally in MLIR.

xDSL Integration
^^^^^^^^^^^^^^^^

The MLIR output is generated using the xDSL framework, which provides:

- Python-native MLIR dialect definitions
- Flexible transformation and rewriting infrastructure  
- Integration with the broader MLIR ecosystem
- Easy extensibility for custom optimizations

The :meth:`to_mlir` method returns an ``xdsl.ir.ModuleOp`` object, which can be:

- Converted to a string using ``str(mlir_module)`` for inspection
- Further transformed using xDSL passes
- Exported to standard MLIR format
- Integrated with other MLIR-based compilation pipelines

