.. _PassManagement:

Pass Management
===============

The pass management module provides a structured framework for applying
sequential circuit transformations in Qrisp. It consists of two core
components:

* :doc:`CircuitPass` — A decorator/wrapper that enforces type safety on
  ``QuantumCircuit → QuantumCircuit`` transformation functions and provides
  built-in verification capabilities (unitary comparison and measurement
  statistics comparison).

* :doc:`PassManager` — An ordered pipeline that chains
  :class:`~qrisp.CircuitPass` instances and applies them sequentially.
  Supports pass insertion, removal, iteration, and bulk verification.

Together they allow users to compose and validate compilation workflows in a
clean, reusable fashion:

Basic Pipeline
~~~~~~~~~~~~~~

Build a compilation pipeline by chaining passes together::

    from qrisp import QuantumCircuit, PassManager
    from qrisp import cancel_inverses, commute_swaps, combine_single_qubit_gates

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.cx(0, 1)   # Self-inverse — will be cancelled
    qc.h(0)
    qc.h(0)        # Another self-inverse pair

    print("Before:", qc, sep="\n")

.. code-block:: none

   Before:
                    ┌───┐┌───┐
    qb_0: ──■────■──┤ H ├┤ H ├
          ┌─┴─┐┌─┴─┐└───┘└───┘
    qb_1: ┤ X ├┤ X ├──────────
          └───┘└───┘          

::

    pm = PassManager()
    pm += cancel_inverses
    pm += commute_swaps
    pm += combine_single_qubit_gates

    optimized_qc = pm.run(qc)
    print("After:", optimized_qc, sep="\n")

.. code-block:: none

   After:
           
    qb_0: 
           
    qb_1: 

::

    # Empty circuit — all gates cancelled

Visualizing Pass Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every :class:`~qrisp.CircuitPass` provides a :meth:`~qrisp.CircuitPass.visualize`
method that prints a before/after comparison of the circuit::

    from qrisp import cancel_inverses

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.cx(0, 1)
    cancel_inverses.visualize(qc)

.. code-block:: none

   ====================  cancel_inverses  =====================
   ────────────────────────── Before ──────────────────────────
                      
    qb_0: ──■────■──
          ┌─┴─┐┌─┴─┐
    qb_1: ┤ X ├┤ X ├
          └───┘└───┘
   ────────────────────────── After ───────────────────────────
            
    qb_0: 
            
    qb_1: 
            
   ============================================================

This works with any pass — ideal for debugging transformations::

    from qrisp import QuantumCircuit
    from qrisp import convert_to_cz

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    convert_to_cz().visualize(qc)

.. code-block:: none

   =====================  _convert_to_cz  =====================
   ────────────────────────── Before ──────────────────────────
                 
    qb_0: ──■──
          ┌─┴─┐
    qb_1: ┤ X ├
          └───┘
   ────────────────────────── After ───────────────────────────
                         
    qb_0: ──────■──────
          ┌───┐ │ ┌───┐
    qb_1: ┤ H ├─■─┤ H ├
          └───┘   └───┘
   ============================================================

Verification
~~~~~~~~~~~~

Verify that each pass preserves the circuit's unitary or measurement
statistics::

    pm = PassManager()
    pm += convert_to_cz()
    pm += cancel_inverses
    pm += combine_single_qubit_gates

    qc = QuantumCircuit(2)
    qc.rx(0.4, 0)
    qc.rz(0.2, 1)
    qc.cx(0, 1)

    # Check that every pass preserves the unitary
    results = pm.verify(qc, "unitary", ignore_gphase=True)
    for pass_name, passed in results:
        print(f"{pass_name}: {'✓' if passed else '✗'}")

Standalone Passes
~~~~~~~~~~~~~~~~~

Passes can also be used directly without a PassManager::

    from qrisp import cancel_inverses

    qc = QuantumCircuit(1)
    qc.x(0)
    qc.x(0)
    qc.y(0)

    optimized_qc = cancel_inverses(qc)
    # optimized_qc now contains only a Y gate

Targeting Native Gate Sets
~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine passes to convert circuits to hardware-native gate sets::

    from qrisp import remove_barriers

    qc = QuantumCircuit(3)
    qc.swap(0, 1)
    qc.cx(1, 2)
    qc.barrier()

    print("Before:", qc, sep="\n")

.. code-block:: none

   Before:
                   ░ 
    qb_0: ─X───────░─
           │       ░ 
    qb_1: ─X───■───░─
             ┌─┴─┐ ░ 
    qb_2: ───┤ X ├─░─
             └───┘ ░ 

::

    pm = PassManager()
    pm += convert_to_cz()           # CX → CZ + single-qubit gates
    pm += combine_single_qubit_gates # Fuse adjacent single-qubit gates
    pm += remove_barriers            # Remove scheduling barriers

    hw_ready_qc = pm.run(qc)
    print("After:", hw_ready_qc, sep="\n")

.. code-block:: none

   After:
                  ┌───┐   ┌───┐                
    qb_0: ──────■─┤ H ├─■─┤ H ├─■──────────────
          ┌───┐ │ ├───┤ │ ├───┤ │ ┌───┐        
    qb_1: ┤ H ├─■─┤ H ├─■─┤ H ├─■─┤ H ├─■──────
          ├───┤   └───┘   └───┘   └───┘ │ ┌───┐
    qb_2: ┤ H ├─────────────────────────■─┤ H ├
          └───┘                           └───┘


Built-in Passes
---------------

Qrisp ships with the following circuit transformation passes:

.. list-table::
   :header-rows: 1

   * - Pass
     - Description
   * - :doc:`arrange_swaps <Passes/arrange_swaps>`
     - Rearrange SWAP gates for better cancellation later
   * - :doc:`cancel_inverses <Passes/cancel_inverses>`
     - Cancel adjacent gate–inverse-gate pairs via DAG analysis
   * - :doc:`cancel_zero_controls <Passes/cancel_zero_controls>`
     - Remove gates controlled on \|0⟩ states
   * - :doc:`combine_single_qubit_gates <Passes/combine_single_qubit_gates>`
     - Fuse consecutive single-qubit gates into one
   * - :doc:`commute_swaps <Passes/commute_swaps>`
     - Commute single-qubit ops past SWAP gates
   * - :doc:`convert_to_cx <Passes/convert_to_cx>`
     - Convert two-qubit gates (CZ, CY, SWAP) to CX-based form
   * - :doc:`convert_to_cz <Passes/convert_to_cz>`
     - Convert two-qubit gates (CX, CY, SWAP) to CZ-based form
   * - :doc:`decompose <Passes/decompose>`
     - Recursively dissolve synthesized gates into elementary gates
   * - :doc:`gray_synth_toffoli <Passes/gray_synth_toffoli>`
     - Synthesize Toffoli gates using Gray-code decomposition
   * - :doc:`manual_layout <Passes/manual_layout>`
     - Re-index qubits according to a user-supplied mapping
   * - :doc:`remove_barriers <Passes/remove_barriers>`
     - Remove barrier instructions from the circuit
   * - :doc:`resolve_swaps <Passes/resolve_swaps>`
     - Resolve SWAP gates by physically permuting qubits
   * - :doc:`reverse_parallelize <Passes/reverse_parallelize>`
     - Reverse-parallelize the circuit for reuse in conjugate

.. toctree::
   :maxdepth: 3
   :hidden:

   CircuitPass
   PassManager
   Passes/index
