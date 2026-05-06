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

::

    from qrisp import PassManager, CircuitPass
    from qrisp import cancel_inverses, commute_swaps, combine_single_qubit_gates

    pm = PassManager()
    pm.add_pass(cancel_inverses)
    pm.add_pass(commute_swaps)
    pm.add_pass(combine_single_qubit_gates)

    optimized_qc = pm.run(qc)

    # Verify the whole pipeline
    results = pm.verify(qc, "measurements")

Built-in Passes
---------------

Qrisp ships with the following circuit transformation passes:

.. list-table::
   :header-rows: 1

   * - Pass
     - Description
   * - :doc:`arange_swaps <Passes/arange_swaps>`
     - Rearrange SWAP gates for better cancellation later
   * - :doc:`cancel_inverses <Passes/cancel_inverses>`
     - Cancel adjacent gate–inverse-gate pairs via DAG analysis
   * - :doc:`cancel_zero_controls <Passes/cancel_zero_controls>`
     - Remove gates controlled on \|0⟩ states
   * - :doc:`combine_single_qubit_gates <Passes/combine_single_qubit_gates>`
     - Fuse consecutive single-qubit gates into one
   * - :doc:`commute_swaps <Passes/commute_swaps>`
     - Commute single-qubit ops past SWAP gates
   * - :doc:`convert_to_cz <Passes/convert_to_cz>`
     - Convert two-qubit gates (CX, CY, SWAP) to CZ-based form
   * - :doc:`gray_synth_toffoli <Passes/gray_synth_toffoli>`
     - Synthesize Toffoli gates using Gray-code decomposition
   * - :doc:`is_toffoli <Passes/is_toffoli>`
     - Check whether an instruction is a Toffoli gate
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
