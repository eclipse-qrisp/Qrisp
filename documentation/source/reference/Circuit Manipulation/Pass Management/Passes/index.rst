.. _Passes:

Built-in Passes
===============

.. currentmodule:: qrisp

Qrisp ships with the following circuit transformation passes. Each pass is a
:class:`~qrisp.CircuitPass` that transforms a
:class:`~qrisp.QuantumCircuit` and can be used standalone or as part of a
:class:`~qrisp.PassManager` pipeline.

.. list-table::
   :header-rows: 1

   * - Pass
     - Description
   * - :doc:`arrange_swaps <arrange_swaps>`
     - Rearrange SWAP gates for better cancellation later
   * - :doc:`fuse_adjacents <fuse_adjacents>`
     - Cancel adjacent gate–inverse-gate pairs via DAG analysis
   * - :doc:`cancel_zero_controls <cancel_zero_controls>`
     - Remove gates controlled on \|0⟩ states
   * - :doc:`combine_single_qubit_gates <combine_single_qubit_gates>`
     - Fuse consecutive single-qubit gates into one
   * - :doc:`commute_swaps <commute_swaps>`
     - Commute single-qubit ops past SWAP gates
   * - :doc:`convert_to_cx <convert_to_cx>`
     - Convert two-qubit gates (CZ, CY, SWAP) to CX-based form
   * - :doc:`convert_to_cz <convert_to_cz>`
     - Convert two-qubit gates (CX, CY, SWAP) to CZ-based form
   * - :doc:`convert_to_prx <convert_to_prx>`
     - Convert single-qubit gates to native PRX (Phased-RX) gates
   * - :doc:`decompose <decompose>`
     - Recursively dissolve synthesized gates into elementary gates
   * - :doc:`gray_synth_toffoli <gray_synth_toffoli>`
     - Synthesize Toffoli gates using Gray-code decomposition
   * - :doc:`manual_layout <manual_layout>`
     - Re-index qubits according to a user-supplied mapping
   * - :doc:`remove_barriers <remove_barriers>`
     - Remove barrier instructions from the circuit
   * - :doc:`resolve_swaps <resolve_swaps>`
     - Resolve SWAP gates by physically permuting qubits
   * - :doc:`reverse_parallelize <reverse_parallelize>`
     - Reverse-parallelize the circuit for reuse in conjugate
   * - :doc:`visualize <visualize>`
     - Print the circuit to stdout for debugging

.. toctree::
   :maxdepth: 1
   :hidden:

   arrange_swaps
   fuse_adjacents
   cancel_zero_controls
   combine_single_qubit_gates
   commute_swaps
   convert_to_cx
   convert_to_cz
   convert_to_prx
   decompose
   gray_synth_toffoli
   manual_layout
   remove_barriers
   resolve_swaps
   reverse_parallelize
   visualize
