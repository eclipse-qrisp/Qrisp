Utilities
=========

.. toctree::
   :maxdepth: 2
   :hidden:

   stim_tools

.. list-table::
   :header-rows: 0
   :widths: 30 70

   * - :func:`~qrisp.multi_measurement`
     - Perform measurement on a list of QuantumVariables or a QuantumCircuit.
   * - :func:`~qrisp.batched_measurement`
     - Perform batched measurement on a list of QuantumVariables or a QuantumCircuit.
   * - :func:`~qrisp.gate_wrap`
     - Decorator to bundle up the quantum instructions of a function into a single gate object.
   * - :func:`~qrisp.custom_control`
     - Adds a custom control behavior to a function.
   * - :func:`~qrisp.lifted`
     - Decorator that indicates that a function computes a phase logic (i.e. is diagonal).
   * - :func:`~qrisp.redirect_qfunction`
     - Redirects a quantum function to a new target qubit.
   * - :func:`~qrisp.as_hamiltonian`
     - Decorator that converts a function returning a phase into a Hamiltonian.
   * - :func:`~qrisp.lock`
     - Context manager that locks a list of qubits, preventing them from being used for automatic allocation.
   * - :func:`~qrisp.unlock`
     - Context manager that unlocks a list of qubits, allowing them to be used for automatic allocation.
   * - :func:`~qrisp.perm_lock`
     - Context manager that permanently locks a list of qubits, preventing them from being used for automatic allocation.
   * - :func:`~qrisp.perm_unlock`
     - Context manager that permanently unlocks a list of qubits, allowing them to be used for automatic allocation.
   * - :func:`~qrisp.t_depth_indicator`
     - Returns the T-depth of a given operation.
   * - :func:`~qrisp.cnot_depth_indicator`
     - Returns the CNOT-depth of a given operation.
   * - :func:`~qrisp.inpl_adder_test`
     - Tests if an inplace adder is working correctly.

.. toctree::
   :hidden:
   
   multi_measurement
   batched_measurement
   gate_wrap
   custom_control
   lifted
   redirect_qfunction
   as_hamiltonian
   lock
   unlock
   perm_lock
   perm_unlock
   t_depth_indicator
   cnot_depth_indicator
   inpl_adder_test   