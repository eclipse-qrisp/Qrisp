.. _SimulatorBackends:

Simulator Backends
==================

.. toctree::
   :maxdepth: 1
   :hidden:

   QrispSimulatorBackend
   StimBackend

Local simulator backends that execute circuits without requiring external
hardware or network access. Note that :ref:`QiskitBackend` can also be used 
to integrate Qiskit compatible simulator backends into Qrisp.

.. list-table::
   :header-rows: 1

   * - Backend
     - Description
   * - :ref:`QrispSimulatorBackend`
     - built-in statevector simulator (the default)
   * - :ref:`StimBackend`
     - high-performance Clifford simulator backed by Stim
