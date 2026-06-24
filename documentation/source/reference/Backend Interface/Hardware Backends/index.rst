.. _HardwareBackends:

Hardware Backends
=================

.. toctree::
   :maxdepth: 1
   :hidden:

   IQMBackend
   QiskitBackend
   AQTBackend
   QiskitRuntimeBackend
   KipuBackends

Backends that interface with physical quantum hardware from different vendors.
Each backend wraps the vendor-specific API and exposes it through Qrisp's
uniform :ref:`Backend` interface.

.. list-table::
   :header-rows: 1

   * - Backend
     - Description
   * - :ref:`IQMBackend`
     - IQM quantum computers
   * - :ref:`QiskitBackend`
     - IBM Quantum backends via Qiskit
   * - :ref:`AQTBackend`
     - AQT (Alpine Quantum Technologies) backends
   * - :ref:`QiskitRuntimeBackend`
     - IBM Quantum runtime services via Qiskit Runtime
   * - :ref:`KipuBackends`
<<<<<<< HEAD
     - Kipu Quantum Cloud backends from multiple providers
=======
     - Kipu Quantum Cloud backend providers continuously expanding
>>>>>>> 49d80204 (feature: Kipu Quantum Cloud doc .rst files added)
