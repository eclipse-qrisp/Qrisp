.. _AbstractBackendInterface:

Abstract Backend Interface
==========================

.. toctree::
   :maxdepth: 1
   :hidden:

   Backend
   MeasurementResult
   Job
   JobResult
   JobStatus

The core abstractions that define Qrisp's backend execution model.
Every concrete backend (hardware or simulator) implements the
:ref:`Backend` base class and returns :ref:`Job` handles.

.. list-table::
   :header-rows: 1

   * - Class
     - Description
   * - :ref:`Backend`
     - abstract base class for all backends
   * - :ref:`MeasurementResult`
     - dict-like container for measurement outcomes
   * - :ref:`Job`
     - handle representing a submitted circuit execution
   * - :ref:`JobResult`
     - container for raw execution results
   * - :ref:`JobStatus`
     - enumeration of job lifecycle states
