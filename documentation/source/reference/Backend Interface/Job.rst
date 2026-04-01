.. _Job:

Job
===

.. currentmodule:: qrisp.interface

.. autoclass:: Job

Properties
----------

.. autosummary::
   :toctree: generated/

   Job.job_id
   Job.backend

Abstract Methods
----------------

.. autosummary::
   :toctree: generated/

   Job.submit
   Job.result
   Job.cancel
   Job.status

Methods
-------

.. autosummary::
   :toctree: generated/

   Job.__init__
   Job.done
   Job.running
   Job.queued
   Job.cancelled
   Job.in_final_state
   Job.refresh
   Job._raise_for_status