.. _Backend:

Backend
=======

.. currentmodule:: qrisp.interface
.. autoclass:: Backend



Methods
=======

.. autosummary::
   :toctree: generated/

   Backend.__init__
   Backend.run



Runtime options
----------------

.. autosummary::
   :toctree: generated/

   Backend.options
   Backend.update_options



Hardware status information
---------------------------

.. autosummary::
   :toctree: generated/

   Backend.backend_health
   Backend.backend_info
   Backend.backend_queue



Hardware metadata
-----------------

.. autosummary::
   :toctree: generated/

   Backend.num_qubits
   Backend.connectivity
   Backend.gate_set
   Backend.error_rates
   Backend.capabilities