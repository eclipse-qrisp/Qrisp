.. _VirtualBackend:

VirtualBackend
==============

.. currentmodule:: qrisp.interface


Options
-------
From ``qrisp.interface`` is it possible to import the following kind of backends.

.. csv-table::
   :header: "Option", "Description"
   :widths: 20, 60

   ":ref:`VirtualQiskitBackend`", "Allows to use a Qiskit Aer backend."
   ":ref:`QiskitRuntimeBackend`", "Allows to use a Qiskit Runtime backend, exploiting the Qiskit's Sampler primitive."
   ":ref:`IQMBackend`", "Allows to use an IQM quantum computer as a backend via IQM Resonance."


.. autoclass:: VirtualBackend

Methods
=======

.. autosummary::
   :toctree: generated/

   VirtualBackend.__init__
