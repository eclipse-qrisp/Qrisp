.. _IQMBackend:

IQMBackend
==========

.. currentmodule:: qrisp.interface

Qrisp backend for executing circuits on IQM quantum hardware.

Install with ``pip install qrisp[iqm]``.

Full API documentation:
`IQMBackend <https://docs.iqm.tech/sdk4_6/iqm-client/api/qrisp_iqm/iqm.qrisp_iqm.backends.IQMBackend.html>`_

Quickstart
----------

.. code-block:: python

   from qrisp.interface import IQMBackend

   backend = IQMBackend(
       token="YOUR_API_TOKEN",
       device_instance="garnet"
   )

   from qrisp import QuantumCircuit
   qc = QuantumCircuit(2)
   qc.h(0)
   qc.cx(0, 1)
   qc.measure(qc.qubits)
   result = backend.run(qc, shots=1000)
