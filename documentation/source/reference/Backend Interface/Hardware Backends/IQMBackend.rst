.. _IQMBackend:

IQMBackend
==============

``IQMBackend`` is provided by the ``iqm-client`` package with Qrisp support.
Install it with:

.. code-block:: bash

   pip install qrisp[iqm]

Then import and use it as:

.. code-block:: python

   from qrisp.interface import IQMBackend

   quantum_computer = IQMBackend(
       api_token="your_api_token",
       device_instance="garnet",
   )

For full API documentation, refer to the ``iqm-client`` package documentation.

