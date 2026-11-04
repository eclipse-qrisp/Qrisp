.. _Encoding:

Encoding
========

.. currentmodule:: qrisp.qml
.. autoclass:: Encoding

Constructors
------------

.. list-table::
   :header-rows: 0
   :widths: 30 70

   * - :meth:`~qrisp.qml.Encoding.angle`
     - Returns an encoder placing one feature in each qubit's rotation angle.
   * - :meth:`~qrisp.qml.Encoding.amplitude`
     - Returns an encoder placing the features in a register's amplitudes.
   * - :meth:`~qrisp.qml.Encoding.dense_angle`
     - Returns an encoder placing two features in each qubit.
   * - :meth:`~qrisp.qml.Encoding.basis`
     - Returns an encoder placing one bit in each qubit.

.. toctree::
   :hidden:

   methods/Encoding.angle
   methods/Encoding.amplitude
   methods/Encoding.dense_angle
   methods/Encoding.basis

Usage
-----

.. list-table::
   :header-rows: 0
   :widths: 30 70

   * - :meth:`~qrisp.qml.Encoding.apply`
     - Encodes one sample into a QuantumVariable.

.. toctree::
   :hidden:

   methods/Encoding.apply

Subclassing
-----------

.. list-table::
   :header-rows: 0
   :widths: 30 70

   * - :meth:`~qrisp.qml.Encoding.encode`
     - Emits the gates encoding one sample. A subclass overrides it.

.. toctree::
   :hidden:

   methods/Encoding.encode

