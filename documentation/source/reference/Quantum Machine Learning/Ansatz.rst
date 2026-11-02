.. _Ansatz:

Ansatz
======

.. currentmodule:: qrisp.qml
.. autoclass:: Ansatz

Constructors
------------

.. list-table::
   :header-rows: 0
   :widths: 30 70

   * - :meth:`~qrisp.qml.Ansatz.hardware_efficient`
     - Returns an ansatz of rotation layers separated by entangling layers.
   * - :meth:`~qrisp.qml.Ansatz.real_amplitudes`
     - Returns an ansatz preparing states with real amplitudes.
   * - :meth:`~qrisp.qml.Ansatz.data_reuploading`
     - Returns an ansatz alternating encoding and ansatz layers.

.. toctree::
   :hidden:

   methods/Ansatz.hardware_efficient
   methods/Ansatz.real_amplitudes
   methods/Ansatz.data_reuploading

Usage
-----

.. list-table::
   :header-rows: 0
   :widths: 30 70

   * - :meth:`~qrisp.qml.Ansatz.apply`
     - Applies the ansatz to a QuantumVariable.

.. toctree::
   :hidden:

   methods/Ansatz.apply

Subclassing
-----------

.. list-table::
   :header-rows: 0
   :widths: 30 70

   * - :meth:`~qrisp.qml.Ansatz.apply_gates`
     - Emits the gates of the ansatz. A subclass overrides it.
   * - :meth:`~qrisp.qml.Ansatz.parameter_count`
     - Returns the number of parameters the ansatz expects. A subclass overrides it.

.. toctree::
   :hidden:

   methods/Ansatz.apply_gates
   methods/Ansatz.parameter_count

