.. _qml_reference:

Quantum Machine Learning
========================

The ``qrisp.qml`` module provides the building blocks of a quantum machine
learning model. A model is made up of an encoding that writes the features into
a register, an ansatz with trainable parameters, and one or more observables
that read out the result. Each part is a separate object, and any of them can
be replaced without touching the others. Encodings and ansatzes also work under
Jasp tracing, which means a whole training loop can be compiled.

The :doc:`QML tutorial </general/tutorial/QMLtutorial>` builds and trains a
model from these parts, and the
:doc:`transfer learning tutorial </general/tutorial/QMLTransferLearning>`
places one inside a larger classical network.

Building a circuit
------------------

.. list-table::
   :header-rows: 0
   :widths: 30 70

   * - :ref:`Encoding`
     - Writes classical features into the qubits of a register.
   * - :ref:`Ansatz`
     - Holds the parameterized circuit that a model trains.

.. list-table::
   :header-rows: 0
   :widths: 30 70

   * - :func:`~qrisp.qml.hardware_efficient_size`
     - Returns the number of parameters of a hardware efficient ansatz.
   * - :func:`~qrisp.qml.real_amplitudes_size`
     - Returns the number of parameters of a real amplitudes ansatz.

Reading out and differentiating
-------------------------------

.. list-table::
   :header-rows: 0
   :widths: 30 70

   * - :func:`~qrisp.qml.expectation`
     - Estimates the expectation values of one or more observables.
   * - :func:`~qrisp.qml.parameter_shift`
     - Estimates expectation values together with their derivatives, using the parameter shift rule.

Preparing features
------------------

.. list-table::
   :header-rows: 0
   :widths: 30 70

   * - :func:`~qrisp.qml.minmax_scale`
     - Scales features into an interval suitable for an encoding.
   * - :func:`~qrisp.qml.normalize`
     - Scales each sample to unit norm for amplitude encoding.
   * - :func:`~qrisp.qml.binarize`
     - Turns features into the bit values basis encoding requires.

Saving and loading
------------------

.. list-table::
   :header-rows: 0
   :widths: 30 70

   * - :func:`~qrisp.qml.save_model`
     - Saves the parts of a model to a file.
   * - :func:`~qrisp.qml.load_model`
     - Loads the parts of a model saved with save_model.

.. toctree::
   :hidden:

   Encoding
   Ansatz
   hardware_efficient_size
   real_amplitudes_size
   expectation
   parameter_shift
   minmax_scale
   normalize
   binarize
   save_model
   load_model

