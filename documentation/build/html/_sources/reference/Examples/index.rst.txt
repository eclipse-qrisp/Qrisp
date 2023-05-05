Examples
========

In this section, we provide a glimpse into the diverse range of applications that can be implemented using Qrisp. With these, we display how Qrisp provides a powerful and flexible platform for implementing and exploring quantum computing applications. These examples are designed to help you understand the capabilities of our language and inspire you to develop your own quantum computing applications.

.. list-table:: Title
   :widths: 25 25
   :header-rows: 1

   * - Title
     - Description
   * - :ref:`AbstractParameters`
     - This example showcases how parametrized circuits can be generated and processed by the Qrisp infrastructure.
   * - :ref:`ExactGrover`
     - Demonstrates how to utilize the ``exact`` keyword of :meth:`grovers_alg <qrisp.grover.grovers_alg>`.
   * - :ref:`DiagonalHamiltonianApplication`
     - An example to demonstrate how to utilize the ``as_hamiltonian`` decorator.
   * - :ref:`HelloWorld`
     - An example to demonstrate the use of :ref:`QuantumStrings <QuantumString>` in the form of the well known "Hello world" script.
   * - :ref:`InplaceMatrixMultiplication`
     - Showcases the use of the :meth:`qrisp.inplace_matrix_app` to apply an invertible classical matrix inplace to a quantum vector.
   * - :ref:`Loops`
     - Illustrates loops with quantum bounds using the ``qrange`` iterator.
   * - :ref:`MatrixMultiplication`
     - Exemplifies how to multiply quantum matrices using the :meth:`qrisp.dot` function.
   * - :ref:`QuantumModDivision`
     - Exhibits how the :meth:`qrisp.q_divmod` function can be utilize to perform division with remainder.
   * - :ref:`QuantumTeleportation`
     - An example on how to use Qrisps QuantumNetwork module to simulate a quantum teleportation.
   * - :ref:`EfficientTSP`
     - A more efficient version of the solution of the traveling salesman problem, that was presented in the :ref:`Tutorial`
                    
        

.. toctree::
   :hidden:
   :maxdepth: 2

   AbstractParameter
   ExactGrover
   DiagonalHamiltonianApplication
   HelloWorld
   InplaceMatrixMultiplication
   Loops
   MatrixMultiplication
   QuantumModDivision
   QuantumTeleportation
   EfficientTSP