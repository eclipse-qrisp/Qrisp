.. _block_encodings:


Block Encodings
---------------


.. image:: /_static/block_encodings.png
    :align: center
    :width: 600


.. toctree::
   :hidden:
   
   BlockEncoding
   block_encodings_applications
   block_encodings_nisq


A High-Level Abstraction for Quantum Linear Algebra
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Qrisp's :ref:`BlockEncoding` provides a powerful programming abstraction for **Quantum Linear Algebra**. 
While traditional quantum programming requires specialized knowledge of quantum algorithms and the intricate use of superposition and entanglement, 
this interface allows users to program quantum computers through the language of linear algebra. 
By leveraging matrix-based abstractions, it makes quantum computing accessible to domain experts in fields like classical machine learning, 
scientific computing, and computational dynamics who are already fluent in these mathematical frameworks.


NumPy-like syntax
^^^^^^^^^^^^^^^^^

Let us consider two Hermitian matrices $A$ and $B$ and evaluate the matrix expression 

.. math::
    
        C = I + A - 2A^2 + B^{-1}
        
applied to a vector $\vec{b}$. Frist, we define the matrices as NumPy arrays and their respective :ref:`BlockEncoding` representations.

:: 

    # For larger systems, restart the kernel and adjust simulator precision
    # import os
    # os.environ["QRISP_SIMULATOR_FLOAT_THRESH"] = "1e-10"
    import numpy as np
    from qrisp import *
    from qrisp.block_encodings import BlockEncoding

    A = np.array([[ 0.66,  0.02, -0.11, -0.16],
        [ 0.02,  0.82,  0.01, -0.12],
        [-0.11,  0.01,  0.93, -0.07],
        [-0.16, -0.12, -0.07,  0.69]])

    B = np.array([[ 0.78, -0.01, -0.16, -0.1 ],
        [-0.01,  0.57, -0.03,  0.08],
        [-0.16, -0.03,  0.69, -0.15],
        [-0.1 ,  0.08, -0.15,  0.88]])

    b = np.array([1., 2., 1., 1.])

    epsilon = 0.001
    kappa = np.linalg.cond(B)

    B_A = BlockEncoding.from_array(A)
    B_B = BlockEncoding.from_array(B)

Next, we evalute the matrix expression using **NumPy** and **Qrisp**:

::

    C = np.eye(4) + A + 2 * A @ A + np.linalg.inv(B) # NumPy

    B_C = B_A.poly([1.,1.,-2.]) + B_B.inv(epsilon, kappa) # Qrisp

Notably, expressing high-level (quantum) linear algebra in Qrisp requires no prior knowledge of quantum computing, 
thanks to its intuitive, NumPy-like syntax.

Finally, we compare the results, demonstrating that the implementation requires only a minimal amount of quantum-specific code.

::

    # Normalize for comparison to quantum solution
    c = C @ b / np.linalg.norm(C @ b)
    print("numpy: ", c)
    # numpy: [0.44707785 0.66049416 0.41341197 0.43927144]

:: 

    # Prepare state |b>
    def prep_b():
        qv = QuantumFloat(2)
        prepare(qv, b)
        return qv

    @terminal_sampling
    def main():
        # apply_rus applies the Hermitian operator C to state |b> 
        # using a repeat-until-success protocol
        return B_C.apply_rus(prep_b)()

    res_dict = main()

    # Convert measurement probabilites to (absolute values of) amplitudes
    amps = np.sqrt([res_dict.get(i, 0) for i in range(len(b))])
    print("qrisp:", amps)
    # qrisp: [0.44630489 0.66211377 0.41182956 0.43910558]

The results agree within the approximation error of the quantum matrix inversion. 
Note that the sign of the output cannot be directly recovered from measurement amplitudes alone, 
as they only reflect the modulus squared of the state.