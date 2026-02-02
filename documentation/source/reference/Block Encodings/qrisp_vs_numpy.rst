.. _qrisp_vs_numpy:

NumPy-like syntax
-----------------

Define two Hermitian matrices $A$ and $B$ and evaluate the matrix expression 
$C = I + 2A + A^2 + B^{-1}$ applied to a vector $\vec{b}$.

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

    kappa = np.linalg.cond(B)

    B_A = BlockEncoding.from_array(A)
    B_B = BlockEncoding.from_array(B)

    b = np.array([1., 1., 2., 0.])

Next, evaluate the matrix expression.

numpy
=====

::

    C = np.eye(4) + 2 * A + A @ A + np.linalg.inv(B) 

qrisp
=====

::

    B_C = B_A.poly(np.array([1.,2.,1.])) + B_B.inv(0.001, kappa)

Notably, expressing high-level quantum linear algebra in Qrisp requires no prior knowledge of quantum computing, thanks to its intuitive, NumPy-like syntax.

Results
=======

Finally, compare the results, which for Qrisp involves a small amount of quantum-specific code.

::

    # Normalize for comparison to quantum solution
    c = C @ b / np.linalg.norm(C @ b)
    print("numpy: ", c)
    # numpy:  [ 0.33242728  0.42117847  0.8418754  -0.05784993]

:: 

    # Prepare state |b>
    def prep_b():
        qv = QuantumFloat(2)
        prepare(qv, b)
        return qv

    @terminal_sampling
    def main():
        return B_C.apply_rus(prep_b)()

    res_dict = main()

    # Convert measurement probabilities to (absolute values of) amplitudes
    for k, v in res_dict.items():
        res_dict[k] = v**0.5

    q = np.array([res_dict.get(key, 0) for key in range(len(b))])
    print("qrisp:", q)
    # qrisp: [0.33181535 0.42225404 0.84150093 0.05896006]

The results agree within the approximation error of the quantum matrix inversion. 
Note that the sign of the output cannot be directly recovered from measurement amplitudes alone, as they only reflect the modulus squared of the state.